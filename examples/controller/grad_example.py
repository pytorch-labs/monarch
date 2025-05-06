import itertools

import torch
from monarch import DeviceMesh, local_mesh, Simulator, Tensor
from monarch.gradient_generator import grad_function, GradGenerator
from monarch.simulator import set_meta

torch.set_default_device("cuda")

NUM_GPUS = 4


def call_layer(w, x):
    return torch.mm(x, w)


def initialize(
    num_gpus, stages_per_gpu, hidden_dim=4, local_batch_size=4, use_real=False
):
    if use_real:
        device_mesh = local_mesh(hosts=1, gpus=num_gpus)
    else:
        device_mesh = Simulator(hosts=1, gpus=num_gpus).mesh
    meshes = [device_mesh(gpu=i) for i in range(num_gpus)]
    layers = [[] for _ in range(num_gpus)]
    for idx, mesh in enumerate(meshes):
        with mesh.activate():
            for _ in range(stages_per_gpu):
                layers[idx].append(
                    torch.randn(hidden_dim, hidden_dim, requires_grad=True)
                )

    def dataloader():
        with meshes[0].activate():
            # a great dataloader
            return torch.randn(local_batch_size, hidden_dim, requires_grad=True)

    return device_mesh, meshes, layers, dataloader


@grad_function
def to_mesh(x: Tensor, mesh: DeviceMesh):
    omesh = x.mesh

    def backward(grad_x: Tensor):
        print(grad_x.mesh, omesh)
        return grad_x.to_mesh(omesh), None

    return x.to_mesh(mesh), backward


# coroutine that runs each microbatch, yields after each stage,
# yield True to indicate phases are over.
# might be better to have it yield more describitive info each time.
def run_mb(mb_num, layers, dataloader):
    mb = dataloader()  # loads the microbatch
    grads = GradGenerator()
    for layer in layers:
        if mb.mesh != layer.mesh:
            grads.grad(mb)
            mb = to_mesh(mb, layer.mesh)
            yield
        with layer.mesh.activate(), set_meta("fw"), set_meta(str(mb_num)):
            # grad weight stage
            if mb.requires_grad:
                # grad input stage
                grads.grad((mb, layer))
            else:
                grads.grad(layer)

            mb = call_layer(layer, mb)
        yield

    yield True

    # could be a loop, but set_meta has to go around
    # the generator step so backward calc is recorded.
    grads.root(mb)
    it = iter(grads)
    while True:
        try:
            with set_meta("bw"), set_meta(str(mb_num)):
                # grad accumulation would go here,
                # if this were marked as the last microbatch
                # gradient steps might also go here.
                next(it)
            yield
        except StopIteration:
            break
    yield True


def fill_drain(num_microbatches, layers, dataloader):
    mbs = [run_mb(i, layers, dataloader) for i in range(num_microbatches)]
    for _phase in ["forward", "backward"]:
        for mb in mbs:
            while not next(mb):
                pass


def fill_drain_interleaved(num_microbatches, layers, dataloader):
    mbs = [run_mb(i, layers, dataloader) for i in range(num_microbatches)]
    for _phase in ["foward", "backward"]:
        finished = set()
        for timestep in itertools.count(0):
            if len(finished) == len(mbs):
                break
            for i, mb in enumerate(mbs):
                micro_batch_start_timestep = 2 * i
                # introduce the initial skew.
                # it is a skew of 2 for foward (step, send)
                # and 3 for backward (gradi, gradw, send)
                if timestep < micro_batch_start_timestep or i in finished:
                    continue
                if next(mb):
                    finished.add(i)


device_mesh, meshes, layers, dataloader = initialize(
    num_gpus=4, stages_per_gpu=2, local_batch_size=32
)

interleave = True
if interleave:
    stages_per_layer = len(layers[0])
    layers = [stage[i] for i in range(stages_per_layer) for stage in layers]
    fill_drain_interleaved(num_microbatches=4, layers=layers, dataloader=dataloader)
else:
    fill_drain(
        num_microbatches=4,
        layers=[la for s in layers for la in s],
        dataloader=dataloader,
    )
# TODO: @chienchin, these methods are deleted
# events = simulate_commands(device_mesh.client.backend.worker_commands)
# visualize_events(events)
# analyze_events(events)
