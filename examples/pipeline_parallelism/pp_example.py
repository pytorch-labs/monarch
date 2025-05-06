"""
Usage:
    case 1: run the example with pure Monarch Mode:
        Model components such as modules, optimizers, and loss layers are created
        by the controller directly. Consequently, parameters and tensors are
        maintained by the controller as Monarch tensors, with storage located on
        workers. The controller calls the forward and backward functions within
        the corresponding stage's mesh context, dispatching each operation to the
        appropriate mesh for execution.

        Command Line:
        python -m pipeline_parallelism.pp_example --mode=monarch

    case 2: run the example with UDF Mode:
        Model components such as modules, optimizers, and loss layers are created
        by user-defined functions (UDFs). These components are created on the
        worker and can be accessed as OpaqueRef on the worker only. The forward
        and backward functions are also created as UDFs, which access the related
        OpaqueRef module components and execute the entire function on the worker.
        The controller's role is limited to dispatching the whole UDF.

        Command Line:
        python -m pipeline_parallelism.pp_example --mode=udf

    case 3: run the example with OpequeMoule Mode:
        In OpequeMoule mode, model components are encapsulated as opaque modules. Optimizers
        and others are still defined as opaqueRef objects. All these components are created
        and managed by the controller. The parameters and tensors are stored as opaque
        references, allowing for efficient management and execution. The controller is
        responsible for orchestrating the forward and backward passes, ensuring that
        operations are executed seamlessly across the distributed system.

        Command Line:
        python -m pipeline_parallelism.pp_example --mode=opm
"""

# pyre-unsafe
from __future__ import annotations

import argparse

import logging

import os
import sys
from contextlib import nullcontext

import torch

import torch.nn as nn
import torch.optim as optim
from monarch import (
    fetch_shard,
    get_active_stream,
    local_mesh,
    no_mesh,
    OpaqueRef,
    remote,
    Simulator,
    Stream,
)
from monarch.opaque_module import OpaqueModule
from monarch.parallel import get_parameter_udf, PipelineParallelism
from monarch.profiler import profile, Schedule

from monarch_supervisor.logging import initialize_logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, device="cuda"):
        """
        Initializes the neural network with one hidden layer.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
            hidden_dim (int, optional): The number of neurons in the hidden layer. Defaults to 128.
        """
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, input_tensor):
        """
        Defines the forward pass of the network.

        Args:
            input_tensor (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = torch.relu(self.fc1(input_tensor))
        x = self.fc2(x)
        return x


def make_input_output(batch_size, input_dim, output_dim, num_batches=10):
    """
    Generates random input and output tensors for a given number of batches.

    Args:
        batch_size (int): The number of samples in each batch.
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        num_batches (int, optional): The number of batches to generate. Defaults to 10.

    Returns:
        tuple: A tuple containing two lists - the first list contains input tensors and the second list contains output tensors.
    """
    inputs_list = []
    labels_list = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, input_dim).cuda()
        labels = torch.randn(batch_size, output_dim).cuda()
        inputs_list.append(inputs)
        labels_list.append(labels)
    return inputs_list, labels_list


# Add current directory to sys.path on the workers.
add_sys_path = remote(
    "monarch.parallel.pipelining.runtime.add_sys_path_impl", propagate="inspect"
)

# Run optimizer.zero() as UDF on workers.
optimizer_zero_grad_udf = remote(
    "monarch.parallel.pipelining.runtime.optimizer_zero_grad", propagate="inspect"
)

# Run optimizer.step() on workers.
optimizer_step_udf = remote(
    "monarch.parallel.pipelining.runtime.optimizer_step", propagate="inspect"
)

# Build a module chunk on worker with UDF.
build_module_chunk_udf = remote(
    "monarch.parallel.pipelining.runtime.build_module_chunk",
    propagate=lambda module_class, *args, **kwargs: OpaqueRef(None),
)

# Add optimizer on worker for module chunk with UDF.
build_optimizer_chunk_udf = remote(
    "monarch.parallel.pipelining.runtime.build_optimizer_chunk",
    propagate=lambda model_chunk, lr: OpaqueRef(None),
)

# Build an OpaqueRef list to store the loss from micro batches.
build_loss_list_udf = remote(
    "monarch.parallel.pipelining.runtime.build_loss_list",
    propagate=lambda: OpaqueRef(None),
)

# Build an OpaqueRef object for the loss layer.
build_pp_loss_layer_udf = remote(
    "monarch.parallel.pipelining.runtime.build_pp_loss_layer",
    propagate=lambda: OpaqueRef(None),
)

log = remote("monarch.worker.worker.log", propagate="inspect")

set_worker_logging_level = remote(
    "monarch.worker.worker.set_worker_logging_level", propagate="inspect"
)

set_worker_random_seed = remote(
    "monarch.worker.worker.set_random_seed_impl", propagate="inspect"
)


def maybe_profile(should_profile: bool):
    if should_profile:
        return profile(
            activities=[
                # pyre-ignore[16]: Module `profiler` has no attribute
                torch.profiler.ProfilerActivity.CPU,
                # pyre-ignore[16]: Module `profiler` has no attribute
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready="./pp_traces/",
            schedule=Schedule(wait=1, warmup=1, active=2, repeat=1),
            record_shapes=True,
        )
    else:
        return nullcontext()


def generate_model_and_optimizer(
    mode, mesh, pp_meshes, input_dim, output_dim, hidden_dim, lr
):
    with mesh.activate():
        torch.set_default_device("cuda")
        if mode == "udf":
            model_chunks = []
            optimizers = []
            # buffers_list = []
            for i, pp_mesh in enumerate(pp_meshes):
                with pp_mesh.activate():
                    model_chunk_ref = build_module_chunk_udf(
                        "pp_example.Net",
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=hidden_dim,
                    )
                    optimizer_chunk_ref = build_optimizer_chunk_udf(model_chunk_ref, lr)
                    model_chunks.append(model_chunk_ref)
                    optimizers.append(optimizer_chunk_ref)

                    if i == len(pp_meshes) - 1:
                        loss_layer_ref = build_pp_loss_layer_udf()
                        loss_list_ref = build_loss_list_udf()
            model = model_chunks
            return model, optimizers, loss_layer_ref, loss_list_ref
        elif mode == "monarch":
            model_chunks = torch.nn.ModuleList()
            optimizers = []
            for _, pp_mesh in enumerate(pp_meshes):
                with pp_mesh.activate():
                    model_chunk = Net(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=hidden_dim,
                    )
                    model_chunk.train()
                    model_chunks.append(model_chunk)
            model = model_chunks
            for module in model:
                optimizer = optim.SGD(module.parameters(), lr=lr)
                optimizers.append(optimizer)
            loss_layer = nn.MSELoss()
            return model, optimizers, loss_layer, []
        elif mode == "opm":
            model_chunks = []
            optimizers = []
            # buffers_list = []
            for i, pp_mesh in enumerate(pp_meshes):
                with pp_mesh.activate():
                    model_chunk_opeque_module = OpaqueModule(
                        "pp_example.Net",
                        input_dim=input_dim,
                        output_dim=output_dim,
                        hidden_dim=hidden_dim,
                    )
                    optimizer_chunk_ref = build_optimizer_chunk_udf(
                        model_chunk_opeque_module._object, lr
                    )
                    model_chunks.append(model_chunk_opeque_module)
                    optimizers.append(optimizer_chunk_ref)

                    if i == len(pp_meshes) - 1:
                        loss_layer_ref = build_pp_loss_layer_udf()
                        loss_list_ref = build_loss_list_udf()
            model = model_chunks
            return model, optimizers, loss_layer_ref, loss_list_ref

        else:
            raise ValueError(f"Unknown mode: {mode}")


def create_reference_model(
    pp_meshes,
    input_dim,
    output_dim,
    hidden_dim,
):
    with pp_meshes[0].activate():
        ref_model_chunks = torch.nn.ModuleList()
        for _ in pp_meshes:
            model_chunk = Net(
                input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
            )
            model_chunk.train()
            ref_model_chunks.append(model_chunk)
        ref_model = ref_model_chunks
        ref_model.to("cuda")
        return ref_model


def train(world_size, pp_dim, batch_size, num_steps):
    parser = argparse.ArgumentParser(description="PP argparser")
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling of the model."
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Use SimulatorController."
    )
    # Define the allowed modes
    allowed_modes = ["monarch", "udf", "opm"]

    # Add the --mode argument
    parser.add_argument(
        "--mode",
        type=str,
        choices=allowed_modes,
        default="monarch",
        help="Mode of operation. Choices are: monarch, UDF, opm (default: monarch ).",
    )

    args = parser.parse_args()

    if args.simulate:
        simulator = Simulator(hosts=1, gpus=world_size, trace_mode="stream_only")
        orig_mesh = simulator.mesh
    else:
        simulator = None
        orig_mesh = local_mesh(hosts=1, gpus=world_size)

    input_dim = 2
    output_dim = 2
    hidden_dim = 2
    lr = 0.01
    mesh = orig_mesh.flatten("gpu").split(
        gpu=(
            "dp",
            "pp",
        ),
        pp=pp_dim,
    )
    torch.set_default_device("cuda")

    pp_meshes = [mesh(pp=i) for i in range(pp_dim)]
    with mesh.activate():
        set_worker_logging_level(logging.DEBUG)
        new_directory = os.path.dirname(os.path.abspath(__file__))
        add_sys_path(new_directory)
        # model init
        model, optimizers, loss_layer_ref, loss_list_ref = generate_model_and_optimizer(
            mode=args.mode,
            mesh=mesh,
            pp_meshes=pp_meshes,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lr=lr,
        )

        model_loss_list = []
        ref_model_loss_list = []
        # all workers should initialize model with the same values
        set_worker_random_seed(12345, 0)
        with pp_meshes[0].activate():
            x_list, y_list = make_input_output(
                batch_size, input_dim, output_dim, num_batches=num_steps
            )
        # Create a ref model to compare the results
        ref_model = create_reference_model(
            pp_meshes=pp_meshes,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )

        pp_stages = [[] for _ in range(world_size)]
        for idx in range(world_size):
            pp_stages[idx].append(model[idx])

        pp_class = PipelineParallelism(
            meshes=pp_meshes,
            stages=pp_stages,
            schedule="dora-dfs",
            batch_size=batch_size,
            loss_fn=loss_layer_ref,
            p2p_stream=Stream("self.p2p_stream"),
            compute_stream=get_active_stream(),
            loss_list=loss_list_ref,
        )
        pp_class.copy_params_to_new_model(ref_model)

        with pp_meshes[0].activate():
            ref_optimizer = optim.SGD(ref_model.parameters(), lr=lr)

        with maybe_profile(args.profile) as prof:
            for i in range(num_steps):
                x = x_list[i]
                y = y_list[i]

                for j, optimizer in enumerate(optimizers):
                    with pp_meshes[j].activate():
                        if args.mode in ["udf", "opm"]:
                            optimizer_zero_grad_udf(optimizer)
                        else:
                            optimizer.zero_grad()
                remote_loss = pp_class.run(x, y)

                for j, optimizer in enumerate(optimizers):
                    with pp_meshes[j].activate():
                        if args.mode in ["udf", "opm"]:
                            optimizer_step_udf(optimizer)
                        else:
                            optimizer.step()

                if args.profile:
                    prof.step()

                loss = fetch_shard(remote_loss).result()
                model_loss_list.append(loss)

        # Train the ref_model
        ref_model.to("cuda")
        criterion = nn.MSELoss()
        for i in range(num_steps):
            with pp_meshes[0].activate():
                x = x_list[i]
                y = y_list[i]
                output = x
                for _, model_chunk in enumerate(ref_model):
                    output = model_chunk(output)
                loss = criterion(output, y)
                ref_optimizer.zero_grad()
                loss.backward()
                ref_optimizer.step()
                loss = fetch_shard(loss).result()
            ref_model_loss_list.append(loss)

    with no_mesh.activate():
        logger.info(f"finished: \n{model_loss_list=}\n{ref_model_loss_list=}")
        # Compare the loss from ref_model and model
        for loss, ref_loss in zip(model_loss_list, ref_model_loss_list):
            assert torch.allclose(
                loss, ref_loss, atol=1e-8, rtol=1e-6
            ), f"{loss} != {ref_loss}"

        # Comparing parameters between ref_model and model.
        for _, (ref_module, module) in enumerate(zip(ref_model, model)):
            ref_params_shape = {}
            for ref_name, ref_param in ref_module.named_parameters():
                ref_params_shape[ref_name] = ref_param.shape

            for ref_name, ref_param in ref_module.named_parameters():
                ref_param_shape = ref_params_shape[ref_name]
                if args.mode in ["udf"]:
                    param = get_parameter_udf(module, ref_name, ref_param_shape)
                elif args.mode == "opm":
                    param = get_parameter_udf(module._object, ref_name, ref_param_shape)
                else:
                    param = dict(module.named_parameters())[ref_name]
                param_local = fetch_shard(param).result()

                ref_param_local = fetch_shard(ref_param).result()
                assert torch.allclose(
                    ref_param_local, param_local, atol=1e-6, rtol=1e-6
                ), f"param_name:{ref_name} {ref_param_local.detach().cpu().numpy()} != {param_local.detach().cpu().numpy()}"
    logging.info("PP example finished successfully")
    orig_mesh.exit()
    orig_mesh.deactivate()


def main():
    initialize_logging()
    torch.manual_seed(23456)
    return train(world_size=4, pp_dim=4, batch_size=4, num_steps=5)


if __name__ == "__main__":
    # run the training loop
    ret = main()
    sys.exit(ret)
