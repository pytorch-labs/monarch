# pyre-unsafe
import os
from contextlib import nullcontext
from enum import Enum

import torch  # isort:skip

from monarch import (
    fetch_shard,
    Future,
    get_active_stream,
    no_mesh,
    notebook as nb,
    Pipe,
    python_local_mesh,
    remote,
    remote_generator,
    rust_backend_mesh,
    Stream,
)

# this function helps get a local device mesh for testing
from monarch._testing import mock_mesh
from monarch_supervisor.logging import initialize_logging


class MeshType(Enum):
    MOCK = 1
    RUST = 2
    NOTEBOOK = 3
    RUST_LOCAL = 4
    PYTHON_LOCAL = 5
    RUST_TEST = 6
    RUST_MAST = 7


def _parse_mesh(env_var_name: str) -> MeshType:
    """Parse backend from environment variable"""
    mesh_str = os.environ.get(env_var_name)
    if mesh_str is None:
        return MeshType.PYTHON_LOCAL  # default value
    try:
        return MeshType[mesh_str.upper()]
    except KeyError:
        raise ValueError(f"Invalid mesh type: {mesh_str}")


mesh = _parse_mesh("MESH_TYPE")

# The notebook module initializes logging on its own.
if mesh != MeshType.NOTEBOOK:
    initialize_logging("example")

match mesh:
    case MeshType.MOCK:
        device_mesh = nullcontext(mock_mesh(hosts=2, gpus=2))
    case MeshType.RUST:
        device_mesh = nullcontext(
            rust_backend_mesh(system_addr=os.environ["SYSTEM_ADDR"], hosts=2, gpus=2)
        )
    case MeshType.NOTEBOOK:
        device_mesh = nullcontext(
            nb.mast_mesh(mast_job_name="your mast job", hosts=2, n_gpus_per_host=2)
        )
    case MeshType.RUST_LOCAL:
        from monarch.rust_local_mesh import local_mesh

        device_mesh = local_mesh(hosts=2, gpus_per_host=2)
    case MeshType.RUST_TEST:
        from monarch.rust_local_mesh import local_mesh, LoggingLocation, SocketType

        device_mesh = local_mesh(
            hosts=2,
            gpus_per_host=2,
            socket_type=SocketType.UNIX,
            logging_location=LoggingLocation.DEFAULT,
        )
    case MeshType.RUST_MAST:
        try:
            from monarch import rust_mast_mesh
        except ImportError:
            from unittest.mock import Mock

            rust_mast_mesh = Mock()
        device_mesh = nullcontext(
            rust_mast_mesh(job_name=os.environ["JOB_NAME"], hosts=2, gpus=2)
        )
    case _:
        device_mesh = nullcontext(python_local_mesh(hosts=2, gpus=2))


# device meshes initially describe the hardware that
# the job will use

# Basics
# ------

with device_mesh as device_mesh:
    # This is normal torch, we create a local cpu tensor:
    local = torch.rand(3, 4)

    # however, we activate the device_mesh and tensors
    # will be created across the mesh

    with device_mesh.activate():
        t = torch.rand(3, 4)

    # for interactive use lets keep this device_mesh active
    device_mesh.activate()

    # user-defined remote functions
    log = remote("monarch.worker._testing_function.log", propagate="inspect")

    def has_nan(t):
        return torch.isnan(t).any().item()

    # run on workers
    log("my tensor: %s", t)

    # you can do mutable stuff
    t.add_(1)

    # devices still work as normal
    t = t.cuda()

    # Communication Operators
    # -----------------------

    # most comms turn into a 'reduce' with
    # different settings, but we will have
    # syntax sugar for common things like 'all_gather'
    # and 'all_to_all'

    x = (
        (device_mesh.rank("host") + 1) * 1000000
        + (device_mesh.rank("gpu") + 1) * 1000
        + torch.arange(6).reshape(2, 3)
        + 1
    )
    log("orig tensor:\n%s", x)

    x = x.cuda()

    t = x.reduce("gpu")

    log("reduced tensor:\n%s", t)

    # inplace

    t.reduce_("host")

    log("reduced tensor:\n%s", t)

    # 'gather'
    gathered = t.reduce("gpu", reduction="stack")
    log("gathered tensor:\n%s", gathered)

    # reduce-scatter
    reduce_scatter = x.reduce("gpu", scatter=True)
    log("before\n%s\nscattered:\n%s\n", x, reduce_scatter)

    # Observing results on controller
    # -------------------------------

    # to get a value locally you can fetch the local
    # value on a particular shard, which returns a future
    local: Future = fetch_shard(reduce_scatter, {"host": 1, "gpu": 0})

    with no_mesh.activate():
        print(local.result())

    # you don't always want tensor though, so you can pre
    # process the value before sending it:
    local = remote(has_nan).call_on_shard_and_fetch(
        reduce_scatter, shard={"host": 1, "gpu": 0}
    )
    print(local.result())

    # Moving Tensors
    # -------

    # you can select a subset of the devices with
    # name-based indexing
    host0 = device_mesh(host=0)
    host1 = device_mesh(host=1)

    with host0.activate():
        a = torch.rand(2, 3, device="cuda")
        # send data from one mesh to another

        # there is not possibility of mismatched send/recv
        # because both are issued at the same time to
        # the correct workers. So no possibility of deadlock!

        # pyre-ignore[16]
        b = a.to_mesh(host1)

        # or slice a bigger mesh and send it to another
        c = t.slice_mesh(host=0).to_mesh(host1)

    # the receiving host can then compute with those
    # sent tensors
    with host1.activate():
        d = b + c
        ld = fetch_shard(d, {"gpu": 1})

    with no_mesh.activate():
        print(ld.result())

    # Errors
    # -------
    # what happens if something goes wrong on a worker
    # that we cannot statically know from the controller?

    do_bogus_tensor_work = remote(
        "monarch.worker._testing_function.do_bogus_tensor_work",
        propagate=lambda x, y, fail_rank=None: x + y,
    )

    t = torch.rand(3, 4, device="cuda")

    r = do_bogus_tensor_work(t, t)

    print(fetch_shard(r).exception())

    x = t + t
    log("x: %s", x)

    # only rank 1 will fail now
    r = do_bogus_tensor_work(t, t, fail_rank=1)

    reduced = r.reduce("gpu")

    # notice the error is still the
    # op that started the failure
    print(fetch_shard(reduced).exception())

    # but we can still compute with things not dependent on the error
    print(fetch_shard(t + t).exception())

    # notice that t + t happened _after_ a reduce operator
    # failed become one of the workers didn't have a valid tensor.
    # with standard distributed, this would have been a deadlock!

    # Streams
    # -------
    # express parallel. Each worker

    # we have been using a default stream
    default = get_active_stream()

    # but we can create others
    # on each worker, code on the same stream runs in order
    # but code on different streams runs in parallel
    comms = Stream("comms")  # argument is just a debug name

    # parallel compute
    t = torch.rand(3, 4, device="cuda")
    with comms.activate():
        t2 = torch.rand(3, 4, device="cuda")

    # you can't directly use tensors across streams
    # (when they are computed it is a race)
    try:
        t.add(t2)
    except Exception as e:
        print(e)

    # but you can borrow a tensor to another stream
    # this will insert the appropriate synchronization

    t2_on_default, borrow = default.borrow(t2)
    r = t.add(t2_on_default)
    print(fetch_shard(r).result())

    # while t2 has a borrow, you can't mutate it
    # because that would race with the borrow.
    # you can also borrow with mutate=True,
    # which allows the borrowing stream to mutate
    # the tensor, but disables _reads_ of the tensor
    # from other streams.
    try:
        with comms.activate():
            t2.add_(1)
    except Exception as e:
        print(e)

    # when you are done, you have to explicitly
    # tell the borrow it is done, because this will
    # synchrionize t2's _memory_ back to the comms stream
    borrow.drop()

    # drop() can actually be used on any tensor
    # it says 'get rid of the memory for this tensor now'
    # if it or any view of it is used later, that will
    # cause an error. This is a good way to make assertions
    # about the lifetime of tensors.

    # Pipes
    # -----
    # We use pipes to communicate tensor data between other actors (checkpointing,
    # dataloading, etc.) and the worker.

    @remote_generator("monarch.worker._testing_function.example_data_loader")
    def example_data_loader(p: Pipe, x, y):
        for _i in range(x, y):
            yield torch.zeros(())

    # Creates a pipe for each device in our mesh
    dl = example_data_loader(3, 7)
    foo = dl.recv()

    # 'foo' is a dtensor
    print(foo)
    print(fetch_shard(foo).result())

    # just for local testing to shut down all the workers
    device_mesh.exit()

    # Necessary for clean exit when device_mesh.activate() is used outside
    # of a with block
    device_mesh.deactivate()
