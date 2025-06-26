# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import ctypes
import sys
import time

import click
import monarch
import torch
from monarch._rust_bindings.hyperactor_extension.alloc import (
    AllocConstraints,
    AllocSpec,
)

from monarch._rust_bindings.monarch_extension.panic import panicking_function

from monarch.actor_mesh import Actor, endpoint, send
from monarch.common import messages
from monarch.common._device_utils import _local_device_count
from monarch.common.remote import remote
from monarch.mesh_controller import spawn_tensor_engine
from monarch.proc_mesh import proc_mesh, ProcMesh


class ErrorActor(Actor):
    """An actor that has endpoints cause segfaults."""

    @endpoint
    async def cause_segfault(self) -> None:
        """Endpoint that causes a segmentation fault."""
        # Create a C function pointer to an invalid memory address
        # This will reliably cause a segmentation fault when called
        function_type = ctypes.CFUNCTYPE(None)
        # Use a non-zero but invalid address to avoid ctypes null pointer checks
        invalid_address = 0xDEADBEEF
        invalid_function = function_type(invalid_address)
        # Calling this function will cause a segfault
        invalid_function()

    @endpoint
    async def cause_panic(self) -> None:
        """Endpoint that calls a Rust function that panics."""
        panicking_function()

    @endpoint
    async def await_then_error(self) -> None:
        await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)
        raise RuntimeError("oh noez")


class ErrorActorSync(Actor):
    """An actor that has endpoints cause segfaults."""

    @endpoint  # pyre-ignore
    def cause_segfault(self) -> None:
        """Endpoint that causes a segmentation fault."""
        # Create a C function pointer to an invalid memory address
        # This will reliably cause a segmentation fault when called
        function_type = ctypes.CFUNCTYPE(None)
        # Use a non-zero but invalid address to avoid ctypes null pointer checks
        invalid_address = 0xDEADBEEF
        invalid_function = function_type(invalid_address)
        # Calling this function will cause a segfault
        invalid_function()

    @endpoint  # pyre-ignore
    def cause_panic(self) -> None:
        """Endpoint that calls a Rust function that panics."""
        panicking_function()


def _run_error_test_sync(num_procs, sync_endpoint, endpoint_name):
    proc = proc_mesh(gpus=num_procs).get()
    if sync_endpoint:
        actor_class = ErrorActorSync
    else:
        actor_class = ErrorActor
    error_actor = proc.spawn("error_actor", actor_class).get()

    # This output is checked in the test to make sure that the process actually got here
    print("I actually ran")
    sys.stdout.flush()

    if endpoint_name == "cause_segfault":
        endpoint = error_actor.cause_segfault
    elif endpoint_name == "cause_panic":
        endpoint = error_actor.cause_panic
    else:
        raise ValueError(f"Unknown endpoint name: {endpoint_name}")

    # Exercise both call() and call_one() in our tests, to check that error
    # aggregation behavior is consistent.
    if num_procs == 1:
        endpoint.call_one().get()
    else:
        endpoint.call().get()


def _run_error_test(num_procs, sync_endpoint, endpoint_name):
    if sync_endpoint:
        actor_class = ErrorActorSync
    else:
        actor_class = ErrorActor

    async def run_test():
        proc = await proc_mesh(gpus=num_procs)
        error_actor = await proc.spawn("error_actor", actor_class)

        # This output is checked in the test to make sure that the process actually got here
        print("I actually ran")
        sys.stdout.flush()

        if endpoint_name == "cause_segfault":
            endpoint = error_actor.cause_segfault
        elif endpoint_name == "cause_panic":
            endpoint = error_actor.cause_panic
        else:
            raise ValueError(f"Unknown endpoint name: {endpoint_name}")

        # Exercise both call() and call_one() in our tests, to check that error
        # aggregation behavior is consistent.
        if num_procs == 1:
            await endpoint.call_one()
        else:
            await endpoint.call()

    asyncio.run(run_test())


def python_throws_error_on_tensor_worker(_mesh):
    throws = remote(
        "monarch.worker._testing_function.throw_python_exception",
        propagate=lambda: torch.tensor(1),
    )
    throws()
    time.sleep(1)
    torch.ones(())


def python_returns_unexpected_value_to_rust_on_tensor_worker(_mesh):
    actually_returns_a_string = remote(
        "monarch.worker._testing_function.returns_a_string",
        propagate=lambda: torch.tensor(1),
    )
    client_thinks_im_a_tensor = actually_returns_a_string()
    client_thinks_im_a_tensor += 1
    time.sleep(2)
    torch.ones(())


def rust_error_on_tensor_worker(mesh):
    class _Recording:
        def __init__(self):
            self.ref = 0

    # Trying to call a recording that doesn't exist will cause a rust error
    # on the worker.
    mesh._send(
        messages.CallRecording(ident=0, recording=_Recording(), results=[], actuals=[])
    )
    time.sleep(1)
    torch.rand(3, 4)


def rust_panic_on_tensor_worker(_mesh):
    # __test_panic is a special invocation for testing inside StreamActor
    # that forces a panic
    panic = remote("__test_panic", propagate=lambda: torch.ones(()))
    panic()
    time.sleep(1)
    torch.rand(3, 4)


def tensor_worker_remote_function_import_error(_mesh):
    import_error = remote(
        "monarch.worker._testing_throws_on_import._an_unusable_function",
        propagate=lambda: torch.ones(()),
    )
    import_error()
    time.sleep(1)
    torch.ones(())


@click.group()
def main():
    pass


@main.command("error-endpoint")
@click.option("--num-procs", type=int, required=True)
@click.option("--sync-test-impl", type=bool, required=True)
@click.option("--sync-endpoint", type=bool, required=True)
@click.option("--endpoint-name", type=str, required=True)
def error_endpoint(num_procs, sync_test_impl, sync_endpoint, endpoint_name):
    print(
        f"Running segfault test: {num_procs=} {sync_test_impl=} {sync_endpoint=}, {endpoint_name=}"
    )

    if sync_test_impl:
        _run_error_test_sync(num_procs, sync_endpoint, endpoint_name)
    else:
        _run_error_test(num_procs, sync_endpoint, endpoint_name)


@main.command("error-bootstrap")
def error_bootstrap():
    print("I actually ran")
    sys.stdout.flush()

    proc_mesh(gpus=4, env={"MONARCH_ERROR_DURING_BOOTSTRAP_FOR_TESTING": "1"}).get()


async def _error_unmonitored():
    print("I actually ran")
    sys.stdout.flush()

    proc = await proc_mesh(gpus=1)
    actor = await proc.spawn("error_actor", ErrorActor)

    # fire and forget
    send(actor.await_then_error, (), {}, None, "all")

    # Wait. Eventually a supervision event will get propagated and the process
    # will exit.
    #
    # If an event is not delivered, the test will time out before this sleep
    # finishes.
    await asyncio.sleep(300)


@main.command("error-unmonitored")
def error_unmonitored():
    asyncio.run(_error_unmonitored())


@main.command("error-client")
def error_client():
    gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=1)
    allocator = monarch.LocalAllocator()
    alloc = allocator.allocate(spec).get()
    ProcMesh.from_alloc(alloc).get()
    # Attempting to reuse alloc for a new proc mesh will cause a rust error.
    ProcMesh.from_alloc(alloc).get()


@main.command("error-tensor-engine")
@click.option("--num-procs", type=int, required=True)
@click.option("--test-name", type=str, required=True)
def error_tensor_engine(num_procs, test_name):
    print(f"Running tensor engine test: {num_procs=} {test_name=}")
    proc = proc_mesh(gpus=num_procs).get()
    mesh = spawn_tensor_engine(proc)
    with mesh.activate():
        test_func = globals().get(test_name)
        if not test_func:
            raise ValueError(f"Function {test_name} not found in the current module.")
        test_func(mesh)
    mesh.exit()


if __name__ == "__main__":
    main()
