# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes

import pytest
from monarch.actor_mesh import Actor, ActorMeshRefCallFailedException, endpoint

from monarch.proc_mesh import proc_mesh


class ExceptionActor(Actor):
    """An actor that has endpoints which raise exceptions or cause segfaults."""

    @endpoint
    async def raise_exception(self) -> None:
        """Endpoint that raises an exception."""
        raise Exception("This is a test exception")

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


class ExceptionActorSync(Actor):
    """An actor that has endpoints which raise exceptions or cause segfaults."""

    @endpoint  # pyre-ignore
    def raise_exception(self) -> None:
        """Endpoint that raises an exception."""
        raise Exception("This is a test exception")

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


@pytest.mark.parametrize(
    "actor_class,actor_name",
    [
        (ExceptionActor, "exception_actor_async_call"),
        (ExceptionActorSync, "exception_actor_sync_call"),
    ],
)
@pytest.mark.parametrize("num_procs", [1, 2])
async def test_actor_exception(actor_class, actor_name, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = await proc_mesh(gpus=num_procs)
    exception_actor = await proc.spawn(actor_name, actor_class)

    with pytest.raises(
        ActorMeshRefCallFailedException, match="This is a test exception"
    ):
        if num_procs == 1:
            await exception_actor.raise_exception.call_one()
        else:
            await exception_actor.raise_exception.call()


@pytest.mark.parametrize(
    "actor_class,actor_name",
    [
        (ExceptionActor, "exception_actor_async_call"),
        (ExceptionActorSync, "exception_actor_sync_call"),
    ],
)
@pytest.mark.parametrize("num_procs", [1, 2])
def test_actor_exception_sync(actor_class, actor_name, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = proc_mesh(gpus=num_procs).get()
    exception_actor = proc.spawn(actor_name, actor_class).get()

    with pytest.raises(
        ActorMeshRefCallFailedException, match="This is a test exception"
    ):
        if num_procs == 1:
            exception_actor.raise_exception.call_one().get()
        else:
            exception_actor.raise_exception.call().get()


@pytest.mark.parametrize(
    "actor_class,actor_name",
    [
        (ExceptionActor, "segfault_actor_async_call"),
        (ExceptionActorSync, "segfault_actor_sync_call"),
    ],
)
@pytest.mark.parametrize("num_procs", [1, 2])
async def test_actor_segfault(actor_class, actor_name, num_procs):
    """
    Test that segfaults in actor endpoints are propagated to the client as exceptions.
    """
    proc = await proc_mesh(gpus=num_procs)
    segfault_actor = await proc.spawn(actor_name, actor_class)

    with pytest.raises(ActorMeshRefCallFailedException):
        if num_procs == 1:
            await segfault_actor.cause_segfault.call_one()
        else:
            await segfault_actor.cause_segfault.call()


@pytest.mark.parametrize(
    "actor_class,actor_name",
    [
        (ExceptionActor, "segfault_actor_async_sync"),
        (ExceptionActorSync, "segfault_actor_sync_sync"),
    ],
)
@pytest.mark.parametrize("num_procs", [1, 2])
def test_actor_segfault_sync(actor_class, actor_name, num_procs):
    """
    Test that segfaults in actor endpoints are propagated to the client as exceptions.
    """
    proc = proc_mesh(gpus=num_procs).get()
    segfault_actor = proc.spawn(actor_name, actor_class).get()

    with pytest.raises(ActorMeshRefCallFailedException):
        if num_procs == 1:
            segfault_actor.cause_segfault.call_one().get()
        else:
            segfault_actor.cause_segfault.call().get()
