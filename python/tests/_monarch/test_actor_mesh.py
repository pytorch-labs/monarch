# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import pickle
from typing import Any, Iterable, List

import monarch
import pytest

from monarch._rust_bindings.monarch_hyperactor.actor import (
    PanicFlag,
    PythonMessage,
    PythonMessageKind,
)
from monarch._rust_bindings.monarch_hyperactor.actor_mesh import (
    PythonActorMesh,
    PythonActorMeshRef,
)

from monarch._rust_bindings.monarch_hyperactor.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    AllocConstraints,
    AllocSpec,
)

from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox, PortReceiver
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh
from monarch._rust_bindings.monarch_hyperactor.selection import Selection
from monarch._rust_bindings.monarch_hyperactor.shape import Shape


async def allocate() -> ProcMesh:
    spec = AllocSpec(AllocConstraints(), replicas=3, hosts=8, gpus=8)
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate(spec)
    proc_mesh = await ProcMesh.allocate_nonblocking(alloc)
    return proc_mesh


class MyActor:
    async def handle(
        self,
        mailbox: Mailbox,
        rank: int,
        shape: Shape,
        message: PythonMessage,
        panic_flag: PanicFlag,
        local_state: Iterable[Any] | None = None,
    ) -> None:
        assert rank is not None

        # Extract response_port from the message kind
        call_method = message.kind
        assert isinstance(call_method, PythonMessageKind.CallMethod)
        assert call_method.response_port is not None

        reply_port = call_method.response_port
        reply_port.send(
            mailbox,
            PythonMessage(
                PythonMessageKind.Result(rank), pickle.dumps(f"rank: {rank}")
            ),
        )


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
async def test_bind_and_pickling() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    with pytest.raises(NotImplementedError, match="use bind()"):
        pickle.dumps(actor_mesh)

    actor_mesh_ref = actor_mesh.bind()
    assert actor_mesh_ref.shape == actor_mesh.shape
    obj = pickle.dumps(actor_mesh_ref)
    unpickled = pickle.loads(obj)
    assert repr(actor_mesh_ref) == repr(unpickled)
    assert actor_mesh_ref.shape == unpickled.shape


async def verify_cast(
    actor_mesh: PythonActorMesh | PythonActorMeshRef,
    mailbox: Mailbox,
    cast_ranks: List[int],
) -> None:
    receiver: PortReceiver
    handle, receiver = mailbox.open_port()
    port_ref = handle.bind()

    message = PythonMessage(
        PythonMessageKind.CallMethod("echo", port_ref), pickle.dumps("ping")
    )
    sel = Selection.from_string("*")
    if isinstance(actor_mesh, PythonActorMesh):
        actor_mesh.cast(sel, message)
    elif isinstance(actor_mesh, PythonActorMeshRef):
        actor_mesh.cast(mailbox, sel, message)

    rcv_ranks = []
    for _ in range(len(cast_ranks)):
        message = await receiver.recv_task().into_future()
        result_kind = message.kind
        assert isinstance(result_kind, PythonMessageKind.Result)
        rank = result_kind.rank
        assert rank is not None
        rcv_ranks.append(rank)
    rcv_ranks.sort()
    assert rcv_ranks == cast_ranks
    # verify no more messages are received
    with pytest.raises(asyncio.exceptions.TimeoutError):
        await asyncio.wait_for(receiver.recv_task().into_future(), timeout=1)


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
@pytest.mark.timeout(30)
async def test_cast_handle() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    await verify_cast(actor_mesh, proc_mesh.client, list(range(3 * 8 * 8)))


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
@pytest.mark.timeout(30)
async def test_cast_ref() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    actor_mesh_ref = actor_mesh.bind()
    await verify_cast(actor_mesh_ref, proc_mesh.client, list(range(3 * 8 * 8)))


async def verify_slice(
    actor_mesh: PythonActorMesh | PythonActorMeshRef,
    mailbox: Mailbox,
) -> None:
    sliced_mesh = actor_mesh.slice(
        gpus=slice(2, 8, 2),
        replicas=slice(None, 2),
        hosts=slice(3, 7),
    )
    sliced_shape = sliced_mesh.shape
    # fmt: off
    # turn off formatting to make the following list more readable
    replica_0_ranks = [
        #  gpus=2,4,6
        24 + 2, 24 + 4, 24 + 6,  # hosts=3
        32 + 2, 32 + 4, 32 + 6,  # hosts=4
        40 + 2, 40 + 4, 40 + 6,  # hosts=5
        48 + 2, 48 + 4, 48 + 6,  # hosts=6
    ]
    # fmt: on
    replica_1_ranks = [rank + 64 for rank in replica_0_ranks]
    assert (
        sliced_shape.ranks() == replica_0_ranks + replica_1_ranks
    ), f"left is {sliced_shape.ranks()}"
    await verify_cast(sliced_mesh, mailbox, sliced_shape.ranks())

    assert sliced_shape.labels == ["replicas", "hosts", "gpus"]
    assert sliced_shape.ndslice.sizes == [2, 4, 3]
    # When slicing a sliced mesh, the user treats this sliced mesh as a
    # continuous mesh, and calculates the dimensions based on that assumption,
    # without considering the original mesh.
    #
    # e.g, the following slicing operation selects index 0 and 2 of the hosts
    # dimension on the sliced mesh. But corresponding index on the original
    #  mesh is 3 and 5.
    sliced_again = sliced_mesh.slice(
        replicas=1,
        hosts=slice(None, None, 2),
        gpus=slice(1, 3),
    )
    again_shape = sliced_again.shape
    assert again_shape.labels == ["replicas", "hosts", "gpus"]
    assert again_shape.ndslice.sizes == [1, 2, 2]
    # fmt: off
    # turn off formatting to make the following list more readable
    selected_ranks = [
        rank + 64 for rank in
        [
            #  gpus=4,6
            24 + 4, 24 + 6,  # hosts=3
            40 + 4, 40 + 6,  # hosts=5
        ]
    ]
    # fmt: on
    assert again_shape.ranks() == selected_ranks, f"left is {sliced_shape.ranks()}"


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
@pytest.mark.timeout(30)
async def test_slice_actor_mesh_handle() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    await verify_slice(actor_mesh, proc_mesh.client)


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
@pytest.mark.timeout(30)
async def test_slice_actor_mesh_ref() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    actor_mesh_ref = actor_mesh.bind()
    await verify_slice(actor_mesh_ref, proc_mesh.client)
