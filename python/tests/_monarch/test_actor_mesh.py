# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import pickle
from typing import Any, Callable, cast, Coroutine, Iterable, List, TYPE_CHECKING

import monarch
import pytest

from monarch._rust_bindings.monarch_hyperactor.actor import (
    MethodSpecifier,
    PanicFlag,
    PythonMessage,
    PythonMessageKind,
)
from monarch._rust_bindings.monarch_hyperactor.actor_mesh import PythonActorMesh

from monarch._rust_bindings.monarch_hyperactor.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    AllocConstraints,
    AllocSpec,
)

if TYPE_CHECKING:
    from monarch._rust_bindings.monarch_hyperactor.actor import PortProtocol

from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox, PortReceiver
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask
from monarch._rust_bindings.monarch_hyperactor.shape import Shape


def run_on_tokio(
    fn: Callable[[], Coroutine[Any, Any, None]],
) -> Callable[[], None]:
    """
    Wrapper for function that use the internal tokio event loop
    APIs and need to run on that event loop.
    """
    return lambda: PythonTask.from_coroutine(fn()).block_on()


async def allocate() -> ProcMesh:
    spec = AllocSpec(AllocConstraints(), replicas=3, hosts=8, gpus=8)
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate_nonblocking(spec)
    proc_mesh = await ProcMesh.allocate_nonblocking(alloc)
    return proc_mesh


class MyActor:
    def __init__(self) -> None:
        #  Note: for the same actor, its rank on the root mesh could be different
        # from its rank on the mesh it is cast to. This is because the cast
        # mesh could be a sliced mesh.
        self._rank_on_root_mesh: int = -1

    async def handle(
        self,
        mailbox: Mailbox,
        rank: int,
        shape: Shape,
        method: MethodSpecifier,
        message: bytes,
        panic_flag: PanicFlag,
        local_state: Iterable[Any],
        response_port: "PortProtocol[Any]",
    ) -> None:
        match method:
            case MethodSpecifier.Init():
                # Since this actor is spawn from the root proc mesh, the rank
                # passed from init should be the rank on the root mesh.
                self._rank_on_root_mesh = rank
                response_port.send(None)
                return None
            case MethodSpecifier.ReturnsResponse(name=_):
                response_port.send(self._rank_on_root_mesh)
                return None
            case MethodSpecifier.ExplicitPort(name=_):
                response_port.exception(
                    NotImplementedError("ExplicitPort is not supported yet")
                )
                return None


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
@pytest.mark.timeout(30)
async def test_bind_and_pickling() -> None:
    @run_on_tokio
    async def run() -> None:
        proc_mesh = await allocate()
        actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
        pickle.dumps(actor_mesh)

        actor_mesh_ref = actor_mesh.new_with_shape(proc_mesh.shape)
        obj = pickle.dumps(actor_mesh_ref)
        unpickled = pickle.loads(obj)
        await proc_mesh.stop_nonblocking()

    run()


async def spawn_actor_mesh(proc_mesh: ProcMesh) -> PythonActorMesh:
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    # init actors to record their root ranks
    receiver: PortReceiver
    handle, receiver = proc_mesh.client.open_port()
    port_ref = handle.bind()

    message = PythonMessage(
        PythonMessageKind.CallMethod(MethodSpecifier.Init(), port_ref),
        pickle.dumps(None),
    )
    actor_mesh.cast(message, "all", proc_mesh.client)
    # wait for init to complete
    for _ in range(len(proc_mesh.shape.ndslice)):
        await receiver.recv_task()

    return actor_mesh


async def cast_to_call(
    actor_mesh: PythonActorMesh,
    mailbox: Mailbox,
    message: PythonMessage,
) -> None:
    actor_mesh.cast(message, "all", mailbox)


async def verify_cast_to_call(
    actor_mesh: PythonActorMesh,
    mailbox: Mailbox,
    root_ranks: List[int],
) -> None:
    receiver: PortReceiver
    handle, receiver = mailbox.open_port()
    port_ref = handle.bind()

    # Now send the real message
    message = PythonMessage(
        PythonMessageKind.CallMethod(MethodSpecifier.ReturnsResponse("echo"), port_ref),
        pickle.dumps("ping"),
    )
    await cast_to_call(actor_mesh, mailbox, message)

    rcv_ranks = []
    for _ in range(len(root_ranks)):
        message = await receiver.recv_task()
        result_kind = message.kind
        assert isinstance(result_kind, PythonMessageKind.Result)
        cast_rank = result_kind.rank
        assert cast_rank is not None
        root_rank = cast(int, pickle.loads(message.message))
        rcv_ranks.append((cast_rank, root_rank))
    rcv_ranks.sort(key=lambda pair: pair[0])
    recv_cast_ranks, recv_root_ranks = zip(*rcv_ranks)
    assert recv_root_ranks == tuple(
        root_ranks
    ), f"recv_root_ranks={recv_root_ranks}, root_ranks={tuple(root_ranks)}"
    assert recv_cast_ranks == tuple(
        range(len(root_ranks))
    ), f"recv_cast_ranks={recv_cast_ranks}, root_ranks={tuple(root_ranks)}"
    # verify no more messages are received
    with pytest.raises(TimeoutError):
        await receiver.recv_task().with_timeout(1)


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
@pytest.mark.timeout(30)
async def test_cast_handle() -> None:
    @run_on_tokio
    async def run() -> None:
        proc_mesh = await allocate()
        actor_mesh = await spawn_actor_mesh(proc_mesh)
        await verify_cast_to_call(actor_mesh, proc_mesh.client, list(range(3 * 8 * 8)))

        await proc_mesh.stop_nonblocking()

    run()


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
@pytest.mark.timeout(30)
async def test_cast_ref() -> None:
    @run_on_tokio
    async def run() -> None:
        proc_mesh = await allocate()
        actor_mesh = await spawn_actor_mesh(proc_mesh)
        actor_mesh_ref = actor_mesh.new_with_shape(proc_mesh.shape)
        await verify_cast_to_call(
            actor_mesh_ref, proc_mesh.client, list(range(3 * 8 * 8))
        )

        await proc_mesh.stop_nonblocking()

    run()
