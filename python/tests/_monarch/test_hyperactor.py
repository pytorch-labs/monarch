# pyre-strict

import multiprocessing
import os
import pickle
import signal
import sys
import time

import monarch
import pytest

from monarch._rust_bindings.hyperactor_extension.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    AllocConstraints,
    AllocSpec,
)

from monarch._rust_bindings.monarch_hyperactor.actor import PythonMessage

from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh


class MyActor:
    async def handle(self, mailbox: Mailbox, message: PythonMessage) -> None:
        return None

    async def handle_cast(
        self,
        mailbox: Mailbox,
        rank: int,
        coordinates: list[tuple[str, int]],
        message: PythonMessage,
    ) -> None:
        reply_port = pickle.loads(message.message)
        mailbox.post(reply_port, PythonMessage("echo", pickle.dumps(coordinates)))


def test_import() -> None:
    try:
        import monarch._rust_bindings  # noqa
    except ImportError as e:
        raise ImportError(f"monarch._rust_bindings failed to import: {e}")


def test_actor_id() -> None:
    actor_id = ActorId(world_name="test", rank=0, actor_name="actor")
    assert actor_id.pid == 0
    assert str(actor_id) == "test[0].actor[0]"


def test_no_hang_on_shutdown() -> None:
    def test_fn() -> None:
        import monarch._rust_bindings  # noqa
        import torch  # noqa

        time.sleep(100)

    proc = multiprocessing.Process(target=test_fn)
    proc.start()
    pid = proc.pid
    assert pid is not None

    os.kill(pid, signal.SIGTERM)
    time.sleep(2)
    pid, code = os.waitpid(pid, os.WNOHANG)
    assert pid > 0
    assert code == signal.SIGTERM, code


async def test_allocator() -> None:
    spec = AllocSpec(AllocConstraints(), replica=2)
    allocator = monarch.LocalAllocator()
    _ = await allocator.allocate(spec)


async def test_proc_mesh() -> None:
    spec = AllocSpec(AllocConstraints(), replica=2)
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate(spec)
    proc_mesh = await ProcMesh.allocate_nonblocking(alloc)
    assert str(proc_mesh) == "<ProcMesh { shape: {replica=2} }>"


async def test_actor_mesh() -> None:
    spec = AllocSpec(AllocConstraints(), replica=2)
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate(spec)
    proc_mesh = await ProcMesh.allocate_nonblocking(alloc)
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)

    assert actor_mesh.get(0) is not None
    assert actor_mesh.get(1) is not None
    assert actor_mesh.get(2) is None

    assert isinstance(actor_mesh.client, Mailbox)


# oss_skip: hangs when run through pytest but not when run through buck
# pyre-ignore[56]
@pytest.mark.oss_skip
async def test_proc_mesh_process_allocator() -> None:
    spec = AllocSpec(AllocConstraints(), replica=2)
    env = {}
    env["PAR_MAIN_OVERRIDE"] = "monarch.bootstrap_main"
    env["HYPERACTOR_MANAGED_SUBPROCESS"] = "1"
    allocator = monarch.ProcessAllocator(sys.argv[0], None, env)
    alloc = await allocator.allocate(spec)
    proc_mesh = await ProcMesh.allocate_nonblocking(alloc)
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    handle, receiver = actor_mesh.client.open_port()
    actor_mesh.cast(PythonMessage("hello", pickle.dumps(handle.bind())))
    coords = {await receiver.recv(), await receiver.recv()}
    # `coords` is a pair of messages. logically:
    # assert coords == {(("replica", 0)), (("replica", 1))}
