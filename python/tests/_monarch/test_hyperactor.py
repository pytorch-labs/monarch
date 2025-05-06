# pyre-strict

import asyncio
import importlib
import multiprocessing
import os
import pickle
import signal
import time
from asyncio import AbstractEventLoop
from typing import Any

from monarch._monarch import hyperactor


class MyActor(hyperactor.Actor):
    async def handle(
        self, mailbox: hyperactor.Mailbox, message: hyperactor.PythonMessage
    ) -> None:
        return None

    async def handle_cast(
        self,
        mailbox: hyperactor.Mailbox,
        rank: int,
        coordinates: list[tuple[str, int]],
        message: hyperactor.PythonMessage,
    ) -> None:
        reply_port = pickle.loads(message.message)
        mailbox.post(
            reply_port, hyperactor.PythonMessage("echo", pickle.dumps(coordinates))
        )


# have to use a single loop for all tests, otherwise there are
# loop closed errors.

loop: AbstractEventLoop = asyncio.get_event_loop()


# pyre-ignore[2,3]
def run_async(x: Any) -> Any:
    return lambda: loop.run_until_complete(x())


def test_import() -> None:
    try:
        import monarch._monarch.hyperactor  # noqa
    except ImportError as e:
        raise ImportError(f"hyperactor failed to import: {e}")


def test_actor_id() -> None:
    actor_id = hyperactor.ActorId(world_name="test", rank=0, actor_name="actor")
    assert actor_id.pid == 0
    assert str(actor_id) == "test[0].actor[0]"


def test_no_hang_on_shutdown() -> None:
    def test_fn() -> None:
        import monarch._monarch.hyperactor  # noqa

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


@run_async
async def test_allocator() -> None:
    hyperactor.init_asyncio_loop()
    spec = hyperactor.AllocSpec(replica=2)
    _ = await hyperactor.LocalAllocator.allocate(spec)


@run_async
async def test_proc_mesh() -> None:
    hyperactor.init_asyncio_loop()
    spec = hyperactor.AllocSpec(replica=2)
    alloc = await hyperactor.LocalAllocator.allocate(spec)
    _ = await hyperactor.ProcMesh.allocate(alloc)


@run_async
async def test_actor_mesh() -> None:
    hyperactor.init_asyncio_loop()
    spec = hyperactor.AllocSpec(replica=2)
    alloc = await hyperactor.LocalAllocator.allocate(spec)
    proc_mesh = await hyperactor.ProcMesh.allocate(alloc)
    actor_mesh = await proc_mesh.spawn("test", MyActor)
    actor_mesh.cast(hyperactor.PythonMessage("hello", b"world"))

    assert actor_mesh.get(0) is not None
    assert actor_mesh.get(1) is not None
    assert actor_mesh.get(2) is None

    assert isinstance(actor_mesh.client, hyperactor.Mailbox)


@run_async
async def test_proc_mesh_process_allocator() -> None:
    hyperactor.init_asyncio_loop()
    spec = hyperactor.AllocSpec(replica=2)
    cmd = importlib.resources.files("monarch.python.tests._monarch").joinpath(
        "bootstrap"
    )
    allocator = hyperactor.ProcessAllocator(str(cmd))
    alloc = await allocator.allocate(spec)
    proc_mesh = await hyperactor.ProcMesh.allocate(alloc)
    actor_mesh = await proc_mesh.spawn("test", MyActor)

    handle, receiver = actor_mesh.client.open_port()
    actor_mesh.cast(hyperactor.PythonMessage("hello", pickle.dumps(handle.bind())))
    coords = {await receiver.recv(), await receiver.recv()}
    assert coords == {[("replica", 0)], [("replica", 1)]}
