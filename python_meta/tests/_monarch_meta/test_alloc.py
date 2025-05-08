# pyre-strict

import asyncio
import pickle
from asyncio import AbstractEventLoop
from typing import Any

from monarch._monarch import hyperactor
from monarch_meta._monarch_meta import hyperactor_meta


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


# dummy test to check if the bindings work.
@run_async
async def test_mast_allocator() -> None:
    allocator = hyperactor_meta.MastAllocator(
        hyperactor_meta.MastAllocatorConfig(job_name="test_job")
    )
    spec = hyperactor.AllocSpec(
        hyperactor.AllocConstraints(
            {hyperactor_meta.MastAllocator.ALLOC_LABEL_TASK_GROUP: "test_task_group"}
        ),
        host=2,
        gpu=2,
    )
    try:
        await allocator.allocate(spec)
    except Exception as e:
        # this will always fail because no such test exists
        assert "not found in job" in str(e)


@run_async
async def test_mock_mast_allocator() -> None:
    mock = hyperactor_meta.MockMast()
    await mock.add_local_task_group("test_task_group", 2)
    allocator = await mock.get_mast_allocator(
        hyperactor_meta.MastAllocatorConfig(job_name="test_job")
    )
    spec = hyperactor.AllocSpec(
        hyperactor.AllocConstraints(
            {hyperactor_meta.MastAllocator.ALLOC_LABEL_TASK_GROUP: "test_task_group"}
        ),
        host=2,
        gpu=2,
    )
    alloc = await allocator.allocate(spec)
    proc_mesh = await hyperactor.ProcMesh.allocate(alloc)
    actor_mesh = await proc_mesh.spawn("test", MyActor)

    assert actor_mesh.get(0) is not None
    assert actor_mesh.get(1) is not None
    assert actor_mesh.get(2) is not None
    assert actor_mesh.get(3) is not None
    assert actor_mesh.get(4) is None

    assert isinstance(actor_mesh.client, hyperactor.Mailbox)
