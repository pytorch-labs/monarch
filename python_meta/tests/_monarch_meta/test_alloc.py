# pyre-strict

import asyncio
from asyncio import AbstractEventLoop
from typing import Any

from monarch._monarch import hyperactor
from monarch_meta._monarch_meta import hyperactor_meta

# have to use a single loop for all tests, otherwise there are
# loop closed errors.
loop: AbstractEventLoop = asyncio.get_event_loop()


# pyre-ignore[2,3]
def run_async(x: Any) -> Any:
    return lambda: loop.run_until_complete(x())


# dummy test to check if the bindings work.
@run_async
async def test_mast_allocator() -> None:
    hyperactor.init_asyncio_loop()
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
