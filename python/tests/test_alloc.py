# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from unittest import IsolatedAsyncioTestCase

from monarch import ProcessAllocator

from monarch._monarch import hyperactor


class TestAlloc(IsolatedAsyncioTestCase):
    async def test_basic(self) -> None:
        cmd = "echo hello"
        allocator = ProcessAllocator(cmd)
        spec = hyperactor.AllocSpec(hyperactor.AllocConstraints(), replica=2)
        alloc = await allocator.allocate(spec)

        print(alloc)
