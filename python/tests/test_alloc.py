# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from unittest import IsolatedAsyncioTestCase

from monarch import ProcessAllocator
from monarch._rust_bindings.hyperactor_extension import (  # @manual=//monarch/monarch_extension:monarch_extension
    AllocConstraints,
    AllocSpec,
)


class TestAlloc(IsolatedAsyncioTestCase):
    async def test_basic(self) -> None:
        cmd = "echo hello"
        allocator = ProcessAllocator(cmd)
        spec = AllocSpec(AllocConstraints(), replica=2)
        alloc = await allocator.allocate(spec)

        print(alloc)
