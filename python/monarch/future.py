import asyncio
from typing import Generic, TypeVar

R = TypeVar("R")


# TODO: consolidate with monarch.common.future
class ActorFuture(Generic[R]):
    def __init__(self, impl, blocking_impl=None):
        self._impl = impl
        self._blocking_impl = blocking_impl

    def get(self) -> R:
        if self._blocking_impl is not None:
            return self._blocking_impl()
        return asyncio.run(self._impl())

    def __await__(self):
        return self._impl().__await__()
