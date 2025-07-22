# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import traceback
from functools import partial
from typing import Generator, Generic, Optional, TypeVar

R = TypeVar("R")


async def _aincomplete(impl, self):
    try:
        return self._set_result(await impl())
    except Exception as e:
        self._set_exception(e)
        raise


# Future is our generic mechanism for providing both a synchronous and asynchronous API for
# Monarch Future objects.

# We treat all code as running in one of two contexts: synchronous (asyncio._get_running_loop() is None)
# or asynchronous.

# Inside of asynchronous code, clients of our API must use `await` to wait for monarch Futures to prevent
# blocking the surrounding event loop.

# In synchronous code users must call get() because the call is comming from an non-async function so
# await is not allowed.

# [avoiding async code duplication]
# Because we allow for two modes, it is tempting as developers of Monarch to start to write two copies of
# of code for each mode. However, this results in a lot of confusing code duplication.
# To avoid this, we utilize the fact that synchronous code is allowed to start/complete an asyncio event loop
# via asyncio.run in order to complete the `get()` operation. So we can just write the async version and use
# it to implement the synchronoous version.

# However, starting and running an event loop is somewhat expensive. For simple messages, using an event loop
# is about 4x slower than just directly waiting on the tokio result. To avoid this slow down we perform an
# optimization. For any case where the `impl` coroutine of a future calls `await` only on PythonFuture
# (a Tokio future returning a Python value) objects, we pass requires_loop=False to the Future. In this mode,
# the future will just run the coroutine manually, and the PythonFuture object will recognize it is being awaited
# without an event loop (search [avoiding code duplication]) and simply do a blocking wait. By avoiding the event
# loop machinery, this gives it the same throughput as if we ran it synchronously.


class Future(Generic[R]):
    def __init__(self, *, impl, requires_loop=True):
        self._aget = partial(_aincomplete, impl)
        self._requires_loop = requires_loop

    def get(self, timeout: Optional[float] = None) -> R:
        if asyncio._get_running_loop() is not None:
            raise RuntimeError("get() cannot be called from within an async context")
        if timeout is not None:
            return asyncio.run(asyncio.wait_for(self._aget(self), timeout))
        if not self._requires_loop:
            try:
                coro = self._aget(self)
                next(coro.__await__())
                tb_str = "".join(traceback.format_stack(coro.cr_frame))
                raise RuntimeError(
                    f"a coroutine paused with a future with requires_loop=False cannot block on a python asyncio.Future. Use requires_loop=True.\n{tb_str}"
                )
            except StopIteration as e:
                return e.value
        return asyncio.run(self._aget(self))

    def __await__(self) -> Generator[R, None, R]:
        return self._aget(self).__await__()

    def _set_result(self, result):
        async def af(self):
            return result

        self._aget = af
        return result

    def _set_exception(self, e):
        async def af(self):
            raise e

        self._aget = af

    # compatibility with old tensor engine Future objects
    # hopefully we do not need done(), add_callback because
    # they are harder to implement right.
    def result(self, timeout: Optional[float] = None) -> R:
        return self.get(timeout)

    def exception(self, timeout: Optional[float] = None):
        try:
            self.get(timeout)
            return None
        except Exception as e:
            return e
