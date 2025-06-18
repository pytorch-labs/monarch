# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from functools import partial
from typing import Generator, Generic, Optional, TypeVar

R = TypeVar("R")


def _incomplete(impl, self):
    try:
        return self._set_result(impl())
    except Exception as e:
        self._set_exception(e)
        raise


async def _aincomplete(impl, self):
    try:
        return self._set_result(await impl())
    except Exception as e:
        self._set_exception(e)
        raise


# TODO: consolidate with monarch.common.future
class ActorFuture(Generic[R]):
    def __init__(self, impl, blocking_impl=None):
        if blocking_impl is None:
            blocking_impl = partial(asyncio.run, impl())
        self._get = partial(_incomplete, blocking_impl)
        self._aget = partial(_aincomplete, impl)

    def get(self, timeout: Optional[float] = None) -> R:
        if timeout is not None:
            return asyncio.run(asyncio.wait_for(self._aget(self), timeout))
        return self._get(self)

    def __await__(self) -> Generator[R, None, R]:
        return self._aget(self).__await__()

    def _set_result(self, result):
        def f(self):
            return result

        async def af(self):
            return result

        self._get, self._aget = f, af
        return result

    def _set_exception(self, e):
        def f(self):
            raise e

        async def af(self):
            raise e

        self._get, self._aget = f, af

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
