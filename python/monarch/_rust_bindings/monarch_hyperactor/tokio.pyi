# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
from typing import Generator, Generic, TypeVar

T = TypeVar("T")

class PythonTask(Generic[T]):
    """
    A tokio::Future whose result returns a python object.
    """
    def into_future(self) -> asyncio.Future[T]:
        """
        Return an asyncio.Future that can be awaited to get the result of this task.
        Consumes the PythonTask object.
        """
        ...

    def block_on(self) -> T:
        """
        Synchronously wait on the result of this task, returning the result.
        Consumes the PythonTask object.
        """
        ...

    def __await__(self) -> Generator[T, None, T]: ...
