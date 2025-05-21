# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import contextlib
import io
from typing import Generator


@contextlib.contextmanager
def capture_stdout() -> Generator[io.StringIO, None, None]:
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        yield buf
