# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from contextlib import contextmanager
from typing import Generator, Optional

import monarch._src.tensor_engine.common._C  # @manual=//monarch/python/monarch/_src/tensor_engine/common:_C
import torch

monarch._src.tensor_engine.common._C.patch_cuda()

_mock_cuda_stream: Optional[torch.cuda.Stream] = None


def get_mock_cuda_stream() -> torch.cuda.Stream:
    global _mock_cuda_stream
    if _mock_cuda_stream is None:
        _mock_cuda_stream = torch.cuda.Stream()
    return _mock_cuda_stream


@contextmanager
def mock_cuda_guard() -> Generator[None, None, None]:
    try:
        with torch.cuda.stream(get_mock_cuda_stream()):
            monarch._src.tensor_engine.common._C.mock_cuda()
            yield
    finally:
        monarch._src.tensor_engine.common._C.unmock_cuda()


def mock_cuda() -> None:
    monarch._src.tensor_engine.common._C.mock_cuda()


def unmock_cuda() -> None:
    monarch._src.tensor_engine.common._C.unmock_cuda()
