# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from monarch.common.remote import remote

"""
Remote functions that return ints/bools are currently wrapped in tensors.
They should be unwrapped once there is broader return type support.
"""


@remote(propagate=lambda: torch.tensor(0))
def device_count() -> torch.Tensor:
    return torch.tensor(torch.cuda.device_count())


@remote(propagate="inspect")
def empty_cache() -> None:
    torch.cuda.empty_cache()


@remote(propagate=lambda: torch.tensor(0))
def memory_allocated() -> torch.Tensor:
    return torch.tensor(torch.cuda.memory_allocated())


@remote(propagate=lambda: torch.tensor(0))
def memory_reserved() -> torch.Tensor:
    return torch.tensor(torch.cuda.memory_reserved())


@remote(propagate=lambda: torch.tensor(int(torch.cuda.is_available())))
def is_available() -> torch.Tensor:
    return torch.tensor(int(torch.cuda.is_available()))
