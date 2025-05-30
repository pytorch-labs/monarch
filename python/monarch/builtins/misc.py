# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from monarch.common.remote import remote


@remote(propagate=lambda dtype: torch.set_default_dtype(dtype))
def set_default_dtype_remote(dtype: torch.dtype) -> None:
    torch.set_default_dtype(dtype)


@remote(propagate=lambda: torch.get_default_dtype())
def get_default_dtype_remote() -> torch.dtype:
    return torch.get_default_dtype()
