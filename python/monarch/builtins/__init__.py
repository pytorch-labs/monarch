# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""
Builtins for Monarch is a set of remote function defintions for PyTorch functions and other utilities.
"""

from .log import log_remote, set_logging_level_remote
from .misc import get_default_dtype_remote, set_default_dtype_remote

__all__ = [
    "log_remote",
    "set_logging_level_remote",
    "get_default_dtype_remote",
    "set_default_dtype_remote",
]
