# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deprecated shim for monarch.common.invocation.

This module has been moved to monarch._src.tensor_engine.common.invocation.
This shim is provided for backward compatibility.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "monarch.common.invocation has been moved to monarch._src.tensor_engine.common.invocation. "
    "Please update your imports. This shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from monarch._src.tensor_engine.common.invocation import *  # noqa: F401, F403, E402
