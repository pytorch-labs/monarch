# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

warnings.warn(
    "monarch.rdma is deprecated, please import from monarch.tensor_engine.rdma instead.",
    DeprecationWarning,
    stacklevel=2,
)

from monarch.tensor_engine import *  # noqa
