# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time

from monarch._rust_bindings.monarch_hyperactor.actor import (
    MethodSpecifier,
    PythonMessage,
    PythonMessageKind,
)


def test_python_message() -> None:
    """
    Verifies that PythonMessage can be constructed reasonably fast.
    """
    method: str = "test_method"
    payload: str = "a" * 2**30  # 1gb
    blob: bytes = payload.encode("utf-8")
    t = time.time()
    PythonMessage(
        PythonMessageKind.CallMethod(MethodSpecifier.ReturnsResponse(method), None),
        blob,
    )
    t_spent = time.time() - t
    assert t_spent < 1
