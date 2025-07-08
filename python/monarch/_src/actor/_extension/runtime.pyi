# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Type hints for the runtime module.
"""

def sleep_indefinitely_for_unit_tests() -> None:
    """
    A test function that sleeps indefinitely in a loop.
    This is used for testing signal handling in signal_safe_block_on.
    The function will sleep forever until interrupted by a signal.

    Raises:
        KeyboardInterrupt: When interrupted by a signal like SIGINT
    """
    ...
