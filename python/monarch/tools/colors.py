# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys

# only print colors if outputting directly to a terminal
if not sys.stdout.closed and sys.stdout.isatty():
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    ORANGE = "\033[38:2:238:76:44m"
    GRAY = "\033[2m"
    CYAN = "\033[36m"
    ENDC = "\033[0m"
else:
    GREEN = ""
    ORANGE = ""
    BLUE = ""
    GRAY = ""
    CYAN = ""
    ENDC = ""
