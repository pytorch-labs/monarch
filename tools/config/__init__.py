# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass, field
from typing import Any, Optional


NOT_SET: str = "__NOT_SET__"


@dataclass
class Config:
    scheduler: str = NOT_SET
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    workspace: Optional[str] = None
    dryrun: bool = False
