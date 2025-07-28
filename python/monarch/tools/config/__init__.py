# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torchx.specs import Role


NOT_SET: str = "__NOT_SET__"


@dataclass
class UnnamedAppDef:
    """
    A TorchX AppDef without a name.
    """

    roles: List[Role] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """
    All configs needed to schedule a mesh of allocators.
    """

    scheduler: str = NOT_SET
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    workspace: Optional[str] = None
    dryrun: bool = False
    appdef: UnnamedAppDef = field(default_factory=UnnamedAppDef)
