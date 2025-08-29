# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

from monarch.tools.config.workspace import Workspace

# Defer the import of Role to avoid requiring torchx at import time
if TYPE_CHECKING:
    from torchx.specs import Role


NOT_SET: str = "__NOT_SET__"


@dataclass
class UnnamedAppDef:
    """
    A TorchX AppDef without a name.
    """

    roles: List["Role"] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """
    All configs needed to schedule a mesh of allocators.
    """

    scheduler: str = NOT_SET
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    workspace: Workspace = field(default_factory=Workspace.null)
    dryrun: bool = False
    appdef: UnnamedAppDef = field(default_factory=UnnamedAppDef)

    def __post_init__(self) -> None:
        # workspace used to be Optional[str]
        # while we type it as class Workspace now, handle workspace=None and str for BC
        if self.workspace is None:
            deprecation_msg = "Setting `workspace=None` is deprecated."
            " Use `workspace=monarch.tools.config.workspace.Workspace(env=None)` instead."
            warnings.warn(deprecation_msg, FutureWarning, stacklevel=2)
            self.workspace = Workspace.null()
        elif isinstance(self.workspace, str):
            deprecation_msg = f"Setting `workspace='{self.workspace}'` is deprecated."
            f" Use `workspace=monarch.tools.config.workspace.Workspace(dirs=['{self.workspace}'])` instead."
            warnings.warn(deprecation_msg, FutureWarning, stacklevel=2)
            # previous behavior (when workspace was a str pointing to the local project dir)
            # was to copy the local dir into $WORKSPACE_DIR. For example:
            # ~/github/torch/** (local) -> $WORKSPACE_DIR/** (remote)
            # so we map it to "".
            self.workspace = Workspace(dirs={self.workspace: ""})
