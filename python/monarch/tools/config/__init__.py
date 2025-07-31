# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass, field

from enum import Enum
from typing import Any, Dict, List

from torchx.specs import Role


class EmptyWorkspaceOption(Enum):
    """
    Allow users to specify the behavior when no local workspace is needed.

    Attributes:
        NO_WORKSPACE: Do not use any workspace. The main function is provided
            by the client script.
        USE_PROVIDED_IMAGE: Use a pre-built container image that has been
            provided in the configuration.
    """

    NO_WORKSPACE = 1
    USE_PROVIDED_IMAGE = 2


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

    This class encapsulates the configuration parameters required to schedule
    and run jobs using the Monarch framework.

    Attributes:
        scheduler: The name of the scheduler to use for job execution.
            Defaults to NOT_SET.
        scheduler_args: Additional arguments to pass to the scheduler.
        workspace: Specifies how the workspace should be configured.
            Can be a string to represent the directory location or
            a EmptyWorkspaceOption enum value when no local workspace is needed.
            A new package will be built if a workspace is explicitly provided
            or if no workspace is needed.
        dryrun: When True, performs a dry run without executing the job.
        appdef: The application definition containing roles and metadata.
    """

    scheduler: str = NOT_SET
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    workspace: str | EmptyWorkspaceOption = EmptyWorkspaceOption.NO_WORKSPACE
    dryrun: bool = False
    appdef: UnnamedAppDef = field(default_factory=UnnamedAppDef)
