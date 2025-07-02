# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import final

from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh

from monarch._rust_bindings.monarch_hyperactor.shape import Shape

class WorkspaceLocation:
    """
    Python binding for the Rust WorkspaceLocation enum.
    """
    @final
    class Constant(WorkspaceLocation):
        def __init__(self, path) -> None: ...

    @final
    class FromEnvVar(WorkspaceLocation):
        def __init__(self, var) -> None: ...

    def resolve(self) -> Path:
        """
        Resolve the workspace location to a Path.
        """
        ...

@final
class RsyncMeshClient:
    """
    Python binding for the Rust RsyncMeshClient.
    """
    @staticmethod
    def spawn_blocking(
        proc_mesh: ProcMesh,
        shape: Shape,
        local_workspace: str,
        remote_workspace: WorkspaceLocation,
    ) -> RsyncMeshClient: ...
    async def sync_workspace(self) -> None: ...
