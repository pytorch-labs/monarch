# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final

from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh

from monarch._rust_bindings.monarch_hyperactor.shape import Shape

class RemoteWorkspace:
    """
    Python binding for the Rust RemoteWorkspace enum.
    """
    @final
    class Constant(RemoteWorkspace):
        def __init__(self, path) -> None: ...

    @final
    class FromEnvVar(RemoteWorkspace):
        def __init__(self, var) -> None: ...

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
        remote_workspace: RemoteWorkspace,
    ) -> RsyncMeshClient: ...
    async def sync_workspace(self) -> None: ...
