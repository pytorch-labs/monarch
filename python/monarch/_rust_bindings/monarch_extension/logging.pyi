# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh

@final
class LoggingMeshClient:
    """
    Python binding for the Rust LoggingMeshClient.
    """
    @staticmethod
    async def spawn(
        proc_mesh: ProcMesh,
    ) -> LoggingMeshClient: ...
    def set_mode(self, stream_to_client: bool) -> None: ...
