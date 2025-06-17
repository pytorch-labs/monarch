# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, NamedTuple, Sequence, Union

from monarch._rust_bindings.monarch_extension import client
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh

from monarch._rust_bindings.monarch_hyperactor.shape import Slice as NDSlice

class _Controller:
    def __init__(self) -> None: ...
    def node(
        self, seq: int, defs: Sequence[object], uses: Sequence[object]
    ) -> None: ...
    def drop_refs(self, refs: Sequence[object]) -> None: ...
    def send(
        self,
        ranks: Union[NDSlice, List[NDSlice]],
        msg: NamedTuple,
    ) -> None: ...
    def _get_next_message(
        self, *, timeout_msec: int | None = None
    ) -> client.WorkerResponse | client.DebuggerMessage | None: ...
    def _debugger_attach(self, debugger_actor_id: ActorId) -> None: ...
    def _debugger_write(self, debugger_actor_id: ActorId, data: bytes) -> None: ...
    def _drain_and_stop(
        self,
    ) -> List[client.LogMessage | client.WorkerResponse | client.DebuggerMessage]: ...
