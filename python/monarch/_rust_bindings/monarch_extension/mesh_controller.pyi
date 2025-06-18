# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from traceback import FrameSummary
from typing import List, NamedTuple, Sequence, Tuple, Union

from monarch._rust_bindings.monarch_extension import client
from monarch._rust_bindings.monarch_hyperactor.mailbox import PortId
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh

from monarch._rust_bindings.monarch_hyperactor.shape import Slice as NDSlice

class _Controller:
    def __init__(self) -> None: ...
    def node(
        self,
        seq: int,
        defs: Sequence[object],
        uses: Sequence[object],
        port: Tuple[PortId, NDSlice] | None,
        tracebacks: List[List[FrameSummary]],
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
    def exit(self, seq: Seq) -> None:
        """
        Treat seq as a barrier for exit. It will recieve None on succesfully reaching
        seq, and throw an exception if there remote failures that were never reported to a future.
        """
        ...
