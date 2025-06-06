# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import traceback
from collections import deque
from logging import Logger
from typing import List, NamedTuple, Optional, Union

import torch.utils._python_dispatch

from monarch import NDSlice
from monarch._rust_bindings.monarch_extension import client, debugger
from monarch._rust_bindings.monarch_extension.client import (  # @manual=//monarch/monarch_extension:monarch_extension
    WorldState,
)
from monarch._rust_bindings.monarch_extension.mesh_controller import _Controller
from monarch._rust_bindings.monarch_hyperactor.proc import (  # @manual=//monarch/monarch_extension:monarch_extension
    ActorId,
)
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh as HyProcMesh
from monarch._rust_bindings.monarch_messages.debugger import DebuggerAction
from monarch.common.client import Client
from monarch.common.controller_api import LogMessage, MessageResult
from monarch.common.device_mesh import DeviceMesh, no_mesh
from monarch.common.invocation import DeviceException, RemoteException
from monarch.controller.debugger import read as debugger_read, write as debugger_write
from monarch.proc_mesh import ProcMesh
from pyre_extensions import none_throws

logger: Logger = logging.getLogger(__name__)


class Controller(_Controller):
    def __init__(self, workers: HyProcMesh) -> None:
        super().__init__()
        # Buffer for messages unrelated to debugging that are received while a
        # debugger session is active.
        self._non_debugger_pending_messages: deque[
            Optional[client.LogMessage | client.WorkerResponse]
        ] = deque()
        self._pending_debugger_sessions: deque[ActorId] = deque()

    def next_message(
        self, timeout: Optional[float]
    ) -> Optional[LogMessage | MessageResult]:
        if self._non_debugger_pending_messages:
            msg = self._non_debugger_pending_messages.popleft()
        else:
            msg = self._get_next_message(timeout_msec=int((timeout or 0.0) * 1000.0))
        if msg is None:
            return None

        if isinstance(msg, client.WorkerResponse):
            return _worker_response_to_result(msg)
        elif isinstance(msg, client.LogMessage):
            return LogMessage(msg.level, msg.message)
        elif isinstance(msg, client.DebuggerMessage):
            self._run_debugger_loop(msg)

    def send(
        self,
        ranks: Union[NDSlice, List[NDSlice]],
        msg: NamedTuple,
    ) -> None:
        with torch.utils._python_dispatch._disable_current_modes():
            return super().send(ranks, msg)

    def drain_and_stop(
        self,
    ) -> List[LogMessage | MessageResult | client.DebuggerMessage]:
        logger.info("rust controller shutting down")
        results = []
        for msg in self._drain_and_stop():
            if isinstance(msg, client.WorkerResponse):
                results.append(_worker_response_to_result(msg))
            elif isinstance(msg, client.LogMessage):
                results.append(LogMessage(msg.level, msg.message))
            elif isinstance(msg, client.DebuggerMessage):
                results.append(msg)
            else:
                raise RuntimeError(f"Unexpected message type {type(msg)}")
        return results

    def _run_debugger_loop(self, message: client.DebuggerMessage) -> None:
        if not isinstance(message.action, DebuggerAction.Paused):
            raise RuntimeError(
                f"Unexpected debugger message {message} when no debugger session is running"
            )

        self._pending_debugger_sessions.append(message.debugger_actor_id)
        while self._pending_debugger_sessions:
            debugger_actor_id = self._pending_debugger_sessions.popleft()
            rank = debugger_actor_id.rank
            proc_id = debugger_actor_id.proc_id
            debugger_write(
                f"pdb attached to proc {proc_id} with rank {rank}, debugger actor {debugger_actor_id} \n"
            )

            self._debugger_attach(debugger_actor_id)
            while True:
                # TODO: Add appropriate timeout.
                msg = self._get_next_message(timeout_msec=None)

                if not isinstance(msg, client.DebuggerMessage):
                    self._non_debugger_pending_messages.append(msg)
                    continue

                if msg.debugger_actor_id != debugger_actor_id:
                    if isinstance(msg.action, DebuggerAction.Paused):
                        self._pending_debugger_sessions.append(msg.debugger_actor_id)
                        continue
                    else:
                        raise RuntimeError(
                            f"unexpected debugger message {msg} from rank {msg.debugger_actor_id.rank} "
                            f"when debugging rank {debugger_actor_id.rank}"
                        )

                action = msg.action
                if isinstance(action, DebuggerAction.Detach):
                    break
                elif isinstance(action, DebuggerAction.Read):
                    self._debugger_write(
                        debugger_actor_id, debugger_read(action.requested_size)
                    )
                elif isinstance(action, DebuggerAction.Write):
                    debugger_write(
                        debugger.get_bytes_from_write_action(action).decode()
                    )
                else:
                    raise RuntimeError(
                        f"unexpected debugger message {msg} when debugging rank {debugger_actor_id.rank}"
                    )

    def worker_world_state(self) -> WorldState:
        raise NotImplementedError("worker world state")

    def stop_mesh(self):
        # I think this is a noop?

        pass


# TODO: Handling conversion of the response can move to a separate module over time
# especially as we have structured error messages.
def _worker_response_to_result(result: client.WorkerResponse) -> MessageResult:
    if not result.is_exception():
        # The result of the message needs to be unwrapped on a real device.
        # Staying as a fake tensor will fail the tensor deserialization.
        with no_mesh.activate():
            return MessageResult(result.seq, result.result(), None)
    exc = none_throws(result.exception())
    if isinstance(exc, client.Error):
        worker_frames = [
            traceback.FrameSummary("<unknown>", None, frame)
            for frame in exc.backtrace.split("\\n")
        ]
        logger.error(f"Worker {exc.actor_id} failed")
        return MessageResult(
            seq=result.seq,
            result=None,
            error=RemoteException(
                seq=exc.caused_by_seq,
                exception=RuntimeError(exc.backtrace),
                controller_frame_index=0,  # TODO: T225205291 fix this once we have recording support in rust
                controller_frames=None,
                worker_frames=worker_frames,
                source_actor_id=exc.actor_id,
                message=f"Worker {exc.actor_id} failed",
            ),
        )
    elif isinstance(exc, client.Failure):
        frames = [
            traceback.FrameSummary("<unknown>", None, frame)
            for frame in exc.backtrace.split("\n")
        ]
        reason = f"Actor {exc.actor_id} crashed on {exc.address}, check the host log for details"
        logger.error(reason)
        return MessageResult(
            seq=0,  # seq is not consumed for DeviceException; it will be directly thrown by the client
            result=None,
            error=DeviceException(
                exception=RuntimeError(reason),
                frames=frames,
                source_actor_id=exc.actor_id,
                message=reason,
            ),
        )
    else:
        raise RuntimeError(f"Unknown exception type: {type(exc)}")


def spawn_tensor_engine(proc_mesh: ProcMesh) -> DeviceMesh:
    # This argument to Controller
    # is currently only used for debug printing. It should be fixed to
    # report the proc ID instead of the rank it currently does.
    gpus = proc_mesh.sizes.get("gpus", 1)
    backend_ctrl = Controller(proc_mesh._proc_mesh)
    client = Client(backend_ctrl, proc_mesh.size(), gpus)
    dm = DeviceMesh(
        client,
        NDSlice.new_row_major(list(proc_mesh.sizes.values())),
        tuple(proc_mesh.sizes.keys()),
    )
    dm.exit = lambda: client.shutdown()
    return dm
