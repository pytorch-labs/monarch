# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import logging
import os
import time
import traceback
from collections import deque
from logging import Logger
from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union

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

if TYPE_CHECKING:
    from monarch._rust_bindings.monarch_hyperactor.proc_mesh import (
        ProcMesh as HyProcMesh,
    )
    from monarch.proc_mesh import ProcMesh

from monarch._rust_bindings.monarch_hyperactor.shape import Point

from monarch._rust_bindings.monarch_messages.debugger import DebuggerAction
from monarch.common.client import Client
from monarch.common.controller_api import LogMessage, MessageResult
from monarch.common.device_mesh import DeviceMesh, no_mesh
from monarch.common.invocation import DeviceException, RemoteException
from monarch.controller.debugger import read as debugger_read, write as debugger_write
from monarch.rust_local_mesh import _get_worker_exec_info
from pyre_extensions import none_throws

logger: Logger = logging.getLogger(__name__)


class Controller(_Controller):
    def __init__(self, workers: "HyProcMesh") -> None:
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
        self._drain_and_stop()
        return []

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
                message=f"Remote function in {exc.actor_id} errored.",
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


def _initialize_env(worker_point: Point, proc_id: str) -> None:
    worker_rank = worker_point.rank
    try:
        _, worker_env = _get_worker_exec_info()
        local_rank = worker_point["gpus"]
        gpus_per_host = worker_point.size("gpus")
        num_worker_procs = len(worker_point.shape)
        process_env = {
            **worker_env,
            "HYPERACTOR_MANAGED_SUBPROCESS": "1",
            "CUDA_VISIBLE_DEVICES": str(local_rank),
            "NCCL_HOSTID": f"{proc_id}_host_{worker_rank // gpus_per_host}",
            # This is needed to avoid a hard failure in ncclx when we do not
            # have backend topology info (eg. on RE).
            "NCCL_IGNORE_TOPO_LOAD_FAILURE": "true",
            "LOCAL_RANK": str(local_rank),
            "RANK": str(worker_rank),
            "WORLD_SIZE": str(num_worker_procs),
            "LOCAL_WORLD_SIZE": str(gpus_per_host),
        }
        os.environ.update(process_env)
    except Exception:
        traceback.print_exc()
        raise


class MeshClient(Client):
    def shutdown(
        self,
        destroy_pg: bool = True,
        error_reason: Optional[RemoteException | DeviceException | Exception] = None,
    ):
        # return
        if self.has_shutdown:
            return
        logger.info("shutting down the client gracefully")

        atexit.unregister(self._atexit)
        self._shutdown = True

        # ensure all pending work is finished.
        # all errors must be messaged back at this point
        self.new_node_nocoalesce([], [], None, [])
        self._request_status()

        ttl = 60
        start_time = time.time()
        end_time = start_time + ttl
        while ttl > 0 and self.last_assigned_seq > self.last_processed_seq:
            ttl = end_time - time.time()
            self.handle_next_message(ttl)
            if self._pending_shutdown_error:
                raise self._pending_shutdown_error

        if ttl <= 0:
            raise RuntimeError("shutdown timed out")

        # we are not expecting anything more now, because we already
        # waited for the responses
        self.inner.drain_and_stop()


def spawn_tensor_engine(proc_mesh: "ProcMesh") -> DeviceMesh:
    # This argument to Controller
    # is currently only used for debug printing. It should be fixed to
    # report the proc ID instead of the rank it currently does.
    gpus = proc_mesh.sizes.get("gpus", 1)
    backend_ctrl = Controller(proc_mesh._proc_mesh)
    client = MeshClient(backend_ctrl, proc_mesh.size(), gpus)
    dm = DeviceMesh(
        client,
        NDSlice.new_row_major(list(proc_mesh.sizes.values())),
        tuple(proc_mesh.sizes.keys()),
    )
    dm.exit = lambda: client.shutdown()
    return dm
