# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import importlib.resources
import logging
import os
import random
import re
import select
import socket
import string
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import (
    Callable,
    Collection,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    TextIO,
    Tuple,
    Type,
    TypeVar,
)

from monarch._rust_bindings.controller.bootstrap import (
    ControllerCommand,
    ControllerServerRequest,
    ControllerServerResponse,
    RunCommand,
)

from monarch._rust_bindings.monarch_hyperactor.proc import (  # @manual=//monarch/monarch_extension:monarch_extension
    ActorId,
)

from monarch._rust_bindings.monarch_tensor_worker.bootstrap import (
    WorkerServerRequest,
    WorkerServerResponse,
)

from monarch.common.device_mesh import DeviceMesh
from monarch.common.fake import fake_call
from monarch.common.invocation import DeviceException, RemoteException
from monarch.rust_backend_mesh import (
    IBootstrap,
    MeshWorld,
    PoolDeviceMeshProvider,
    rust_backend_mesh_provider,
    rust_backend_meshes,
)

logger: logging.Logger = logging.getLogger(__name__)
_MONARCH_TENSOR_WORKER_MAIN = "monarch.tensor_worker_main"

try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = bool(fbmake.get("par_style"))
except ImportError:
    IN_PAR = False


class SocketType(Enum):
    """Enum representing socket types."""

    TCP = "tcp"
    UNIX = "unix"


class LoggingLocation(Enum):
    """Enum representing where to flush stderr and stdout."""

    DEFAULT = "default"
    FILE = "file"


class SupervisionParams(NamedTuple):
    # If system actor does not receive supervision update within this time,
    # it will treate this proc as dead.
    update_timeout_in_sec: int
    # How often controller queries supervision status from system actor.
    query_interval_in_sec: int
    # How often proc actor sends supervision update to system actor.
    update_interval_in_sec: int


class ControllerParams(NamedTuple):
    # How often the controller will poll for operations that have not completed within a timeout duration
    # indicating that it may be stuck.
    worker_progress_check_interval_in_sec: int

    # How long we will wait for an operation before letting the client know that it may be stuck.
    operation_timeout_in_sec: int

    # The number of operations invoked before we proactively check worker progress. If a large number
    # of operations are invoked all at once, it is expected that it will take a while for all operations
    # to complete so we want to inject progress requests at a higher frequency to check if we are making progress
    operations_per_worker_progress_request: int

    # If the controller should propagate a failure to the client if the workers become stuck.
    fail_on_worker_timeout: bool


_PROC_ENV: dict[str, str] = {}


def get_controller_main() -> tuple[Path, dict[str, str]]:
    with (
        importlib.resources.as_file(
            importlib.resources.files("monarch") / "monarch_controller"
        ) as controller_main,
    ):
        if not controller_main.exists():
            if IN_PAR:
                raise ImportError(
                    "Monarch env not found, please define a custom 'monarch_env' or "
                    "add '//monarch/python/monarch:default_env-library' to your binary dependencies "
                    "in TARGETS"
                )
            else:
                raise ImportError(
                    "Monarch env not found, please re-run ./scripts/install.sh in fbcode/monarch"
                )
        env: dict[str, str] = {}

        # Hack to make exploded wheel workflow work in the face of broken
        # build-time RPATHs...
        #
        # If we're running under a conda env...
        if not IN_PAR:
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix is not None and sys.executable.startswith(
                conda_prefix + "/"
            ):
                # .. and Monarch is coming from "outside" the env, via `PYTHONPATH`s ...
                spec = importlib.util.find_spec("monarch")
                assert spec is not None
                origin = spec.origin
                assert origin is not None
                monarch_root = str(Path(origin).parent.parent)
                if (
                    not monarch_root.startswith(conda_prefix + "/")
                    and monarch_root in sys.path
                ):
                    import torch

                    # then assume we're running via exploded .whl, which means
                    # we need to manually set library paths to find the necessary
                    # native libs from the conda env.
                    env["LD_LIBRARY_PATH"] = ":".join(
                        [
                            os.path.join(os.path.dirname(torch.__file__), "lib"),
                            os.path.join(conda_prefix, "lib"),
                        ]
                    )

        return controller_main, env


def _create_logging_locations(
    logging_dir: str, name: str, logging_location: LoggingLocation
) -> tuple[TextIO | None, TextIO | None]:
    if logging_location == LoggingLocation.FILE:
        stdout_file: TextIO = open(os.path.join(logging_dir, f"{name}.stdout"), "a+")
        stderr_file: TextIO = open(os.path.join(logging_dir, f"{name}.stderr"), "a+")
        return stdout_file, stderr_file
    elif logging_location == LoggingLocation.DEFAULT:
        return None, None
    else:
        raise ValueError(f"Unknown logging location: {logging_location}")


def _get_labels(flag_name: str, labels: Dict[str, str]) -> List[str]:
    params = []
    for k, v in labels.items():
        assert k not in params, f"Duplicate label: {k}"
        assert "=" not in k, f"Key cannot contain '=': {k}"
        params.append(f"--{flag_name}")
        params.append(f"{k}={v}")
    return params


def _start_worker_cmd(
    *,
    world_uuid: str,
    worker_rank: int,
    gpus_per_host: int,
    num_worker_procs: int,
    args: list[str],
    env: dict[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | int | None = subprocess.DEVNULL,
    pass_fds: Collection[int] = (),
) -> subprocess.Popen[bytes]:
    worker_cmd, worker_env = _get_worker_exec_info()
    local_rank = worker_rank % gpus_per_host
    process_env = {
        **(_PROC_ENV | worker_env),
        "CUDA_VISIBLE_DEVICES": str(local_rank),
        "NCCL_HOSTID": f"{world_uuid}_host_{worker_rank // gpus_per_host}",
        # This is needed to avoid a hard failure in ncclx when we do not
        # have backend topology info (eg. on RE).
        "NCCL_IGNORE_TOPO_LOAD_FAILURE": "true",
        "LOCAL_RANK": str(local_rank),
        "RANK": str(worker_rank),
        "WORLD_SIZE": str(num_worker_procs),
        "LOCAL_WORLD_SIZE": str(gpus_per_host),
        **os.environ,
    }
    cmd = []
    cmd.extend(worker_cmd)
    cmd.extend(args)
    if env is not None:
        process_env.update(env)
    return subprocess.Popen(
        cmd,
        env=process_env,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        pass_fds=pass_fds,
    )


ServerT = TypeVar("ServerT")


class ServerInstance:
    TIMEOUT = 10.0

    def __init__(
        self,
        *,
        server: "ServerBase[ServerT]",
    ) -> None:
        self._server = server
        self._terminated: float = 0

        # TODO
        assert self._server._proc is not None
        self.pid: int = self._server._proc.pid

    def __enter__(self) -> "ServerInstance":
        return self

    def terminate(self) -> None:
        # Start the timeout clock now.
        self._terminated = time.time()

    def kill(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        timeout = max(0, self._terminated + self.TIMEOUT - time.time())
        try:
            self._server._finish(timeout=timeout)
        except Exception as exc:
            if exc_type is None:
                raise
            else:
                logger.warning(f"failed waiting for instance to finish: {exc}")


class ServerBase(contextlib.AbstractContextManager[ServerT, None]):
    def __init__(
        self,
        *,
        name: str,
        response_cls: Type[WorkerServerResponse | ControllerServerResponse],
        request_cls: Type[WorkerServerRequest | ControllerServerRequest],
    ) -> None:
        self._name = name
        self._response_cls: Type[WorkerServerResponse | ControllerServerResponse] = (
            response_cls
        )
        self._request_cls: Type[WorkerServerRequest | ControllerServerRequest] = (
            request_cls
        )

        self._aborted = False
        self._shutdown_started = False
        self._contexts: contextlib.ExitStack[None] = contextlib.ExitStack()
        self._proc: subprocess.Popen[bytes] | None = None
        self._pipe: Tuple[TextIO, TextIO] | None = None
        self._lock: threading.Lock | None = None

    def _send(self, msg: WorkerServerRequest | ControllerServerRequest) -> None:
        logger.debug(f"{self._name}: sending server request: {msg}")
        assert not self._aborted
        assert self._lock is not None
        if not self._lock.acquire(blocking=False):
            raise Exception("server in use")
        assert self._pipe is not None
        self._pipe[1].write(msg.to_json() + "\n")
        assert self._pipe is not None
        self._pipe[1].flush()

    def _recv(
        self, timeout: float | None = None
    ) -> WorkerServerResponse | ControllerServerResponse:
        assert not self._aborted
        assert self._lock is not None
        assert self._lock.locked()
        assert self._pipe is not None
        ready, _, _ = select.select([self._pipe[0]], [], [], timeout)
        if not ready:
            assert self._proc is not None
            assert timeout is not None
            raise subprocess.TimeoutExpired(self._proc.args, timeout)
        output = ready[0].readline()
        logger.info(f"{self._name}: Got response: {output}")
        response = self._response_cls.from_json(output)
        assert self._lock is not None
        self._lock.release()
        logger.debug(f"{self._name}: received response: {response}")
        return response

    def _launch_server(
        self,
        read_fd: int,
        write_fd: int,
    ) -> subprocess.Popen[bytes]:
        raise NotImplementedError()

    def __enter__(self) -> ServerT:
        assert self._proc is None, "already running"
        logger.debug(f"{self._name}: launching worker server")
        self._lock = threading.Lock()
        send = os.pipe2(0)
        recv = os.pipe2(0)
        self._proc = self._contexts.enter_context(
            self._launch_server(
                read_fd=send[0],
                write_fd=recv[1],
            ),
        )
        self._pipe = (
            self._contexts.enter_context(os.fdopen(recv[0], "r")),
            self._contexts.enter_context(os.fdopen(send[1], "w")),
        )
        os.close(send[0])
        os.close(recv[1])
        # pyre-ignore: Incompatible return type [7]
        return self

    def initiate_shutdown(self) -> None:
        if not self._shutdown_started and not self._aborted:
            assert self._lock is not None
            assert not self._lock.locked()
            self._shutdown_started = True
            self._send(self._request_cls.Exit())
            assert self._pipe is not None
            self._pipe[1].close()

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None or self._aborted:
            assert self._proc is not None
            self._proc.kill()
        else:
            # attempt a clean shutdown
            self.initiate_shutdown()
            assert self._proc is not None
            assert self._proc.wait(timeout=5) == 0
        self._contexts.__exit__(exc_type, exc_val, exc_tb)

    def _finish(self, timeout: float | None = None) -> None:
        try:
            response = self._recv(timeout=timeout)
            assert isinstance(response, self._response_cls.Finished), str(response)
            # pyre-ignore: Undefined attribute [16]
            assert response.error is None, response.error
        except:
            self._aborted = True
            raise

    def _launch_instance(
        self,
        *,
        msg: WorkerServerRequest | ControllerServerRequest,
    ) -> ServerInstance:
        self._send(msg)
        return ServerInstance(server=self)


class ISystemFactory:
    def launch(
        self,
        *,
        bootstrap_addr: str,
        supervision_params: SupervisionParams,
    ) -> ServerInstance | subprocess.Popen[bytes]:
        raise NotImplementedError()


class IControllerFactory:
    def launch(
        self,
        *,
        worker_world: str,
        bootstrap_addr: str,
        controller_id: ActorId,
        num_worker_procs: int,
        gpus_per_host: int,
        supervision_params: SupervisionParams,
        controller_params: ControllerParams,
        labels: Dict[str, str],
    ) -> subprocess.Popen[bytes] | ServerInstance:
        raise NotImplementedError()


class ControllerFactoryBase:
    def __init__(
        self,
        *,
        logging_location: LoggingLocation,
        logging_dir: str,
    ) -> None:
        self.logging_location = logging_location
        self.logging_dir = logging_dir

        self.controller_main: Path
        self.controller_env: dict[str, str]
        self.controller_main, self.controller_env = get_controller_main()


class SystemFactory(ControllerFactoryBase, ISystemFactory):
    def launch(
        self,
        *,
        bootstrap_addr: str,
        supervision_params: SupervisionParams,
    ) -> subprocess.Popen[bytes]:
        stdout_location, stderr_location = _create_logging_locations(
            self.logging_dir,
            "system",
            self.logging_location,
        )
        return subprocess.Popen(
            [
                self.controller_main,
                "system",
                "--system-addr",
                bootstrap_addr,
                "--supervision-update-timeout-in-sec",
                str(supervision_params.update_timeout_in_sec),
            ],
            stdout=stdout_location,
            stderr=stderr_location,
            stdin=subprocess.DEVNULL,
            env=_PROC_ENV | self.controller_env,
        )


class ControllerFactory(ControllerFactoryBase, IControllerFactory):
    def launch(
        self,
        *,
        worker_world: str,
        bootstrap_addr: str,
        controller_id: ActorId,
        num_worker_procs: int,
        gpus_per_host: int,
        supervision_params: SupervisionParams,
        controller_params: ControllerParams,
        labels: Dict[str, str],
    ) -> subprocess.Popen[bytes]:
        stdout_location, stderr_location = _create_logging_locations(
            self.logging_dir,
            controller_id.world_name,
            self.logging_location,
        )
        command = [
            self.controller_main,
            "controller",
            "--worker-world",
            worker_world,
            "--system-addr",
            bootstrap_addr,
            "--controller-actor-id",
            str(controller_id),
            "--world-size",
            str(num_worker_procs),
            "--num-procs-per-host",
            str(gpus_per_host),
            "--supervision-query-interval-in-sec",
            str(supervision_params.query_interval_in_sec),
            "--supervision-update-interval-in-sec",
            str(supervision_params.update_interval_in_sec),
            "--worker-progress-check-interval-in-sec",
            str(controller_params.worker_progress_check_interval_in_sec),
            "--operation-timeout-in-sec",
            str(controller_params.operation_timeout_in_sec),
            "--operations-per-worker-progress-request",
            str(controller_params.operations_per_worker_progress_request),
        ]

        if controller_params.fail_on_worker_timeout:
            command.append("--fail-on-worker-timeout")

        return subprocess.Popen(
            command + _get_labels("extra-proc-labels", labels),
            stdout=stdout_location,
            stderr=stderr_location,
            stdin=subprocess.DEVNULL,
            env=_PROC_ENV | self.controller_env,
        )


class ControllerServerBase(ServerBase[ServerT]):
    def __init__(
        self,
        *,
        uuid: str,
        logging_location: LoggingLocation,
        logging_dir: str,
    ) -> None:
        super().__init__(
            name=uuid,
            response_cls=ControllerServerResponse,
            request_cls=ControllerServerRequest,
        )
        self.uuid = uuid
        self.logging_location = logging_location
        self.logging_dir = logging_dir

        self.controller_main: Path
        self.controller_env: dict[str, str]
        self.controller_main, self.controller_env = get_controller_main()

    def _launch_server(
        self,
        read_fd: int,
        write_fd: int,
    ) -> subprocess.Popen[bytes]:
        stdout_location, stderr_location = _create_logging_locations(
            self.logging_dir,
            self.uuid,
            self.logging_location,
        )
        return subprocess.Popen(
            [
                self.controller_main,
                "serve",
                str(read_fd),
                str(write_fd),
            ],
            stdout=stdout_location,
            pass_fds=(read_fd, write_fd),
            stderr=stderr_location,
            stdin=subprocess.DEVNULL,
            env=_PROC_ENV | self.controller_env | dict(os.environ),
        )


class SystemServer(ControllerServerBase["SystemServer"], ISystemFactory):
    def launch(
        self,
        *,
        bootstrap_addr: str,
        supervision_params: SupervisionParams,
    ) -> ServerInstance:
        return self._launch_instance(
            msg=ControllerServerRequest.Run(
                RunCommand.System(
                    system_addr=bootstrap_addr,
                    supervision_update_timeout_in_sec=supervision_params.update_timeout_in_sec,
                    world_eviction_timeout_in_sec=10,
                ),
            ),
        )


class ControllerServer(ControllerServerBase["ControllerServer"], IControllerFactory):
    def launch(
        self,
        *,
        worker_world: str,
        bootstrap_addr: str,
        controller_id: ActorId,
        num_worker_procs: int,
        gpus_per_host: int,
        supervision_params: SupervisionParams,
        controller_params: ControllerParams,
        labels: Dict[str, str],
    ) -> ServerInstance:
        return self._launch_instance(
            msg=ControllerServerRequest.Run(
                RunCommand.Controller(
                    ControllerCommand(
                        worker_world=worker_world,
                        system_addr=bootstrap_addr,
                        controller_actor_id=str(controller_id),
                        world_size=num_worker_procs,
                        num_procs_per_host=gpus_per_host,
                        worker_name="worker",
                        program=None,
                        supervision_query_interval_in_sec=supervision_params.query_interval_in_sec,
                        supervision_update_interval_in_sec=supervision_params.update_interval_in_sec,
                        worker_progress_check_interval_in_sec=controller_params.worker_progress_check_interval_in_sec,
                        operation_timeout_in_sec=controller_params.operation_timeout_in_sec,
                        operations_per_worker_progress_request=controller_params.operations_per_worker_progress_request,
                        fail_on_worker_timeout=controller_params.fail_on_worker_timeout,
                        is_cpu_worker=False,
                        extra_proc_labels=list(labels.items()),
                    ),
                ),
            ),
        )


class IWorkerFactory:
    def launch(
        self,
        *,
        worker_world: str,
        worker_rank: int,
        bootstrap_addr: str,
        labels: Dict[str, str],
    ) -> ServerInstance | subprocess.Popen[bytes]:
        raise NotImplementedError()


class WorkerFactory(IWorkerFactory):
    def __init__(
        self,
        *,
        num_worker_procs: int,
        gpus_per_host: int,
        logging_location: LoggingLocation,
        logging_dir: str,
    ) -> None:
        self.num_worker_procs = num_worker_procs
        self.gpus_per_host = gpus_per_host
        self.logging_location = logging_location
        self.logging_dir = logging_dir

    def launch(
        self,
        *,
        worker_world: str,
        worker_rank: int,
        bootstrap_addr: str,
        labels: Dict[str, str],
    ) -> subprocess.Popen[bytes]:
        stdout_location, stderr_location = _create_logging_locations(
            self.logging_dir,
            f"{worker_world}_{worker_rank}",
            self.logging_location,
        )
        return _start_worker_cmd(
            world_uuid=worker_world,
            worker_rank=worker_rank,
            gpus_per_host=self.gpus_per_host,
            num_worker_procs=self.num_worker_procs,
            args=[
                "worker",
                "--world-id",
                worker_world,
                "--proc-id",
                f"{worker_world}[{worker_rank}]",
                "--bootstrap-addr",
                bootstrap_addr,
            ]
            + _get_labels("extra-proc-labels", labels),
            stdout=stdout_location,
            stderr=stderr_location,
        )


class WorkerServer(ServerBase["WorkerServer"]):
    def __init__(
        self,
        *,
        uuid: str,
        num_worker_procs: int,
        gpus_per_host: int,
        world_rank: int,
        logging_location: LoggingLocation,
        logging_dir: str,
    ) -> None:
        super().__init__(
            name=uuid,
            response_cls=WorkerServerResponse,
            request_cls=WorkerServerRequest,
        )
        self.uuid = uuid
        self.num_worker_procs = num_worker_procs
        self.gpus_per_host = gpus_per_host
        self.world_rank = world_rank
        self.logging_location = logging_location
        self.logging_dir = logging_dir

    def _launch_server(
        self,
        read_fd: int,
        write_fd: int,
    ) -> subprocess.Popen[bytes]:
        stdout_location, stderr_location = _create_logging_locations(
            self.logging_dir,
            f"{self.uuid}_{self.world_rank}",
            self.logging_location,
        )
        return _start_worker_cmd(
            world_uuid=self.uuid,
            worker_rank=self.world_rank,
            gpus_per_host=self.gpus_per_host,
            num_worker_procs=self.num_worker_procs,
            args=["worker-server", str(read_fd), str(write_fd)],
            pass_fds=(read_fd, write_fd),
            stdin=subprocess.PIPE,
            stdout=stdout_location,
            stderr=stderr_location,
        )

    def launch(
        self,
        *,
        worker_world: str,
        bootstrap_addr: str,
        labels: Dict[str, str],
    ) -> ServerInstance:
        return self._launch_instance(
            msg=WorkerServerRequest.Run(
                world_id=worker_world,
                proc_id=f"{worker_world}[{self.world_rank}]",
                bootstrap_addr=bootstrap_addr,
                labels=list(labels.items()),
            )
        )


class WorkerServers(IWorkerFactory):
    def __init__(
        self,
        *,
        workers: dict[int, WorkerServer],
    ) -> None:
        self._workers = workers
        self._contexts: contextlib.ExitStack[None] = contextlib.ExitStack()

    @staticmethod
    def create(
        uuid: str,
        num_worker_procs: int,
        gpus_per_host: int,
        logging_location: LoggingLocation,
        logging_dir: str,
    ) -> "WorkerServers":
        return WorkerServers(
            workers={
                world_rank: WorkerServer(
                    uuid=uuid,
                    num_worker_procs=num_worker_procs,
                    gpus_per_host=gpus_per_host,
                    world_rank=world_rank,
                    logging_location=logging_location,
                    logging_dir=logging_dir,
                )
                for world_rank in range(num_worker_procs)
            },
        )

    def initiate_shutdown(self) -> None:
        for worker in self._workers.values():
            worker.initiate_shutdown()

    def __enter__(self) -> "WorkerServers":
        for worker in self._workers.values():
            self._contexts.enter_context(worker)
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.initiate_shutdown()
        self._contexts.__exit__(exc_type, exc_val, exc_tb)

    def launch(
        self,
        *,
        worker_world: str,
        worker_rank: int,
        bootstrap_addr: str,
        labels: Dict[str, str],
    ) -> ServerInstance | subprocess.Popen[bytes]:
        return self._workers[worker_rank].launch(
            worker_world=worker_world,
            bootstrap_addr=bootstrap_addr,
            labels=labels,
        )


class ProcessCache:
    def __init__(
        self,
        *,
        logging_location: LoggingLocation,
        logging_dir: str,
    ) -> None:
        self.logging_location: LoggingLocation = logging_location
        self.logging_dir: str = logging_dir

        self._system_cache: SystemServer | None = None
        self._controller_cache: ControllerServer | None = None
        self._worker_cache: dict[Tuple[int, int], WorkerServers] = {}
        self._contexts: contextlib.ExitStack[None] = contextlib.ExitStack()

    def __enter__(self) -> "ProcessCache":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._system_cache is not None:
            self._system_cache.initiate_shutdown()
        if self._controller_cache is not None:
            self._controller_cache.initiate_shutdown()
        for workers in self._worker_cache.values():
            workers.initiate_shutdown()
        self._contexts.__exit__(exc_type, exc_val, exc_tb)

    def get_system_server(self) -> SystemServer:
        if self._system_cache is None:
            system = SystemServer(
                uuid="cached_system",
                logging_location=self.logging_location,
                logging_dir=self.logging_dir,
            )
            self._system_cache = self._contexts.enter_context(system)
        assert self._system_cache is not None
        return self._system_cache

    def get_controller_server(self) -> ControllerServer:
        if self._controller_cache is None:
            controller = ControllerServer(
                uuid="cached_controller",
                logging_location=self.logging_location,
                logging_dir=self.logging_dir,
            )
            self._controller_cache = self._contexts.enter_context(controller)
        assert self._controller_cache is not None
        return self._controller_cache

    def get_worker_servers(
        self,
        *,
        num_worker_procs: int,
        gpus_per_host: int,
    ) -> WorkerServers:
        key = (num_worker_procs, gpus_per_host)
        workers = self._worker_cache.get(key)
        if workers is None:
            workers = WorkerServers.create(
                uuid=f"cached_workers_{num_worker_procs}_{gpus_per_host}",
                num_worker_procs=num_worker_procs,
                gpus_per_host=gpus_per_host,
                logging_location=self.logging_location,
                logging_dir=self.logging_dir,
            )
            self._worker_cache[key] = self._contexts.enter_context(workers)
        return workers


class Bootstrap:
    def __init__(
        self,
        *,
        meshes: int,
        hosts_per_mesh: int,
        gpus_per_host: int,
        worker_factory: IWorkerFactory | None = None,
        controller_factory: IControllerFactory | None = None,
        system_factory: ISystemFactory | None = None,
        socket_type: SocketType,
        logging_location: LoggingLocation,
        supervision_params: SupervisionParams | None,
        controller_params: ControllerParams | None,
        auto_epoch: bool,
        controller_labels: Dict[str, str] | None = None,
        worker_labels: Dict[str, str] | None = None,
    ) -> None:
        if supervision_params is None:
            supervision_params = SupervisionParams(
                update_timeout_in_sec=20,
                query_interval_in_sec=2,
                update_interval_in_sec=2,
            )
        self.supervision_params: SupervisionParams = supervision_params

        if controller_params is None:
            controller_params = ControllerParams(
                worker_progress_check_interval_in_sec=10,
                operation_timeout_in_sec=120,
                operations_per_worker_progress_request=100,
                fail_on_worker_timeout=False,
            )
        self.controller_params: ControllerParams = controller_params

        self.epoch: int | None = 0 if auto_epoch else None

        # hyperactor_telemetry will take the execution id and use it across all processes
        execution_id = "rust_local_" + uuid.uuid4().hex
        os.environ["HYPERACTOR_EXECUTION_ID"] = execution_id

        # Create a temporary directory for logging
        self.logging_dir: str = (
            tempfile.mkdtemp(prefix="rust_local_mesh_")
            if logging_location == LoggingLocation.FILE
            else "N/A"
        )
        logger.info(
            f"Creating Rust local mesh with {meshes} meshes X {hosts_per_mesh} hosts X {gpus_per_host} gpus.\n"
            f"Logging directory: \033[92;1m{self.logging_dir}\033[0m\n"
            f"Execution id: {execution_id}"
        )
        self.logging_location: LoggingLocation = logging_location

        if controller_factory is None:
            controller_factory = ControllerFactory(
                logging_location=self.logging_location,
                logging_dir=self.logging_dir,
            )
        self.controller_factory: IControllerFactory = controller_factory

        if system_factory is None:
            system_factory = SystemFactory(
                logging_location=self.logging_location,
                logging_dir=self.logging_dir,
            )
        self.system_factory: ISystemFactory = system_factory

        # do a fake call to instantiate ThreadPoolExecutor so we don't block GIL later
        if worker_factory is None:
            worker_factory = WorkerFactory(
                num_worker_procs=hosts_per_mesh * gpus_per_host,
                gpus_per_host=gpus_per_host,
                logging_location=self.logging_location,
                logging_dir=self.logging_dir,
            )
        self.worker_factory: IWorkerFactory = worker_factory

        # do a fake call to instantiate ThreadPoolExecutor so we don't block GIL later
        fake_call(lambda: 0)

        self.bootstrap_addr: str
        if socket_type == SocketType.TCP:
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            self.bootstrap_addr = f"tcp![::1]:{port}"
        elif socket_type == SocketType.UNIX:
            # provide a random unix socket address
            self.bootstrap_addr: str = f"unix!@{''.join(random.choice(string.ascii_lowercase) for _ in range(14))}-system"
        else:
            raise ValueError(f"Unknown socket type: {socket_type}")

        env = os.environ.copy()
        self.env: dict[str, str] = env

        # Launch a single system globally
        self.processes: list[subprocess.Popen[bytes] | ServerInstance] = []
        self.processes.append(self._launch_system())

        self.has_shutdown: bool = False
        self.gpus_per_host: int = gpus_per_host
        self.num_worker_procs: int = hosts_per_mesh * gpus_per_host
        self.controller_ids: list[ActorId] = []
        self.mesh_worlds: dict[
            MeshWorld, list[subprocess.Popen[bytes] | ServerInstance]
        ] = {}

        # Create meshes, each of which contains a single controller and multiple workers.
        # All of them will connect to the same system.
        pids: dict[str, list[int]] = {}
        for i in range(meshes):
            mesh_name: str = f"mesh_{i}"
            controller_world: str = f"{mesh_name}_controller"
            worker_world: str = f"{mesh_name}_worker"
            controller_id: ActorId = ActorId(
                world_name=controller_world,
                rank=0,
                actor_name="controller",
            )
            self.mesh_worlds[(worker_world, controller_id)] = []
            self.controller_ids.append(controller_id)

            processes: list[subprocess.Popen[bytes] | ServerInstance] = (
                self.launch_mesh(
                    controller_id,
                    worker_world,
                    controller_labels=controller_labels,
                    worker_labels=worker_labels,
                )
            )

            self.processes.extend(processes)
            pids[mesh_name] = [p.pid for p in processes]

        log_message = (
            f"All processes started successfully:\n system: {self.processes[0].pid}\n"
        )
        for mesh, procs in pids.items():
            log_message += f"{mesh}: controller: {procs[0]}, "
            worker_messages = []
            for i in range(1, len(procs)):
                worker_messages.append(f"{i-1}: {procs[i]}")
            log_message += "workers: " + ", ".join(worker_messages)
            log_message += "\n"

        self._contexts: contextlib.ExitStack[None] = contextlib.ExitStack()

        logger.info(log_message)

    def _launch_system(
        self,
    ) -> ServerInstance | subprocess.Popen[bytes]:
        logger.info("launching system")
        try:
            return self.system_factory.launch(
                bootstrap_addr=self.bootstrap_addr,
                supervision_params=self.supervision_params,
            )
        except Exception as e:
            logger.error(f"Failed to start system process: {e}")
            raise e

    def _launch_controller(
        self,
        controller_id: ActorId,
        worker_world: str,
        epoch: str | None = None,
        labels: Dict[str, str] | None = None,
    ) -> subprocess.Popen[bytes] | ServerInstance:
        logger.info("launching controller")
        try:
            return self.controller_factory.launch(
                bootstrap_addr=self.bootstrap_addr,
                worker_world=worker_world
                if epoch is None
                else f"{worker_world}_{epoch}",
                controller_id=ActorId.from_string(
                    (
                        f"{controller_id.world_name + '_' + epoch if epoch else controller_id.world_name}"
                        f"[{controller_id.rank}]."
                        f"{controller_id.actor_name}[{controller_id.pid}]"
                    )
                ),
                num_worker_procs=self.num_worker_procs,
                gpus_per_host=self.gpus_per_host,
                supervision_params=self.supervision_params,
                controller_params=self.controller_params,
                labels={} if labels is None else labels,
            )
        except Exception as e:
            logger.error(f"Failed to start controller process: {e}")
            raise e

    def _launch_worker(
        self,
        worker_world: str,
        worker_rank: int,
        epoch: str | None = None,
        labels: Dict[str, str] | None = None,
    ) -> subprocess.Popen[bytes] | ServerInstance:
        logger.info("launching worker")
        try:
            return self.worker_factory.launch(
                worker_world=worker_world
                if epoch is None
                else f"{worker_world}_{epoch}",
                worker_rank=worker_rank,
                bootstrap_addr=self.bootstrap_addr,
                labels={} if labels is None else labels,
            )
        except Exception as e:
            logger.error(f"Failed to start worker process {worker_rank}: {e}")
            raise e

    def get_mesh_worlds(self) -> list[MeshWorld]:
        return list(self.mesh_worlds.keys())

    def kill_mesh(self, mesh_world: MeshWorld) -> None:
        logger.info(f"Killing mesh {mesh_world}")
        procs = self.mesh_worlds[mesh_world]
        procs[-1].kill()

    def spawn_mesh(self, mesh_world: MeshWorld) -> None:
        self.launch_mesh(mesh_world[1], mesh_world[0])

    def launch_mesh(
        self,
        controller_id: ActorId,
        worker_world: str,
        controller_labels: Dict[str, str] | None = None,
        worker_labels: Dict[str, str] | None = None,
    ) -> list[subprocess.Popen[bytes] | ServerInstance]:
        """
        Create a single controller and multiple workers for a mesh.
        The first process of the return is the controller.
        The remaining ones are workers.
        """
        logger.info(
            f"Launching mesh {worker_world} with controller {controller_id} epoch {self.epoch}"
        )
        epoch: str | None = None
        if self.epoch is not None:
            epoch = f"epoch_{self.epoch}"
            self.epoch += 1

        processes: list[subprocess.Popen[bytes] | ServerInstance] = []
        controller_process = self._launch_controller(
            controller_id,
            worker_world,
            epoch,
            controller_labels,
        )
        processes.append(controller_process)

        for i in range(self.num_worker_procs):
            worker_process = self._launch_worker(worker_world, i, epoch, worker_labels)
            processes.append(worker_process)
        self.mesh_worlds[(worker_world, controller_id)] = processes
        return processes

    def __enter__(self) -> "Bootstrap":
        for process in self.processes:
            self._contexts.enter_context(process)
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for process in self.processes:
            process.terminate()
        self._contexts.__exit__(exc_type, exc_val, exc_tb)


def _local_device_count() -> int:
    dev_path = Path("/dev")
    pattern = re.compile(r"nvidia\d+$")
    nvidia_devices = [dev for dev in dev_path.iterdir() if pattern.match(dev.name)]
    return len(nvidia_devices)


def _get_worker_exec_info() -> tuple[list[str], dict[str, str]]:
    if IN_PAR:
        cmd = [sys.argv[0]]
        env = {
            "PAR_MAIN_OVERRIDE": _MONARCH_TENSOR_WORKER_MAIN,
        }
    else:
        cmd = [sys.executable, "-m", _MONARCH_TENSOR_WORKER_MAIN]
        env = {}

    env["MONARCH_TENSOR_WORKER_MAIN"] = _MONARCH_TENSOR_WORKER_MAIN
    env["MONARCH_TENSOR_WORKER_EXE"] = cmd[0]
    return cmd, env


@contextlib.contextmanager
def local_mesh(
    *,
    hosts: int = 1,
    gpus_per_host: int | None = None,
    socket_type: SocketType = SocketType.TCP,
    logging_location: LoggingLocation = LoggingLocation.FILE,
    supervision_params: SupervisionParams | None = None,
    controller_params: ControllerParams | None = None,
    worker_factory: IWorkerFactory | None = None,
    controller_factory: IControllerFactory | None = None,
    system_factory: ISystemFactory | None = None,
) -> Generator[DeviceMesh, None, None]:
    """
    Creates a single local device mesh with the given number of per host.

    Args:
        hosts               : number of hosts, primarily used for simulating multiple machines locally.
                              Default: 1
        gpus_per_host       : number of gpus per host.
                              Default: the number of GPUs this machine has.
        socket_type         : socket type to use for communication between processes.
                              Default: TCP.

    Example::
        with local_mesh().activate():
            x = torch.rand(3, 4)
            local_tensor = fetch_shard(x).result()
    """
    with local_meshes(
        meshes=1,
        hosts_per_mesh=hosts,
        gpus_per_host=gpus_per_host,
        socket_type=socket_type,
        logging_location=logging_location,
        supervision_params=supervision_params,
        controller_params=controller_params,
        worker_factory=worker_factory,
        controller_factory=controller_factory,
        system_factory=system_factory,
    ) as dms:
        assert len(dms) == 1
        yield dms[0]


@contextlib.contextmanager
def local_meshes(
    *,
    meshes: int = 1,
    hosts_per_mesh: int = 1,
    gpus_per_host: int | None = None,
    socket_type: SocketType = SocketType.TCP,
    logging_location: LoggingLocation = LoggingLocation.FILE,
    supervision_params: SupervisionParams | None = None,
    controller_params: ControllerParams | None = None,
    worker_factory: IWorkerFactory | None = None,
    controller_factory: IControllerFactory | None = None,
    system_factory: ISystemFactory | None = None,
) -> Generator[list[DeviceMesh], None, None]:
    """
    Creates multiple local device meshes.

    Args:
        meshes              : number of global meshes to create.
                              Default: 1
        hosts_per_mesh      : number of hosts per mesh, primarily used for simulating multiple machines locally.
                              Default: 1
        gpus_per_host       : number of gpus per host.
                              Default: the number of GPUs this machine has.
        socket_type         : socket type to use for communication between processes.
                              Default: TCP.
    """
    (dms, bootstrap) = local_meshes_and_bootstraps(
        meshes=meshes,
        hosts_per_mesh=hosts_per_mesh,
        gpus_per_host=gpus_per_host,
        socket_type=socket_type,
        logging_location=logging_location,
        supervision_params=supervision_params,
        controller_params=controller_params,
        worker_factory=worker_factory,
        controller_factory=controller_factory,
        system_factory=system_factory,
    )
    with bootstrap:
        maybe_error = None
        try:
            yield dms
        except Exception as e:
            maybe_error = e
            raise
        finally:
            for dm in dms:
                dm.exit(maybe_error)


def local_meshes_and_bootstraps(
    *,
    meshes: int = 1,
    hosts_per_mesh: int = 1,
    gpus_per_host: int | None = None,
    socket_type: SocketType = SocketType.TCP,
    logging_location: LoggingLocation = LoggingLocation.FILE,
    supervision_params: SupervisionParams | None = None,
    controller_params: ControllerParams | None = None,
    auto_epoch: bool = False,
    worker_factory: IWorkerFactory | None = None,
    controller_factory: IControllerFactory | None = None,
    system_factory: ISystemFactory | None = None,
) -> tuple[list[DeviceMesh], Bootstrap]:
    """
    Same as local_meshes, but also returns the bootstrap object. This is
    useful in tests where we want to maniputate the bootstrap object.
    """

    if gpus_per_host is None:
        gpus_per_host = _local_device_count()
    assert (
        0 < gpus_per_host <= 8
    ), "Number of GPUs must be greater than 0 and at most 8."
    bootstrap: Bootstrap = Bootstrap(
        meshes=meshes,
        hosts_per_mesh=hosts_per_mesh,
        gpus_per_host=gpus_per_host,
        socket_type=socket_type,
        logging_location=logging_location,
        supervision_params=supervision_params,
        controller_params=controller_params,
        auto_epoch=auto_epoch,
        worker_factory=worker_factory,
        controller_factory=controller_factory,
        system_factory=system_factory,
    )

    def create_exit(
        dm: DeviceMesh, bootstrap: Bootstrap
    ) -> Callable[[Optional[RemoteException | DeviceException | Exception]], None]:
        def exit(
            error: Optional[RemoteException | DeviceException | Exception] = None,
        ) -> None:
            # We only support one single client proc.
            if not bootstrap.has_shutdown:
                dm.client.shutdown(True, error)
                bootstrap.has_shutdown = True

        # We do not need to shutdown bootstrap and clean up the processes
        # as they will be cleaned up with the parent.
        return exit

    dms = rust_backend_meshes(
        system_addr=bootstrap.bootstrap_addr,
        hosts=hosts_per_mesh,
        gpus=gpus_per_host,
        requested_meshes=meshes,
    )

    for dm in dms:
        dm.exit = create_exit(dm, bootstrap)

    return (dms, bootstrap)


def local_mesh_provider(
    *,
    meshes: int = 1,
    hosts_per_mesh: int = 1,
    gpus_per_host: int | None = None,
    socket_type: SocketType = SocketType.TCP,
    logging_location: LoggingLocation = LoggingLocation.FILE,
    supervision_params: SupervisionParams | None = None,
    controller_params: ControllerParams | None = None,
    auto_epoch: bool = False,
    controller_labels: Dict[str, str] | None = None,
    worker_labels: Dict[str, str] | None = None,
    worker_factory: IWorkerFactory | None = None,
    controller_factory: IControllerFactory | None = None,
    system_factory: ISystemFactory | None = None,
    # pyre-fixme[11]: Annotation `DeviceMeshProvider` is not defined as a type.
) -> tuple[PoolDeviceMeshProvider, Bootstrap]:
    if gpus_per_host is None:
        gpus_per_host = _local_device_count()
    assert (
        0 < gpus_per_host <= 8
    ), "Number of GPUs must be greater than 0 and at most 8."
    bootstrap: Bootstrap = Bootstrap(
        meshes=meshes,
        hosts_per_mesh=hosts_per_mesh,
        gpus_per_host=gpus_per_host,
        socket_type=socket_type,
        logging_location=logging_location,
        supervision_params=supervision_params,
        controller_params=controller_params,
        auto_epoch=auto_epoch,
        controller_labels=controller_labels,
        worker_labels=worker_labels,
        worker_factory=worker_factory,
        controller_factory=controller_factory,
        system_factory=system_factory,
    )

    provider = rust_backend_mesh_provider(
        system_addr=bootstrap.bootstrap_addr,
        hosts=hosts_per_mesh,
        gpus=gpus_per_host,
    )
    return (provider, bootstrap)
