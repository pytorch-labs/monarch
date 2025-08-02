# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import os
import sys
import threading
import warnings
from contextlib import AbstractContextManager

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TYPE_CHECKING,
    TypeVar,
)

from monarch._rust_bindings.monarch_extension.logging import LoggingMeshClient
from monarch._rust_bindings.monarch_hyperactor.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    Alloc,
    AllocConstraints,
    AllocSpec,
)

from monarch._rust_bindings.monarch_hyperactor.proc_mesh import (
    ProcMesh as HyProcMesh,
    ProcMeshMonitor,
)
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask, Shared
from monarch._rust_bindings.monarch_hyperactor.shape import Shape, Slice
from monarch._src.actor.actor_mesh import _Actor, _ActorMeshRefImpl, Actor, ActorMeshRef

from monarch._src.actor.allocator import (
    AllocateMixin,
    AllocHandle,
    LocalAllocator,
    ProcessAllocator,
    SimAllocator,
)
from monarch._src.actor.code_sync import (
    CodeSyncMeshClient,
    RemoteWorkspace,
    WorkspaceLocation,
    WorkspaceShape,
)
from monarch._src.actor.debugger import (
    _DEBUG_MANAGER_ACTOR_NAME,
    DebugClient,
    DebugManager,
)

from monarch._src.actor.device_utils import _local_device_count

from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.future import DeprecatedNotAFuture, Future
from monarch._src.actor.shape import MeshTrait

HAS_TENSOR_ENGINE = False
try:
    # Torch is needed for tensor engine
    import torch  # @manual

    # Confirm that rust bindings were built with tensor engine enabled
    from monarch._rust_bindings.rdma import (  # type: ignore[import]
        _RdmaBuffer,
        _RdmaManager,
    )

    # type: ignore[16]
    HAS_TENSOR_ENGINE = torch.cuda.is_available()
except ImportError:
    logging.warning("Tensor engine is not available on this platform")


if TYPE_CHECKING:
    Tensor = Any
    DeviceMesh = Any


class SetupActor(Actor):
    """
    A helper actor to setup the proc mesh with user defined setup method.
    Typically used to setup the environment variables.
    """

    def __init__(self, env: Callable[[], None]) -> None:
        """
        Initialize the setup actor with the user defined setup method.
        """
        self._setup_method = env

    @endpoint
    async def setup(self) -> None:
        """
        Call the user defined setup method with the monarch context.
        """
        self._setup_method()


T = TypeVar("T")
try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = bool(fbmake.get("par_style"))
except ImportError:
    IN_PAR = False


class ProcMesh(MeshTrait, DeprecatedNotAFuture):
    def __init__(
        self,
        hy_proc_mesh: "Shared[HyProcMesh]",
        shape: Shape,
        _device_mesh: Optional["DeviceMesh"] = None,
    ) -> None:
        self._proc_mesh = hy_proc_mesh
        self._shape = shape
        # until we have real slicing support keep track
        # of whether this is a slice of a real proc_meshg
        self._slice = False
        # type: ignore[21]
        self._rdma_manager: Optional["_RdmaManager"] = None
        self._debug_manager: Optional[DebugManager] = None
        self._code_sync_client: Optional[CodeSyncMeshClient] = None
        self._logging_mesh_client: Optional[LoggingMeshClient] = None
        self._maybe_device_mesh: Optional["DeviceMesh"] = _device_mesh
        self._stopped = False

    @property
    def initialized(self) -> Future[Literal[True]]:
        """
        Future completes with 'True' when the ProcMesh has initialized.
        Because ProcMesh are remote objects, there is no guarentee that the ProcMesh is
        still usable after this completes, only that at some point in the past it was usable.
        """
        pm: Shared[HyProcMesh] = self._proc_mesh

        async def task() -> Literal[True]:
            await pm
            return True

        return Future(coro=task())

    def _init_manager_actors(self, setup: Callable[[], None] | None = None) -> None:
        self._proc_mesh = PythonTask.from_coroutine(
            self._init_manager_actors_coro(self._proc_mesh, setup)
        ).spawn()

    async def _init_manager_actors_coro(
        self,
        proc_mesh_: "Shared[HyProcMesh]",
        setup: Callable[[], None] | None = None,
    ) -> "HyProcMesh":
        proc_mesh = await proc_mesh_
        # WARNING: it is unsafe to await self._proc_mesh here
        # because self._proc_mesh is the result of this function itself!

        self._logging_mesh_client = await LoggingMeshClient.spawn(proc_mesh=proc_mesh)
        self._logging_mesh_client.set_mode(
            stream_to_client=True,
            aggregate_window_sec=3,
            level=logging.INFO,
        )

        _rdma_manager = (
            # type: ignore[16]
            await _RdmaManager.create_rdma_manager_nonblocking(proc_mesh)
            # type: ignore[16]
            if HAS_TENSOR_ENGINE and _RdmaBuffer.rdma_supported()
            else None
        )

        _debug_manager = await self._spawn_nonblocking_on(
            proc_mesh, _DEBUG_MANAGER_ACTOR_NAME, DebugManager, await _debug_client()
        )

        self._debug_manager = _debug_manager
        self._rdma_manager = _rdma_manager

        if setup is not None:
            # If the user has passed the setup lambda, we need to call
            # it here before any of the other actors are spawned so that
            # the environment variables are set up before cuda init.
            setup_actor = await self._spawn_nonblocking_on(
                proc_mesh, "setup", SetupActor, setup
            )
            # pyre-ignore
            await setup_actor.setup.call()._status.coro

        return proc_mesh

    @property
    def _ndslice(self) -> Slice:
        return self._shape.ndslice

    @property
    def _labels(self) -> List[str]:
        return self._shape.labels

    def _new_with_shape(self, shape: Shape) -> "ProcMesh":
        device_mesh = (
            None
            if self._maybe_device_mesh is None
            else self._device_mesh._new_with_shape(shape)
        )
        pm = ProcMesh(self._proc_mesh, shape, _device_mesh=device_mesh)
        pm._slice = True
        return pm

    def spawn(self, name: str, Class: Type[T], *args: Any, **kwargs: Any) -> Future[T]:
        if self._slice:
            raise NotImplementedError("NYI: spawn on slice of a proc mesh.")
        return Future(coro=self._spawn_nonblocking(name, Class, *args, **kwargs))

    @property
    async def _proc_mesh_for_asyncio_fixme(self) -> HyProcMesh:
        """
        Get ProcMesh on the asyncio event stream.
        We should redo this functionality to work on the tokio stream.
        This must be called on the asyncio stream.
        """
        assert asyncio.get_running_loop() is not None
        return await Future(coro=self._proc_mesh.task())

    async def monitor(self) -> ProcMeshMonitor:
        """
        Get a monitor (async iterator) of the proc mesh, it is used to
        monitor the status of the proc mesh. This function can be called at most once.

        Note: This API is experimental and subject to change.

        Example:

        async def monitor_loop(monitor):
            async for event in monitor:
                await handle_exception_event(event)

        # Kick off in background
        asyncio.create_task(monitor_loop(monitor))
        """
        # todo: move monitor to tokio loop
        proc_mesh = await Future(coro=self._proc_mesh.task())
        return await proc_mesh.monitor()

    @classmethod
    def from_alloc(
        self,
        alloc: AllocHandle,
        setup: Callable[[], None] | None = None,
        _init_manager_actors: bool = True,
    ) -> "ProcMesh":
        """
        Allocate a process mesh according to the provided alloc.
        Returns when the mesh is fully allocated.

        Arguments:
        - `alloc`: The alloc to allocate according to.
        - `setup`: An optional lambda function to configure environment variables on the allocated mesh.
        Use the `current_rank()` method within the lambda to obtain the rank.

        Example of a setup method to initialize torch distributed environment variables:
        ```
        def setup():
            rank = current_rank()
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(len(rank.shape))
            os.environ["LOCAL_RANK"] = str(rank["gpus"])
        ```
        """

        async def task() -> HyProcMesh:
            return await HyProcMesh.allocate_nonblocking(await alloc._hy_alloc)

        shape = Shape(
            list(alloc._extent.keys()),
            Slice.new_row_major(list(alloc._extent.values())),
        )
        pm = ProcMesh(PythonTask.from_coroutine(task()).spawn(), shape)

        if _init_manager_actors:
            pm._init_manager_actors(setup)
        return pm

    def __repr__(self) -> str:
        return repr(self._proc_mesh)

    def __str__(self) -> str:
        return str(self._proc_mesh)

    async def _spawn_nonblocking(
        self, name: str, Class: Type[T], *args: Any, **kwargs: Any
    ) -> T:
        return await self._spawn_nonblocking_on(
            await self._proc_mesh, name, Class, *args, **kwargs
        )

    async def _spawn_nonblocking_on(
        self, pm: HyProcMesh, name: str, Class: Type[T], *args: Any, **kwargs: Any
    ) -> T:
        if not issubclass(Class, Actor):
            raise ValueError(
                f"{Class} must subclass monarch.service.Actor to spawn it."
            )
        actor_mesh = await pm.spawn_nonblocking(name, _Actor)
        service = ActorMeshRef(
            Class,
            _ActorMeshRefImpl.from_hyperactor_mesh(pm.client, actor_mesh, self),
            pm.client,
        )
        # useful to have this separate, because eventually we can reconstitute ActorMeshRef objects across pickling by
        # doing `ActorMeshRef(Class, actor_handle)` but not calling _create.
        service._create(args, kwargs)
        return cast(T, service)

    @property
    def _device_mesh(self) -> "DeviceMesh":
        if not HAS_TENSOR_ENGINE:
            raise RuntimeError(
                "DeviceMesh is not available because tensor_engine was not compiled (USE_TENSOR_ENGINE=0)"
            )

        # type: ignore[21]
        from monarch.mesh_controller import spawn_tensor_engine  # @manual

        if self._maybe_device_mesh is None:
            if self._slice:
                raise NotImplementedError(
                    "NYI: activating a proc mesh must first happen on the root proc_mesh until we fix spawning on submeshes."
                )
            # type: ignore[21]
            self._maybe_device_mesh = spawn_tensor_engine(self)
        return self._maybe_device_mesh

    # pyre-ignore
    def activate(self) -> AbstractContextManager:
        return self._device_mesh.activate()

    def rank_tensor(self, dim: str | Sequence[str]) -> "Tensor":
        return self._device_mesh.rank(dim)

    def rank_tensors(self) -> Dict[str, "Tensor"]:
        return self._device_mesh.ranks

    async def sync_workspace(self, auto_reload: bool = False) -> None:
        if self._code_sync_client is None:
            self._code_sync_client = CodeSyncMeshClient.spawn_blocking(
                proc_mesh=await self._proc_mesh_for_asyncio_fixme,
            )
        # TODO(agallagher): We need some way to configure and pass this
        # in -- right now we're assuming the `gpu` dimension, which isn't
        # correct.
        # The workspace shape (i.e. only perform one rsync per host).
        assert set(self._shape.labels).issubset({"gpus", "hosts"})
        assert self._code_sync_client is not None
        await self._code_sync_client.sync_workspace(
            # TODO(agallagher): Is there a better way to infer/set the local
            # workspace dir, rather than use PWD?
            local=os.getcwd(),
            remote=RemoteWorkspace(
                location=WorkspaceLocation.FromEnvVar("WORKSPACE_DIR"),
                shape=WorkspaceShape.shared("gpus"),
            ),
            auto_reload=auto_reload,
        )

    async def logging_option(
        self,
        stream_to_client: bool = True,
        aggregate_window_sec: int | None = 3,
        level: int = logging.INFO,
    ) -> None:
        """
        Set the logging options for the remote processes

        Args:
            stream_to_client (bool): If True, logs from the remote processes will be streamed to the client.
            Defaults to True.
            aggregate_window_sec (Optional[int]): If not None, logs from the remote processes will be aggregated
            and sent to the client every aggregate_window_sec seconds. Defaults to 3 seconds, meaning no aggregation.
            Error will be thrown if aggregate_window_sec is set and stream_to_client is False.
            level (int): The logging level of the logger. Defaults to logging.INFO.

        Returns:
            None
        """
        if level < 0 or level > 255:
            raise ValueError("Invalid logging level: {}".format(level))
        await self.initialized

        assert self._logging_mesh_client is not None
        self._logging_mesh_client.set_mode(
            stream_to_client=stream_to_client,
            aggregate_window_sec=aggregate_window_sec,
            level=level,
        )

    async def __aenter__(self) -> "ProcMesh":
        if self._stopped:
            raise RuntimeError("`ProcMesh` has already been stopped")
        return self

    def stop(self) -> Future[None]:
        async def _stop_nonblocking() -> None:
            await (await self._proc_mesh).stop_nonblocking()
            self._stopped = True

        return Future(coro=_stop_nonblocking())

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        # In case there are multiple nested "async with" statements, we only
        # want it to close once.
        if not self._stopped:
            await self.stop()

    # Finalizer to check if the proc mesh was closed properly.
    def __del__(self) -> None:
        if not self._stopped:
            warnings.warn(
                f"unstopped ProcMesh {self!r}",
                ResourceWarning,
                stacklevel=2,
                source=self,
            )
            # Cannot call stop here because it is async.


def local_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> ProcMesh:
    return _proc_mesh_from_allocator(allocator=LocalAllocator(), gpus=gpus, hosts=hosts)


def sim_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> ProcMesh:
    return _proc_mesh_from_allocator(allocator=SimAllocator(), gpus=gpus, hosts=hosts)


_BOOTSTRAP_MAIN = "monarch._src.actor.bootstrap_main"


def _get_bootstrap_args() -> tuple[str, Optional[list[str]], dict[str, str]]:
    if IN_PAR:
        cmd = sys.argv[0]
        args = None
        env = {
            "PAR_MAIN_OVERRIDE": _BOOTSTRAP_MAIN,
        }
    else:
        cmd = sys.executable
        args = ["-m", _BOOTSTRAP_MAIN]
        env = {}

    return cmd, args, env


async def _hy_proc_mesh_from_alloc_coro(
    alloc: "Shared[Alloc] | PythonTask[Alloc]",
) -> HyProcMesh:
    return await HyProcMesh.allocate_nonblocking(await alloc)


def _proc_mesh_from_allocator(
    *,
    allocator: AllocateMixin,
    gpus: Optional[int],
    hosts: int,
    setup: Callable[[], None] | None = None,
    _init_manager_actors: bool = True,
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    # gpus must come last in this order because
    # test_remote_function_all_gather expects that hosts comes before gpus
    # in the order of the dimensions.
    spec: AllocSpec = AllocSpec(AllocConstraints(), hosts=hosts, gpus=gpus)
    alloc = allocator.allocate(spec)
    return ProcMesh.from_alloc(alloc, setup, _init_manager_actors)


def proc_mesh(
    *,
    gpus: Optional[int] = None,
    hosts: int = 1,
    env: dict[str, str] | None = None,
    setup: Callable[[], None] | None = None,
) -> ProcMesh:
    env = env or {}
    # Todo: Deprecate the env field from the ProcessAllocator
    # The PAR_MAIN_OVERRIDE needs to be passed as an env
    # to the proc mesh construction in rust, so can not be moved to the
    # SetupActor yet
    cmd, args, bootstrap_env = _get_bootstrap_args()
    env.update(bootstrap_env)
    return _proc_mesh_from_allocator(
        allocator=ProcessAllocator(cmd, args, env),
        hosts=hosts,
        gpus=gpus,
        setup=setup,
        _init_manager_actors=True,
    )


_debug_client_init = threading.Lock()
_debug_proc_mesh: Optional["ProcMesh"] = None
_debug_client_mesh: "Optional[Shared[DebugClient]]" = None


# Lazy init so that the debug client and proc does not produce logs when it isn't used.
# Checking for the client needs a lock otherwise two initializing procs will both
# try to init resulting in duplicates. The critical region is not blocking: it spawns
# a separate task to do the init, asigns the Shared[Client] from that task to the global
# and releases the lock.
def _debug_client() -> "Shared[DebugClient]":
    global _debug_client_mesh, _debug_proc_mesh

    async def create() -> DebugClient:
        _debug_proc_mesh = _proc_mesh_from_allocator(
            gpus=1, hosts=1, allocator=LocalAllocator(), _init_manager_actors=False
        )
        return await _debug_proc_mesh._spawn_nonblocking("debug_client", DebugClient)

    with _debug_client_init:
        if _debug_client_mesh is None:
            _debug_client_mesh = PythonTask.from_coroutine(create()).spawn()

    return _debug_client_mesh


def debug_client() -> DebugClient:
    return Future(coro=_debug_client().task()).get()
