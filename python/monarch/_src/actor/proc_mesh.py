# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import sys
import warnings
from contextlib import AbstractContextManager

from typing import (
    Any,
    cast,
    Dict,
    List,
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
from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import (
    ProcMesh as HyProcMesh,
    ProcMeshMonitor,
)
from monarch._rust_bindings.monarch_hyperactor.shape import Shape, Slice
from monarch._src.actor.actor_mesh import _Actor, _ActorMeshRefImpl, Actor, ActorMeshRef
from monarch._src.actor.allocator import LocalAllocator, ProcessAllocator, SimAllocator
from monarch._src.actor.code_sync import RsyncMeshClient, WorkspaceLocation
from monarch._src.actor.code_sync.auto_reload import AutoReloadActor
from monarch._src.actor.debugger import (
    _DEBUG_MANAGER_ACTOR_NAME,
    DebugClient,
    DebugManager,
)

from monarch._src.actor.device_utils import _local_device_count
from monarch._src.actor.future import Future
from monarch._src.actor.shape import MeshTrait

HAS_TENSOR_ENGINE = False
try:
    # TODO: while the tensor_engine submodule doesn't exist yet, use the
    # available of monarch.rdma as a proxy.
    # type: ignore
    from monarch.rdma import RDMAManager  # @manual

    HAS_TENSOR_ENGINE = True
except ImportError:
    pass


if TYPE_CHECKING:
    Tensor = Any
    DeviceMesh = Any


T = TypeVar("T")
try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
except ImportError:
    IN_PAR = False


async def _allocate_nonblocking(alloc: Alloc) -> "ProcMesh":
    return ProcMesh(await HyProcMesh.allocate_nonblocking(alloc))


def _allocate_blocking(alloc: Alloc) -> "ProcMesh":
    return ProcMesh(HyProcMesh.allocate_blocking(alloc))


class ProcMesh(MeshTrait):
    def __init__(
        self,
        hy_proc_mesh: HyProcMesh,
        _mock_shape: Optional[Shape] = None,
        _device_mesh: Optional["DeviceMesh"] = None,
        _is_initializing_debugger: bool = False,
    ) -> None:
        self._proc_mesh = hy_proc_mesh
        self._mock_shape: Optional[Shape] = _mock_shape
        # type: ignore[21]
        self._rdma_manager: Optional["RDMAManager"] = None
        self._debug_manager: Optional[DebugManager] = None
        self._mailbox: Mailbox = self._proc_mesh.client
        self._rsync_mesh_client: Optional[RsyncMeshClient] = None
        self._auto_reload_actor: Optional[AutoReloadActor] = None
        self._logging_mesh_client: Optional[LoggingMeshClient] = None
        self._maybe_device_mesh: Optional["DeviceMesh"] = _device_mesh
        self._stopped = False
        if _mock_shape is None and HAS_TENSOR_ENGINE:
            # type: ignore[21]
            self._rdma_manager = self._spawn_blocking("rdma_manager", RDMAManager)
        if not _is_initializing_debugger and _mock_shape is None:
            self._debug_manager = self._spawn_blocking(
                _DEBUG_MANAGER_ACTOR_NAME, DebugManager, debug_client()
            )

    @property
    def _shape(self) -> Shape:
        return self._proc_mesh.shape if self._mock_shape is None else self._mock_shape

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
        return ProcMesh(self._proc_mesh, _mock_shape=shape, _device_mesh=device_mesh)

    def spawn(self, name: str, Class: Type[T], *args: Any, **kwargs: Any) -> Future[T]:
        if self._mock_shape is not None:
            raise NotImplementedError("NYI: spawn on slice of a proc mesh.")
        return Future(
            lambda: self._spawn_nonblocking(name, Class, *args, **kwargs),
            lambda: self._spawn_blocking(name, Class, *args, **kwargs),
        )

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
        return await self._proc_mesh.monitor()

    @classmethod
    def from_alloc(self, alloc: Alloc) -> Future["ProcMesh"]:
        return Future(
            lambda: _allocate_nonblocking(alloc),
            lambda: _allocate_blocking(alloc),
        )

    def _spawn_blocking(
        self, name: str, Class: Type[T], *args: Any, **kwargs: Any
    ) -> T:
        if not issubclass(Class, Actor):
            raise ValueError(
                f"{Class} must subclass monarch.service.Actor to spawn it."
            )

        actor_mesh = self._proc_mesh.spawn_blocking(name, _Actor)
        service = ActorMeshRef(
            Class,
            _ActorMeshRefImpl.from_hyperactor_mesh(self._mailbox, actor_mesh, self),
            self._mailbox,
        )
        # useful to have this separate, because eventually we can reconstitute ActorMeshRef objects across pickling by
        # doing `ActorMeshRef(Class, actor_handle)` but not calling _create.
        service._create(args, kwargs)
        return cast(T, service)

    def __repr__(self) -> str:
        return repr(self._proc_mesh)

    def __str__(self) -> str:
        return str(self._proc_mesh)

    async def _spawn_nonblocking(
        self, name: str, Class: Type[T], *args: Any, **kwargs: Any
    ) -> T:
        if not issubclass(Class, Actor):
            raise ValueError(
                f"{Class} must subclass monarch.service.Actor to spawn it."
            )

        actor_mesh = await self._proc_mesh.spawn_nonblocking(name, _Actor)
        service = ActorMeshRef(
            Class,
            _ActorMeshRefImpl.from_hyperactor_mesh(self._mailbox, actor_mesh, self),
            self._mailbox,
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
            if self._mock_shape is not None:
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
        if self._rsync_mesh_client is None:
            # TODO(agallagher): We need some way to configure and pass this
            # in -- right now we're assuming the `gpu` dimension, which isn't
            # correct.
            assert set(self._proc_mesh.shape.labels).issubset({"gpus", "hosts"})
            # The workspace shape (i.e. only perform one rsync per host).
            workspace_shape = self.slice(gpus=slice(0, 1, 1))._mock_shape
            assert workspace_shape is not None
            # TODO(agallagher): We should probably hide this behind something
            # like a `Workspace` class and support abstracting/configuring
            # different sync methods.
            self._rsync_mesh_client = RsyncMeshClient.spawn_blocking(
                proc_mesh=self._proc_mesh,
                shape=workspace_shape,
                # TODO(agallagher): Is there a better way to infer/set the local
                # workspace dir, rather than use PWD?
                local_workspace=os.getcwd(),
                remote_workspace=WorkspaceLocation.FromEnvVar("WORKSPACE_DIR"),
            )
            self._auto_reload_actor = self._spawn_blocking(
                "auto_reload",
                AutoReloadActor,
                WorkspaceLocation.FromEnvVar("WORKSPACE_DIR"),
            )
        assert self._rsync_mesh_client is not None
        await self._rsync_mesh_client.sync_workspace()
        if auto_reload:
            assert self._auto_reload_actor is not None
            await self._auto_reload_actor.reload.call()

    def logging_option(self, stream_to_client: bool = False) -> None:
        """
        Set the logging options for the remote processes

        Args:
            stream_to_client (bool): If True, logs from the remote processes will be streamed to the client.
            Defaults to False.

        Returns:
            None
        """
        if self._logging_mesh_client is None:
            self._logging_mesh_client = LoggingMeshClient.spawn_blocking(
                proc_mesh=self._proc_mesh,
            )
        self._logging_mesh_client.set_mode(stream_to_client)

    async def __aenter__(self) -> "ProcMesh":
        if self._stopped:
            raise RuntimeError("`ProcMesh` has already been stopped")
        return self

    def stop(self) -> Future[None]:
        async def _stop_nonblocking() -> None:
            await self._proc_mesh.stop_nonblocking()
            self._stopped = True

        def _stop_blocking() -> None:
            self._proc_mesh.stop_blocking()
            self._stopped = True

        return Future(
            lambda: _stop_nonblocking(),
            lambda: _stop_blocking(),
        )

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


async def local_proc_mesh_nonblocking(
    *, gpus: Optional[int] = None, hosts: int = 1
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = LocalAllocator()
    alloc = await allocator.allocate(spec)
    return await ProcMesh.from_alloc(alloc)


def local_proc_mesh_blocking(
    *,
    gpus: Optional[int] = None,
    hosts: int = 1,
    _is_initializing_debugger: bool = False,
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = LocalAllocator()
    alloc = allocator.allocate(spec).get()
    return ProcMesh(
        HyProcMesh.allocate_blocking(alloc),
        _is_initializing_debugger=_is_initializing_debugger,
    )


def local_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> Future[ProcMesh]:
    return Future(
        lambda: local_proc_mesh_nonblocking(gpus=gpus, hosts=hosts),
        lambda: local_proc_mesh_blocking(gpus=gpus, hosts=hosts),
    )


async def sim_proc_mesh_nonblocking(
    *, gpus: Optional[int] = None, hosts: int = 1
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = SimAllocator()
    alloc = await allocator.allocate(spec)
    return await ProcMesh.from_alloc(alloc)


def sim_proc_mesh_blocking(*, gpus: Optional[int] = None, hosts: int = 1) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = SimAllocator()
    alloc = allocator.allocate(spec).get()
    return ProcMesh.from_alloc(alloc).get()


def sim_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> Future[ProcMesh]:
    return Future(
        lambda: sim_proc_mesh_nonblocking(gpus=gpus, hosts=hosts),
        lambda: sim_proc_mesh_blocking(gpus=gpus, hosts=hosts),
    )


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


async def proc_mesh_nonblocking(
    *, gpus: Optional[int] = None, hosts: int = 1, env: Optional[dict[str, str]] = None
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    # gpus must come last in this order because
    # test_remote_function_all_gather expects that hosts comes before gpus
    # in the order of the dimensions.
    spec = AllocSpec(AllocConstraints(), hosts=hosts, gpus=gpus)
    env = env or {}
    cmd, args, base_env = _get_bootstrap_args()
    env.update(base_env)
    allocator = ProcessAllocator(cmd, args, env)
    alloc = await allocator.allocate(spec)
    return await ProcMesh.from_alloc(alloc)


def proc_mesh_blocking(
    *, gpus: Optional[int] = None, hosts: int = 1, env: Optional[dict[str, str]] = None
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    # gpus must come last in this order
    # because test_remote_function_all_gather expects that hosts comes before gpus
    # in the order of the dimensions.
    spec = AllocSpec(AllocConstraints(), hosts=hosts, gpus=gpus)
    env = env or {}
    cmd, args, base_env = _get_bootstrap_args()
    env.update(base_env)
    allocator = ProcessAllocator(cmd, args, env)
    alloc = allocator.allocate(spec).get()
    return ProcMesh.from_alloc(alloc).get()


def proc_mesh(
    *, gpus: Optional[int] = None, hosts: int = 1, env: Optional[dict[str, str]] = None
) -> Future[ProcMesh]:
    return Future(
        lambda: proc_mesh_nonblocking(gpus=gpus, hosts=hosts, env=env),
        lambda: proc_mesh_blocking(gpus=gpus, hosts=hosts, env=env),
    )


_debug_proc_mesh: Optional["ProcMesh"] = None


# Lazy init of the debug proc mesh so that importing monarch.proc_mesh
# doesn't trigger the debug client to spawn, which could cause confusing
# logs. This is defined in proc_mesh.py instead of debugger.py for
# circular import reasons.
def _get_debug_proc_mesh() -> "ProcMesh":
    global _debug_proc_mesh
    if _debug_proc_mesh is None:
        _debug_proc_mesh = local_proc_mesh_blocking(
            gpus=1, hosts=1, _is_initializing_debugger=True
        )
    return _debug_proc_mesh


_debug_client_mesh: Optional[DebugClient] = None


# Lazy init for the same reason as above. This is defined in proc_mesh.py
# instead of debugger.py for circular import reasons.
def debug_client() -> DebugClient:
    global _debug_client_mesh
    if _debug_client_mesh is None:
        _debug_client_mesh = (
            _get_debug_proc_mesh().spawn("debug_client", DebugClient).get()
        )
    return _debug_client_mesh
