# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import sys
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

if TYPE_CHECKING:
    import torch

import monarch
from monarch import ActorFuture as Future

# Conditionally import DeviceMesh and spawn_tensor_engine only if tensor_engine is available
# pyre-ignore[21]
from monarch._rust_bindings import has_tensor_engine

from monarch._rust_bindings.hyperactor_extension.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension  # @manual=//monarch/monarch_extension:monarch_extension
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
from monarch.actor_mesh import _Actor, _ActorMeshRefImpl, Actor, ActorMeshRef
from monarch.code_sync import RsyncMeshClient, WorkspaceLocation
from monarch.code_sync.auto_reload import AutoReloadActor
from monarch.common._device_utils import _local_device_count
from monarch.common.shape import MeshTrait
from monarch.rdma import RDMAManager

if has_tensor_engine():
    from monarch.common.device_mesh import DeviceMesh
    from monarch.mesh_controller import spawn_tensor_engine
else:
    DeviceMesh = None
    spawn_tensor_engine = None

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
        _device_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        self._proc_mesh = hy_proc_mesh
        self._mock_shape: Optional[Shape] = _mock_shape
        self._mailbox: Mailbox = self._proc_mesh.client
        self._rdma_manager: Optional[RDMAManager] = None
        self._rsync_mesh_client: Optional[RsyncMeshClient] = None
        self._auto_reload_actor: Optional[AutoReloadActor] = None
        self._maybe_device_mesh: Optional[DeviceMesh] = _device_mesh
        self._stopped = False
        if _mock_shape is None:
            self._rdma_manager = self._spawn_blocking("rdma_manager", RDMAManager)

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
            if self._device_mesh is None
            else self._device_mesh._new_with_shape(shape)
        )
        return ProcMesh(self._proc_mesh, _mock_shape=shape, _device_mesh=device_mesh)

    def spawn(
        self, name: str, Class: Type[T], *args: Any, **kwargs: Any
    ) -> Future[ActorMeshRef[T]]:
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
            _ActorMeshRefImpl.from_hyperactor_mesh(self._mailbox, actor_mesh),
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
            _ActorMeshRefImpl.from_hyperactor_mesh(self._mailbox, actor_mesh),
            self._mailbox,
        )
        # useful to have this separate, because eventually we can reconstitute ActorMeshRef objects across pickling by
        # doing `ActorMeshRef(Class, actor_handle)` but not calling _create.
        service._create(args, kwargs)
        return cast(T, service)

    @property
    def _device_mesh(self) -> "DeviceMesh":
        if spawn_tensor_engine is None:
            raise RuntimeError(
                "DeviceMesh is not available because tensor_engine was not compiled (USE_TENSOR_ENGINE=0)"
            )
        if self._maybe_device_mesh is None:
            if self._mock_shape is not None:
                raise NotImplementedError(
                    "NYI: activating a proc mesh must first happen on the root proc_mesh until we fix spawning on submeshes."
                )
            self._maybe_device_mesh = spawn_tensor_engine(self)
        return self._maybe_device_mesh

    # pyre-ignore
    def activate(self) -> AbstractContextManager:
        return self._device_mesh.activate()

    def rank_tensor(self, dim: str | Sequence[str]) -> "torch.Tensor":
        return self._device_mesh.rank(dim)

    def rank_tensors(self) -> Dict[str, "torch.Tensor"]:
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

    async def stop(self) -> None:
        await self._proc_mesh.stop()
        self._stopped = True

    async def __aenter__(self) -> "ProcMesh":
        if self._stopped:
            raise RuntimeError("`ProcMesh` has already been stopped")
        return self

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
            import warnings

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
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate(spec)
    return await ProcMesh.from_alloc(alloc)


def local_proc_mesh_blocking(*, gpus: Optional[int] = None, hosts: int = 1) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = monarch.LocalAllocator()
    alloc = allocator.allocate(spec).get()
    return ProcMesh.from_alloc(alloc).get()


def local_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> Future[ProcMesh]:
    return Future(
        lambda: local_proc_mesh_nonblocking(gpus=gpus, hosts=hosts),
        lambda: local_proc_mesh_blocking(gpus=gpus, hosts=hosts),
    )


_BOOTSTRAP_MAIN = "monarch.bootstrap_main"


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
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    env = env or {}
    cmd, args, base_env = _get_bootstrap_args()
    env.update(base_env)
    allocator = monarch.ProcessAllocator(cmd, args, env)
    alloc = await allocator.allocate(spec)
    return await ProcMesh.from_alloc(alloc)


def proc_mesh_blocking(
    *, gpus: Optional[int] = None, hosts: int = 1, env: Optional[dict[str, str]] = None
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    env = env or {}
    cmd, args, base_env = _get_bootstrap_args()
    env.update(base_env)
    allocator = monarch.ProcessAllocator(cmd, args, env)
    alloc = allocator.allocate(spec).get()
    return ProcMesh.from_alloc(alloc).get()


def proc_mesh(
    *, gpus: Optional[int] = None, hosts: int = 1, env: Optional[dict[str, str]] = None
) -> Future[ProcMesh]:
    return Future(
        lambda: proc_mesh_nonblocking(gpus=gpus, hosts=hosts, env=env),
        lambda: proc_mesh_blocking(gpus=gpus, hosts=hosts, env=env),
    )
