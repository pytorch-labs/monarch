# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys

from typing import Any, cast, List, Optional, Type, TypeVar

import monarch
from monarch import ActorFuture as Future

from monarch._rust_bindings.hyperactor_extension.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension  # @manual=//monarch/monarch_extension:monarch_extension
    Alloc,
    AllocConstraints,
    AllocSpec,
)
from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh as HyProcMesh
from monarch._rust_bindings.monarch_hyperactor.shape import Shape, Slice
from monarch.actor_mesh import _Actor, _ActorMeshRefImpl, Actor, ActorMeshRef

from monarch.common._device_utils import _local_device_count
from monarch.common.shape import MeshTrait
from monarch.rdma import RDMAManager

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
    def __init__(self, hy_proc_mesh: HyProcMesh) -> None:
        self._proc_mesh = hy_proc_mesh
        self._mailbox: Mailbox = self._proc_mesh.client
        self._rdma_manager: RDMAManager = self._spawn_blocking(
            "rdma_manager", RDMAManager
        )

    @property
    def _ndslice(self) -> Slice:
        return self._proc_mesh.shape.ndslice

    @property
    def _labels(self) -> List[str]:
        return self._proc_mesh.shape.labels

    def _new_with_shape(self, shape: Shape) -> "ProcMesh":
        raise NotImplementedError("ProcMesh slicing is not implemeted yet.")

    def spawn(self, name: str, Class: Type[T], *args: Any, **kwargs: Any) -> Future[T]:
        return Future(
            lambda: self._spawn_nonblocking(name, Class, *args, **kwargs),
            lambda: self._spawn_blocking(name, Class, *args, **kwargs),
        )

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
    env["HYPERACTOR_MANAGED_SUBPROCESS"] = "1"
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
    env["HYPERACTOR_MANAGED_SUBPROCESS"] = "1"
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
