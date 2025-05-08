import sys

from functools import cache
from typing import Any, cast, Optional, Type, TypeVar

import monarch
import monarch._monarch.hyperactor as hyperactor
from monarch import ActorFuture as Future
from monarch._monarch.hyperactor import Alloc

from monarch.python_local_mesh import _local_device_count
from monarch.rdma import RDMAManager
from monarch.service import _Actor, Actor, ActorMeshRef, Service

T = TypeVar("T")
try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
except ImportError:
    IN_PAR = False


async def _allocate_nonblocking(alloc: Alloc) -> "ProcMesh":
    return ProcMesh(await hyperactor.ProcMesh.allocate_nonblocking(alloc))


def _allocate_blocking(alloc: Alloc) -> "ProcMesh":
    return ProcMesh(hyperactor.ProcMesh.allocate_blocking(alloc))


class ProcMesh:
    def __init__(self, hy_proc_mesh: hyperactor.ProcMesh) -> None:
        self._proc_mesh = hy_proc_mesh
        self._mailbox: hyperactor.Mailbox = self._proc_mesh.client
        self._rdma_manager = self._spawn_blocking("rdma_manager", RDMAManager)

    def spawn(self, name: str, Class: Type[T], *args: Any, **kwargs: Any) -> Future[T]:
        return Future(
            self._spawn_nonblocking(name, Class, *args, **kwargs),
            lambda: self._spawn_blocking(name, Class, *args, **kwargs),
        )

    @classmethod
    def from_alloc(self, alloc: Alloc) -> Future["ProcMesh"]:
        return Future(
            _allocate_nonblocking(alloc),
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
        service = Service(
            Class,
            ActorMeshRef.from_hyperactor_mesh(self._mailbox, actor_mesh),
            self._mailbox,
        )
        # useful to have this separate, because eventually we can reconstitute Service objects across pickling by
        # doing `Service(Class, actor_handle)` but not calling _create.
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
        service = Service(
            Class,
            ActorMeshRef.from_hyperactor_mesh(self._mailbox, actor_mesh),
            self._mailbox,
        )
        # useful to have this separate, because eventually we can reconstitute Service objects across pickling by
        # doing `Service(Class, actor_handle)` but not calling _create.
        service._create(args, kwargs)
        return cast(T, service)


async def local_proc_mesh_nonblocking(
    *, gpus: Optional[int] = None, hosts: int = 1
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = hyperactor.AllocSpec(hyperactor.AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate(spec)
    return await ProcMesh.from_alloc(alloc)


def local_proc_mesh_blocking(*, gpus: Optional[int] = None, hosts: int = 1) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = hyperactor.AllocSpec(hyperactor.AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = monarch.LocalAllocator()
    alloc = allocator.allocate(spec).get()
    return ProcMesh.from_alloc(alloc).get()


def local_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> Future[ProcMesh]:
    return Future(
        local_proc_mesh_nonblocking(gpus=gpus, hosts=hosts),
        lambda: local_proc_mesh_blocking(gpus=gpus, hosts=hosts),
    )


_BOOTSTRAP_MAIN = "monarch._monarch.hyperactor.bootstrap_main"


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
    spec = hyperactor.AllocSpec(hyperactor.AllocConstraints(), gpus=gpus, hosts=hosts)
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
    spec = hyperactor.AllocSpec(hyperactor.AllocConstraints(), gpus=gpus, hosts=hosts)
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
        proc_mesh_nonblocking(gpus=gpus, hosts=hosts, env=env),
        lambda: proc_mesh_blocking(gpus=gpus, hosts=hosts, env=env),
    )
