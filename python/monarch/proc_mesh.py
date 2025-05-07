import sys

from functools import cache
from typing import Any, Awaitable, cast, Optional, Type, TypeVar

import monarch._monarch.hyperactor as hyperactor

from monarch.python_local_mesh import _local_device_count
from monarch.rdma import RDMAManager
from monarch.service import _Actor, Actor, ActorMeshRef, Service

T = TypeVar("T")
try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
except ImportError:
    IN_PAR = False


class ProcMesh:
    def __init__(self, hy_proc_mesh: hyperactor.ProcMesh) -> None:
        self._proc_mesh = hy_proc_mesh
        self._mailbox: hyperactor.Mailbox = self._proc_mesh.client
        self._rdma_manager_awaitable: Awaitable[RDMAManager] = self.spawn(
            "rdma_manager", RDMAManager
        )

    def __repr__(self) -> str:
        return repr(self._proc_mesh)

    def __str__(self) -> str:
        return str(self._proc_mesh)

    async def spawn(self, name: str, Class: Type[T], *args: Any, **kwargs: Any) -> T:
        if not issubclass(Class, Actor):
            raise ValueError(
                f"{Class} must subclass monarch.service.Actor to spawn it."
            )
        # init isn't async but we do not need the rdma_manager initialized until
        # we spawn something else. When there is a distinction between the client.
        if self._rdma_manager_awaitable is not None:
            self._rdma_manager_awaitable, awaitable = None, self._rdma_manager_awaitable
            await awaitable
        actor_mesh = await self._proc_mesh.spawn(name, _Actor)
        service = Service(
            Class,
            ActorMeshRef.from_hyperactor_mesh(self._mailbox, actor_mesh),
            self._mailbox,
        )
        # useful to have this separate, because eventually we can reconstitute Service objects across pickling by
        # doing `Service(Class, actor_handle)` but not calling _create.
        service._create(args, kwargs)
        return cast(T, service)


init_asyncio_loop: Any = cache(hyperactor.init_asyncio_loop)


async def local_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> ProcMesh:
    init_asyncio_loop()
    if gpus is None:
        gpus = _local_device_count()
    spec = hyperactor.AllocSpec(hyperactor.AllocConstraints(), gpus=gpus, hosts=hosts)
    alloc = await hyperactor.LocalAllocator.allocate(spec)
    return ProcMesh(await hyperactor.ProcMesh.allocate(alloc))


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


async def proc_mesh(
    *, gpus: Optional[int] = None, hosts: int = 1, env: Optional[dict[str, str]] = None
) -> ProcMesh:
    init_asyncio_loop()
    if gpus is None:
        gpus = _local_device_count()
    spec = hyperactor.AllocSpec(hyperactor.AllocConstraints(), gpus=gpus, hosts=hosts)
    env = env or {}
    cmd, args, base_env = _get_bootstrap_args()
    env.update(base_env)
    env["HYPERACTOR_MANAGED_SUBPROCESS"] = "1"
    allocator = hyperactor.ProcessAllocator(cmd, args, env)
    alloc = await allocator.allocate(spec)
    return ProcMesh(await hyperactor.ProcMesh.allocate(alloc))
