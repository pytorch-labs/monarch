# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import sys
import warnings
from contextlib import AbstractContextManager

from typing import (
    Any,
    Callable,
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
from monarch._src.actor.actor_mesh import (
    _Actor,
    _ActorMeshRefImpl,
    Actor,
    ActorMeshRef,
    fake_sync_state,
)

from monarch._src.actor.allocator import LocalAllocator, ProcessAllocator, SimAllocator
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
from monarch._src.actor.future import Future
from monarch._src.actor.shape import MeshTrait

HAS_TENSOR_ENGINE = False
try:
    from monarch._rust_bindings.rdma import (  # type: ignore[import]
        _RdmaBuffer,
        _RdmaManager,
    )

    # type: ignore[16]
    HAS_TENSOR_ENGINE = _RdmaBuffer.rdma_supported()
except ImportError:
    logging.warning("RDMA is not available on this platform")


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


async def _allocate_nonblocking(
    alloc: Alloc, setup: Callable[[], None] | None = None
) -> "ProcMesh":
    _proc_mesh = await HyProcMesh.allocate_nonblocking(alloc)
    if setup is None:
        return ProcMesh(_proc_mesh)
    # If the user has passed the setup lambda, we need to call
    # it here before any of the other actors are spawned so that
    # the environment variables are set up before cuda init.
    proc_mesh = ProcMesh(_proc_mesh)
    setup_actor = await proc_mesh.spawn("setup", SetupActor, setup)
    await setup_actor.setup.call()
    del setup_actor
    return proc_mesh


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
        self._rdma_manager: Optional["_RdmaManager"] = None
        self._debug_manager: Optional[DebugManager] = None
        self._mailbox: Mailbox = self._proc_mesh.client
        self._code_sync_client: Optional[CodeSyncMeshClient] = None
        self._logging_mesh_client: Optional[LoggingMeshClient] = None
        self._maybe_device_mesh: Optional["DeviceMesh"] = _device_mesh
        self._stopped = False

        # This code is unsafe in async contexts, but we currently do it all over the place
        # we need to refactor this by moving it to the first time we try to spawn on the mesh.
        # Right now we simply preserve the previous behavior and disable the check that prevents
        # end users from doing the same.
        with fake_sync_state():
            if _mock_shape is None and HAS_TENSOR_ENGINE:
                # type: ignore[21]
                self._rdma_manager = _RdmaManager.create_rdma_manager_blocking(
                    self._proc_mesh
                )
            if not _is_initializing_debugger and _mock_shape is None:
                self._debug_manager = self.spawn(
                    _DEBUG_MANAGER_ACTOR_NAME, DebugManager, debug_client()
                ).get()

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
            impl=lambda: self._spawn_nonblocking(name, Class, *args, **kwargs),
            requires_loop=False,
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
    def from_alloc(
        self, alloc: Alloc, setup: Callable[[], None] | None = None
    ) -> Future["ProcMesh"]:
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
        return Future(
            impl=lambda: _allocate_nonblocking(alloc, setup),
            requires_loop=False,
        )

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
        if self._code_sync_client is None:
            self._code_sync_client = CodeSyncMeshClient.spawn_blocking(
                proc_mesh=self._proc_mesh,
            )
        # TODO(agallagher): We need some way to configure and pass this
        # in -- right now we're assuming the `gpu` dimension, which isn't
        # correct.
        # The workspace shape (i.e. only perform one rsync per host).
        assert set(self._proc_mesh.shape.labels).issubset({"gpus", "hosts"})
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
        stream_to_client: bool = False,
        aggregate_window_sec: int | None = None,
    ) -> None:
        """
        Set the logging options for the remote processes

        Args:
            stream_to_client (bool): If True, logs from the remote processes will be streamed to the client.
            Defaults to False.
            aggregate_window_sec (Optional[int]): If not None, logs from the remote processes will be aggregated
            and sent to the client every aggregate_window_sec seconds. Defaults to None, meaning no aggregation.
            aggregate_window_sec will be ignored if stream_to_client is False.

        Returns:
            None
        """
        if self._logging_mesh_client is None:
            self._logging_mesh_client = await LoggingMeshClient.spawn(
                proc_mesh=self._proc_mesh
            )
        self._logging_mesh_client.set_mode(
            stream_to_client, aggregate_window_sec=aggregate_window_sec
        )

    async def __aenter__(self) -> "ProcMesh":
        if self._stopped:
            raise RuntimeError("`ProcMesh` has already been stopped")
        return self

    def stop(self) -> Future[None]:
        async def _stop_nonblocking() -> None:
            await self._proc_mesh.stop_nonblocking()
            self._stopped = True

        return Future(
            impl=lambda: _stop_nonblocking(),
            requires_loop=False,
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
    *,
    gpus: Optional[int] = None,
    hosts: int = 1,
    _is_initializing_debugger: bool = False,
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=hosts)
    allocator = LocalAllocator()
    alloc = await allocator.allocate(spec)
    proc_mesh = HyProcMesh.allocate_nonblocking(alloc)
    return ProcMesh(
        await proc_mesh,
        _is_initializing_debugger=_is_initializing_debugger,
    )


def local_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> Future[ProcMesh]:
    return Future(
        impl=lambda: local_proc_mesh_nonblocking(gpus=gpus, hosts=hosts),
        requires_loop=False,
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


def sim_proc_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> Future[ProcMesh]:
    return Future(
        impl=lambda: sim_proc_mesh_nonblocking(gpus=gpus, hosts=hosts),
        requires_loop=False,
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
    *,
    gpus: Optional[int] = None,
    hosts: int = 1,
    env: dict[str, str] | None = None,
    setup: Callable[[], None] | None = None,
) -> ProcMesh:
    if gpus is None:
        gpus = _local_device_count()
    # gpus must come last in this order because
    # test_remote_function_all_gather expects that hosts comes before gpus
    # in the order of the dimensions.
    spec = AllocSpec(AllocConstraints(), hosts=hosts, gpus=gpus)
    env = env or {}
    # Todo: Deprecate the env field from the ProcessAllocator
    # The PAR_MAIN_OVERRIDE needs to be passed as an env
    # to the proc mesh construction in rust, so can not be moved to the
    # SetupActor yet
    cmd, args, bootstrap_env = _get_bootstrap_args()
    env.update(bootstrap_env)
    allocator = ProcessAllocator(cmd, args, env)
    alloc = await allocator.allocate(spec)

    return await ProcMesh.from_alloc(
        alloc,
        setup=setup,
    )


def proc_mesh(
    *,
    gpus: Optional[int] = None,
    hosts: int = 1,
    env: dict[str, str] | None = None,
    setup: Callable[[], None] | None = None,
) -> Future[ProcMesh]:
    return Future(
        impl=lambda: proc_mesh_nonblocking(
            gpus=gpus, hosts=hosts, env=env, setup=setup
        ),
        requires_loop=False,
    )


_debug_proc_mesh: Optional["ProcMesh"] = None


# Lazy init of the debug proc mesh so that importing monarch.proc_mesh
# doesn't trigger the debug client to spawn, which could cause confusing
# logs. This is defined in proc_mesh.py instead of debugger.py for
# circular import reasons.
def _get_debug_proc_mesh() -> "ProcMesh":
    global _debug_proc_mesh
    if _debug_proc_mesh is None:
        _debug_proc_mesh = Future(
            impl=lambda: local_proc_mesh_nonblocking(
                gpus=1, hosts=1, _is_initializing_debugger=True
            ),
            requires_loop=False,
        ).get()
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
