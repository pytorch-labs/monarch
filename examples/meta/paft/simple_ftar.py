# pyre-strict

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import random
import socket
import sys
from dataclasses import dataclass
from typing import cast, Dict, List, Literal, Optional, Tuple

import torch

from monarch import (
    DeviceMesh,
    fetch_shard,
    Future,
    OpaqueRef,
    python_local_mesh,
    remote,
)
from monarch.common.invocation import DeviceException, RemoteException
from monarch.rust_backend_mesh import IPoolDeviceMeshProvider, MeshWorld
from monarch.rust_local_mesh import (
    IBootstrap,
    local_mesh_provider,
    LoggingLocation,
    SocketType,
    SupervisionParams,
)
from monarch.sim_mesh import sim_mesh_provider


@dataclass
class MeshInfo:
    mesh: DeviceMesh
    replica_id: int
    ports: Tuple[int, int]
    all_reduce_val: float
    ar_ref: OpaqueRef | None = None


def _provide_rust_meshes(
    num_meshes: int,
    hosts_per_mesh: int,
    gpus_per_host: int,
    # pyre-fixme[11]: Annotation `DeviceMeshProvider` is not defined as a type.
) -> tuple[IPoolDeviceMeshProvider, IBootstrap]:
    return local_mesh_provider(
        meshes=num_meshes,
        hosts_per_mesh=hosts_per_mesh,
        gpus_per_host=gpus_per_host,
        socket_type=SocketType.UNIX,
        logging_location=LoggingLocation.DEFAULT,
        supervision_params=SupervisionParams(
            update_timeout_in_sec=10,  # Fail fast
            query_interval_in_sec=1,
            update_interval_in_sec=1,
        ),
        auto_epoch=True,
    )


def _provide_sim_meshes(
    num_meshes: int,
    hosts_per_mesh: int,
    gpus_per_host: int,
) -> tuple[IPoolDeviceMeshProvider, IBootstrap]:
    return sim_mesh_provider(num_meshes, hosts_per_mesh, gpus_per_host)


setup_ftar = remote(  # pyre-ignore [5]
    "paft.paft_worker.setup_ftar",
    propagate=lambda global_rank, replica_size, ports, device_id: OpaqueRef(None),
)

reconfig_ftar = remote(  # pyre-ignore [5]
    "paft.paft_worker.reconfig_ftar",
    propagate=lambda ar_manager_ref, quorum, step: torch.tensor(1),
)

run_allreduce = remote(  # pyre-ignore [5]
    "paft.paft_worker.run_allreduce",
    propagate=lambda ar_manager_ref, tensor_to_reduce: torch.tensor(1),
)


def _queue_reconfig(
    meshes: list[MeshInfo],
    step: int,
) -> Dict[str, Future]:
    """
    Kicks off FTAR reconfig on all meshes.
    Returns a dict of replica id -> futures of reconfig to complete.
    """
    dummy_quorum = [(mesh_info.ports, mesh_info.replica_id) for mesh_info in meshes]
    futures = {}
    for mesh_info in meshes:
        assert mesh_info.ar_ref is not None
        with mesh_info.mesh.activate():
            fut = reconfig_ftar(
                ar_manager_ref=mesh_info.ar_ref, quorum=dummy_quorum, step=step
            )
            futures[mesh_info.mesh.mesh_name] = fut
    return futures


def _queue_allreduce(
    meshes: list[MeshInfo],
) -> Dict[str, Future]:
    """
    Kicks off allreduces on all meshes.
    Returns a dict of replica id -> futures of allreduces.
    """
    ar_futures = {}
    for mesh_info in meshes:
        with mesh_info.mesh.activate():
            ar_fut = run_allreduce(
                ar_manager_ref=mesh_info.ar_ref,
                tensor_to_reduce=torch.tensor(
                    mesh_info.all_reduce_val, dtype=torch.float32
                ),
            )
            ar_futures[mesh_info.mesh.mesh_name] = ar_fut
    return ar_futures


def _wait_futures(
    meshes: list[MeshInfo], futures: Dict[str, Future]
) -> list[torch.Tensor]:
    """
    Waits for all futures to complete. Future i corresponds to mesh i.

    futures: Replica id -> future
    """
    results = []
    for mesh_info in meshes:
        assert futures[mesh_info.mesh.mesh_name] is not None
        with mesh_info.mesh.activate():
            results.append(fetch_shard(futures[mesh_info.mesh.mesh_name]).result())
    return results


def _exit_meshes(meshes: List[MeshInfo]) -> None:
    print("Done. exiting", file=sys.stderr, flush=True)
    for mesh_info in meshes:
        mesh_info.mesh.exit()


class MiniScheduler:
    """
    A scheduler responsible for the lifecycle management of a set of meshes.
    This is a mock implementation of MAST.

    The scheduler can randomly kill a mesh and recover it.
    """

    def __init__(self, bootstrap: IBootstrap) -> None:
        self._bootstrap = bootstrap

    def restart_a_mesh(self) -> None:
        """
        Randomly kill a mesh from the existing spawned meshes.
        It will also perform a recovery of the killed mesh to mimick the behavior of MAST
        as if it is restarting a task group.
        """
        mesh_worlds = self._bootstrap.get_mesh_worlds()

        # Kill a random mesh
        mesh_to_kill: MeshWorld = random.choice(mesh_worlds)
        self._bootstrap.kill_mesh(mesh_to_kill)

        # Recover the killed mesh
        self._bootstrap.spawn_mesh(mesh_to_kill)


def _find_free_port() -> Tuple[int, socket.socket]:
    """Finds a free port on the host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    return s.getsockname()[1], s


def _build_meshes(
    mesh_type: Literal["rust_local", "python", "sim"],
    num_meshes: int,
    mesh_to_allreduce_val: dict[int, float] | None = None,
) -> Tuple[List[MeshInfo], Optional[IBootstrap], Optional[IPoolDeviceMeshProvider]]:
    """
    Builds a list of meshes. Each mesh is a single GPU on a single host.
    """
    ports_and_sockets = {
        i: (_find_free_port(), _find_free_port()) for i in range(num_meshes)
    }

    mesh_to_allreduce_val = mesh_to_allreduce_val or {
        i: float(i + 1) for i in range(num_meshes)
    }

    meshes, bootstrap, mesh_provider = None, None, None

    match mesh_type:
        case "python":
            meshes = [python_local_mesh(hosts=1, gpus=1) for _ in range(num_meshes)]
        case "rust_local":
            mesh_provider, bootstrap = _provide_rust_meshes(
                num_meshes=num_meshes, hosts_per_mesh=1, gpus_per_host=1
            )
            meshes = []
            while len(meshes) < num_meshes:
                mesh = mesh_provider.new_mesh()
                meshes.append(mesh)
        case "sim":
            mesh_provider, bootstrap = _provide_sim_meshes(
                num_meshes=num_meshes, hosts_per_mesh=1, gpus_per_host=1
            )
            meshes = []
            while len(meshes) < num_meshes:
                mesh = mesh_provider.new_mesh()
                meshes.append(mesh)

    meshes_with_info = [
        MeshInfo(
            mesh=mesh,
            replica_id=i,
            ports=(ports_and_sockets[i][0][0], ports_and_sockets[i][1][0]),
            all_reduce_val=mesh_to_allreduce_val[i],
        )
        for i, mesh in enumerate(meshes)
    ]

    # Close the sockets when you're ready to use the ports
    for i in range(num_meshes):
        ports_and_sockets[i][0][1].close()
        ports_and_sockets[i][1][1].close()
    return meshes_with_info, bootstrap, mesh_provider


def main(
    mesh_type: Literal["rust_local", "python", "sim"],
    num_meshes: int,
    mesh_to_allreduce_val: dict[int, float] | None = None,
) -> None:
    """
    Very simple FTAR running on Monarch using N meshes and FTAR
    communication across meshes.

    Each mesh uses the same host and a single GPU each. Quorum + port
    information is hardcoded.
    """

    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= num_meshes
    ), f"need at least {num_meshes} GPUs"
    if mesh_to_allreduce_val is None:
        mesh_to_allreduce_val = {i: float(i + 1) for i in range(num_meshes)}
    assert (
        len(mesh_to_allreduce_val) == num_meshes
    ), f"need {num_meshes} values to allreduce"

    meshes, bootstrap, mesh_provider = _build_meshes(
        mesh_type, num_meshes, mesh_to_allreduce_val
    )

    for mesh_info in meshes:
        with mesh_info.mesh.activate():
            ar_ref = setup_ftar(
                global_rank=mesh_info.replica_id,
                replica_size=1,
                ports=mesh_info.ports,
                device_id=0,
            )
            mesh_info.ar_ref = ar_ref

    reconfig_futures = _queue_reconfig(meshes, step=0)
    _wait_futures(meshes, reconfig_futures)

    ar_futures = _queue_allreduce(meshes)
    results = _wait_futures(meshes, ar_futures)

    for replica_id, result in enumerate(results):
        print(f"r{replica_id}_result: {result}", flush=True)

    # only rust local supports shrink and grow.
    if mesh_type != "rust_local":
        return

    result_ints_healthy = {result.item() for result in results}
    assert (
        len(result_ints_healthy) == 1
    ), f"Mismatch allreduce results: {result_ints_healthy}"
    assert result_ints_healthy.pop() == sum([mesh.all_reduce_val for mesh in meshes])

    assert bootstrap is not None

    scheduler = MiniScheduler(bootstrap)
    # Kill a random mesh and restart it.
    scheduler.restart_a_mesh()

    # The killed mesh will become unhealthy, resulting in FTAR failure.
    # Wait on allreduces, which will fail on both meshes, with Worker or user error.
    ar_futures = _queue_allreduce(meshes)
    healthy_meshes: List[MeshInfo] = []
    unhealthy_meshes: List[MeshInfo] = []
    for mesh_info in meshes:
        mesh = mesh_info.mesh
        with mesh.activate():
            try:
                result = fetch_shard(ar_futures[mesh.mesh_name]).result()
                healthy_meshes.append(mesh_info)
            except RemoteException as e:
                # Allreduce failure on a healthy mesh, since it tried to reduce
                # with an unhealthy mesh. Currently we just return a hardcoded RuntimeError.
                # In the future, we should see if FTAR library can actually raise the exception
                # indicating timeout.
                assert "AllReduce failed" in str(e)
                healthy_meshes.append(mesh_info)
            except DeviceException:
                # Allreduce failure on an unhealthy mesh that crashed or was
                # unresponsive
                unhealthy_meshes.append(mesh_info)

    assert len(unhealthy_meshes) == 1
    killed_mesh = unhealthy_meshes[0]
    assert len(healthy_meshes) == num_meshes - 1
    meshes = healthy_meshes

    reconfig_futures = _queue_reconfig(meshes, step=1)
    _wait_futures(meshes, reconfig_futures)
    ar_futures = _queue_allreduce(meshes)
    results_shrink = _wait_futures(meshes, ar_futures)

    results_shrink = {r.item() for r in results_shrink}
    assert len(results_shrink) == 1, f"Mismatch allreduce results: {results_shrink}"
    assert results_shrink.pop() == sum([mesh.all_reduce_val for mesh in meshes])

    assert mesh_provider is not None

    # Get a brand new mesh for the unhealthy replica
    # The new mesh is available because the MiniScheduler is restarting a new mesh
    recovered_mesh = mesh_provider.new_mesh()
    ports_and_sockets = (_find_free_port(), _find_free_port())
    recovered_mesh_info = MeshInfo(
        replica_id=killed_mesh.replica_id,
        mesh=recovered_mesh,
        ports=(ports_and_sockets[0][0], ports_and_sockets[1][0]),
        all_reduce_val=killed_mesh.all_reduce_val,
    )
    meshes.append(recovered_mesh_info)

    # Resetup FTAR on the recovered mesh
    ports_and_sockets[0][1].close()
    ports_and_sockets[1][1].close()
    with recovered_mesh_info.mesh.activate():
        ar_ref = setup_ftar(
            global_rank=recovered_mesh_info.replica_id,
            replica_size=1,
            ports=recovered_mesh_info.ports,
            device_id=0,
        )
        recovered_mesh_info.ar_ref = ar_ref

    # Reconfig + allreduce on the recovered mesh
    reconfig_futures = _queue_reconfig(meshes, step=2)
    _wait_futures(meshes, reconfig_futures)

    ar_futures = _queue_allreduce(meshes)
    results_grow = _wait_futures(meshes, ar_futures)
    results_grow = {result.item() for result in results_grow}
    assert len(results_grow) == 1, f"Mismatch allreduce results: {results_grow}"
    assert results_grow.pop() == sum(mesh_to_allreduce_val.values())

    _exit_meshes(meshes)


if __name__ == "__main__":
    main(
        mesh_type=cast(Literal["python", "rust_local", "sim"], sys.argv[1]),
        num_meshes=2,
    )
