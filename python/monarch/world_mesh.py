# pyre-unsafe

from typing import List

from monarch.common.client import Client

from monarch.common.device_mesh import DeviceMesh
from monarch.common.shape import NDSlice

from monarch.controller.backend import ProcessBackend

from monarch.controller.controller import Controller
from monarch_supervisor import Context, Host


def world_mesh(
    ctx: Context,
    hosts: List[Host],
    gpu_per_host: int,
    _processes=None,
) -> DeviceMesh:
    backend = ProcessBackend(ctx, hosts, gpu_per_host, _processes=_processes)
    client = Client(Controller(backend), backend.world_size, backend.gpu_per_host)
    return DeviceMesh(
        client,
        NDSlice(offset=0, sizes=[len(hosts), gpu_per_host], strides=[gpu_per_host, 1]),
        ("host", "gpu"),
    )
