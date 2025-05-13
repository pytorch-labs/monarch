# pyre-unsafe
import os
import re
import subprocess
from pathlib import Path
from time import sleep
from typing import Optional, TYPE_CHECKING

import monarch_supervisor
from monarch.common.fake import fake_call
from monarch.common.invocation import DeviceException, RemoteException
from monarch.world_mesh import world_mesh
from monarch_supervisor import Context, HostConnected
from monarch_supervisor.python_executable import PYTHON_EXECUTABLE

if TYPE_CHECKING:
    from monarch.common.device_mesh import DeviceMesh


class PythonLocalContext:
    def __init__(self, N: int):
        # do a fake call to instantiate ThreadPoolExecutor so we don't block GIL later
        fake_call(lambda: 0)

        self.ctx = ctx = Context()
        ctx.request_hosts(N)

        # we want ctx to start its listener threads
        # before creating the hosts because
        # initialization will happen faster in this case
        sleep(0)
        supervisor_addr = f"tcp://127.0.0.1:{ctx.port}"

        env = {
            **os.environ,
            "TORCH_SUPERVISOR_HEARTBEAT_INTERVAL": str(
                monarch_supervisor.HEARTBEAT_INTERVAL
            ),
            # This is needed to avoid a hard failure in ncclx when we do not
            # have backend topology info (eg. on RE).
            "NCCL_IGNORE_TOPO_LOAD_FAILURE": "true",
        }

        # start_new_session=True, because we want the host managers to be able to kill
        # any worker processes before they exit, even if the supervisor crashes, or we ctrl-c
        # it in testing.
        self.host_managers = [
            subprocess.Popen(
                [
                    PYTHON_EXECUTABLE,
                    "-m",
                    "monarch_supervisor.host",
                    supervisor_addr,
                ],
                env=env,
                start_new_session=True,
            )
            for _ in range(N)
        ]
        connections = ctx.messagefilter(HostConnected)
        self.hosts = [connections.recv(timeout=30).sender for _ in range(N)]

    def shutdown(self):
        self.ctx.shutdown()
        for host_manager in self.host_managers:
            host_manager.wait(timeout=10)


def _local_device_count():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    dev_path = Path("/dev")
    pattern = re.compile(r"nvidia\d+$")
    nvidia_devices = [dev for dev in dev_path.iterdir() if pattern.match(dev.name)]
    return len(nvidia_devices)


def python_local_mesh(*, gpus: Optional[int] = None, hosts: int = 1) -> "DeviceMesh":
    """
    Creates a local device mesh with the given number of hosts and gpus per host.
    Easy way to use PythonLocalContext.

    Args:
        gpus (Optional[int]): number of gpus per host.
                              Default: the number of GPUs this machine has.

        hosts (int): number of hosts, primarily used for simulating multiple machines locally.
                     Default: 1

    Example::
        local_mesh = python_local_mesh(gpus=2)
        with local_mesh.activate():
            x = torch.rand(3, 4)
            local_tensor = fetch_shard(x).result()

        # Cleanly shut down the local mesh and exit.
        local_mesh.exit()
    """
    ctx = PythonLocalContext(hosts)
    if gpus is None:
        gpus = _local_device_count()
    dm = world_mesh(ctx.ctx, ctx.hosts, gpus)

    def exit(
        error: Optional[RemoteException | DeviceException | Exception] = None,
    ) -> None:
        dm.client.shutdown(True, error)
        ctx.shutdown()

    dm.exit = exit
    return dm
