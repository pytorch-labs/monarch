# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# @noautodeps
# pyre-ignore-all-errors

import argparse
import asyncio
import getpass
import json
import logging
import os
import pathlib
import sys

import cloudpickle

from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec
# from monarch._src.actor.meta.allocator import MastAllocator, MastAllocatorConfig

from monarch.actor import ProcMesh
from monarch.tools import commands
from monarch.tools.components import hyperactor
from monarch.tools.config import Config, UnnamedAppDef
from monarch._src.actor.allocator import RemoteAllocator, StaticRemoteAllocInitializer, TorchXRemoteAllocInitializer


import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from monarch.actor import Actor, current_rank, current_size, endpoint


# class TestActor(Actor):
#     """Silly actor that computes the world size by all-reducing rank-hot tensors"""

#     def __init__(self) -> None:
#         pass

#     @endpoint
#     async def compute_world_size(self, master_addr: str, master_port: int) -> int:
#         rank: int = current_rank().rank
#         world_size: int = math.prod(current_size().values())

#         backend = "gloo"
#         os.environ["MASTER_ADDR"] = master_addr
#         os.environ["MASTER_PORT"] = str(master_port)

#         print(f"""Initializing process group `{backend}`:
#   MASTER_ADDR = {master_addr}
#   MASTER_PORT = {master_port}
#   RANK        = {rank}
#   WORLD_SIZE  = {world_size}""")

#         dist.init_process_group(backend, rank=rank, world_size=world_size)

#         try:
#             t = F.one_hot(torch.tensor(rank), num_classes=dist.get_world_size())
#             dist.all_reduce(t)
#             return int(torch.sum(t).item())
#         finally:
#             dist.destroy_process_group()


USER = getpass.getuser()
HOME = pathlib.Path().home()
CWD = os.getcwd()
DEACTIVATE = None

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


logger: logging.Logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--force-restart",
    action="store_true",
    default=False,
    help="Force restart the job (default: False)",
)

args = parser.parse_args()

class TestActor(Actor):
    def __init__(self):
        pass

    @endpoint
    async def compute_world_size(self, master_addr, master_port):
        return 42


async def main():
    jobname = f"monarch-{USER}"

    # similar to Docker image; should contain a conda env in the $img_root/conda/ directory
    # when config.workspace is not None, an ephemeral fbpkg version is created
    # that conda-packs the currently active local conda env AND the directory specified by workspace
    image = "monarch_default_workspace:latest"

    num_hosts = 2
    num_gpus_per_host = 4

    config = Config(
        scheduler="slurm",
        scheduler_args={
            # NOTE: replace with your own values
            "hpcIdentity": "pytorch_distributed",
            "hpcJobOncall": "monarch",
            "hpcClusterUuid": "MastProdCluster",
            "rmAttribution": "pytorch4all_clients_approved",
        },
        appdef=hyperactor.host_mesh(
            image=image,
            # TODO: For some reason gpu.medium doens't work here
            meshes=[f"mesh0:{num_hosts}:gpu.large"],  # mesh_name:num_hosts:host_type
        ),
        workspace=str(CWD),  # or None to disable building ephemeral,
    )

    server_info = await commands.get_or_create(
        jobname,
        config,
        force_restart=args.force_restart,
    )
    print(f"ahmad: {server_info}")

    logger.info(
        "\n===== Server Info =====\n%s",
        json.dumps(server_info.to_json(), indent=2),
    )

    mesh_dimensions = {
        "host": server_info.get_mesh_spec("mesh0").num_hosts,
        "gpu": server_info.get_mesh_spec("mesh0").gpus,
    }
    # this is redundant but is here for example sake
    mesh_name = server_info.get_mesh_spec("mesh0").name

    allocator = RemoteAllocator(world_id="foo", initializer=TorchXRemoteAllocInitializer(f"slurm:///{server_info.name}"))
    alloc = await allocator.allocate(AllocSpec(AllocConstraints(), hosts=num_hosts, gpus=num_gpus_per_host))

    proc_mesh = await ProcMesh.from_alloc(alloc)
    actor = await proc_mesh.spawn("compute_world_size_actor", TestActor)

    logger.info("computing world size...")
    values = await actor.compute_world_size.call(
        master_addr=server_info.host0(mesh_name),
        master_port=29500,
    )

    values_by_rank = {f"rank_{p.rank}": v for p, v in list(values.flatten("rank"))}

    logger.info(f"""computed world_sizes:
{'-'*40}
{json.dumps(values_by_rank, indent=2)}
{'-'*40}""")
    commands.kill(f"slurm:///{server_info.name}")


if __name__ == "__main__":
    cloudpickle.register_pickle_by_value(sys.modules[TestActor.__module__])

    asyncio.run(main())