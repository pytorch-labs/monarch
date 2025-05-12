import sys
from typing import List

import torch

from llama3.config import TrainConfig

from llama3.train import train
from monarch import world_mesh
from monarch_supervisor import Context, Host
from monarch_supervisor.launchers import mast


def run_mast():
    def supervise(ctx: Context, hosts: List[Host]):
        n_gpus = torch.cuda.device_count()
        device_mesh = world_mesh(ctx, hosts, n_gpus)
        world_size = len(hosts) * n_gpus
        TrainConfig.configure(sys.argv[1:])
        return train(device_mesh, world_size)

    return mast(supervise)


if __name__ == "__main__":
    ret = run_mast()
    sys.exit(ret)
