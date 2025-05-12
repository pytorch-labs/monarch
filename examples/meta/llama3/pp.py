# pyre-unsafe
import logging
from itertools import chain
from typing import Dict, List

import torch.nn as nn
import torch.utils._pytree as pytree
from monarch import Tensor
from monarch.common.device_mesh import DeviceMesh
from monarch.gradient_generator import grad_function

logging.basicConfig(level=logging.INFO)


@grad_function
def to_mesh(x, mesh):
    omesh = x.mesh

    def backward(grad_x: Tensor):
        return grad_x.to_mesh(omesh), None

    return x.to_mesh(mesh), backward


class PipelineParallelism:
    def __init__(
        self,
        meshes: List[DeviceMesh],
        stages: List[List[nn.Module]],
    ):
        self.stages = stages
        self.meshes = meshes

    def initialize(
        self,
    ):
        pp_stages = self.stages
        assert len(pp_stages) == len(self.meshes)

        for stage_idx, stage in enumerate(pp_stages):
            for module in stage:
                state_dict = module.state_dict()
                for k, v in state_dict.items():
                    if isinstance(v, Tensor):
                        state_dict[k] = v.to_mesh(self.meshes[stage_idx])
                module.load_state_dict(state_dict, assign=True)

        for mesh, stages in zip(self.meshes, pp_stages):

            def move_mb(mesh):
                def hook(module, args):
                    return pytree.tree_map(lambda x: to_mesh(x, mesh), args)

                return hook

            stages[0].register_forward_pre_hook(move_mb(mesh))

    def configure_optimizers(self, config, config_fn):
        optimizers = []

        for stage in self.stages:
            params = list(chain(*[list(m.parameters()) for m in stage]))
            optimizers.append(
                config_fn(
                    config.weight_decay,
                    config.learning_rate,
                    (config.beta1, config.beta2),
                    config.device_type,
                    config.optimizer,
                    params,
                )
            )

        return optimizers
