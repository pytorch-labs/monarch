# pyre-unsafe
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn

from dlrm.data_loader import generate_id_list_features
from monarch.common.opaque_ref import OpaqueRef
from monarch.examples.dlrm.model import get_dlrm
from torch import optim
from torch.distributed import ProcessGroup

from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.types import ShardingEnv


@dataclass
class DLRMModel:
    core_model: nn.Module
    pg: ProcessGroup
    float_features_count: int
    num_sprase_features: int
    embedding_dim: int
    embedding_hash_size: int
    loss_fn: nn.Module
    optimizer: optim.Optimizer


def initialize_model(
    rank,
    world_size,
    float_features_count,
    dense_arch_layers,
    over_arch_layers,
    num_sprase_features,
    embedding_dim,
    embedding_hash_size,
):
    torch.set_default_device("cuda")
    model = get_dlrm(
        float_features_count,
        dense_arch_layers,
        over_arch_layers,
        num_sprase_features,
        embedding_dim,
        embedding_hash_size,
    )
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:12345",
        timeout=timedelta(seconds=30),
        rank=int(rank),
        world_size=int(world_size),
    )
    world_pg = dist.group.WORLD

    dmp = DistributedModelParallel(
        module=model,
        device=torch.device("cuda"),
        env=ShardingEnv.from_process_group(world_pg),
        init_data_parallel=True,
        init_parameters=True,
    )
    return OpaqueRef(
        DLRMModel(
            core_model=dmp,
            pg=world_pg,
            float_features_count=float_features_count,
            num_sprase_features=num_sprase_features,
            embedding_dim=embedding_dim,
            embedding_hash_size=embedding_hash_size,
            loss_fn=nn.BCEWithLogitsLoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
        )
    )


def dlrm_train_setp(model_ref, batch_size):
    model = model_ref.value
    dlrm = model.core_model
    loss_fn = model.loss_fn
    optimizer = model.optimizer

    dense_features = torch.rand(batch_size, model.float_features_count)
    kjt = generate_id_list_features(
        model.num_sprase_features,
        model.embedding_hash_size,
        batch_size,
    )

    # ## zero_grad ##
    optimizer.zero_grad()

    # ## forward ##
    logits = dlrm(dense_features, kjt)
    labels = torch.randn(batch_size, 1)
    loss = loss_fn(logits, labels)

    # ## backward ##
    loss.sum().backward()

    # ## optimizer ##
    optimizer.step()
    return loss
