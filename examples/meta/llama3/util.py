# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
# much of this is adapted from torchtitan/utils.py
import json
import logging
import os

from functools import cache
from logging import Logger
from typing import Type, Union

import torch
from llama3.config import TrainConfig

logger: Logger = logging.getLogger(__name__)


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        # pyre-ignore[16]: `torch.nn.modules.module.Module` has no attribute `tok_embeddings`
        num_params -= sum(p.numel() for p in model.tok_embeddings.parameters())
    return num_params


def get_num_flop_per_token(
    num_params: int, model_config: Union[TrainConfig, Type[TrainConfig]], seq_len: int
) -> int:
    l, h, q, t = (
        model_config.n_layer,
        model_config.n_head,
        model_config.dim // model_config.n_head,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


@cache
def get_gpu_flops() -> float:
    device_name = torch._utils._get_device_module("cuda").get_device_name()
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


def estimate_mfu(
    num_flop_per_token: int,
    batch_size: int,
    seq_len: int,
    dt: float,
) -> float:
    """Estimate model flops utilization (MFU)."""
    flops_per_fwdbwd = num_flop_per_token * seq_len
    flops_per_iter = flops_per_fwdbwd * batch_size
    # express our flops throughput as ratio of GPU bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = get_gpu_flops()
    return flops_achieved / flops_promised


def write_perf_stats_to_file(s_per_iter: Union[torch.Tensor, float]) -> None:
    # s_per_iter could be tensor of shape: torch.Size([]), or a float
    if type(s_per_iter) is torch.Tensor:
        s_per_iter_float = s_per_iter.item()
    else:
        s_per_iter_float = s_per_iter

    logger.info(f"{s_per_iter_float=}")

    perf_stats_file = os.environ.get("PERF_STATS_FILE", None)
    if perf_stats_file:
        perf_stats = {
            "time_per_iter": s_per_iter_float,
        }
        with open(perf_stats_file, "w") as f:
            json.dump(perf_stats, f)
        logger.info(f"Performance stats written to {perf_stats_file}")
    else:
        logger.info("PERF_STATS_FILE env var not set, skipping perf stats write")
