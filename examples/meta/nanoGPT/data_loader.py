# pyre-unsafe
import os
from dataclasses import dataclass
from typing import Tuple, Type

import numpy as np
import torch
from monarch.common.pipe import FakePipe, remote_generator
from monarch.worker.worker import ProcessPipe

from .config import NanoGPTConfig


@dataclass
class DataLoaderConfig:
    data_dir: str
    block_size: int
    batch_size: int
    device_type: str
    device: torch.device | str
    random_data: bool = False

    @staticmethod
    def from_config(config: Type[NanoGPTConfig]):
        return DataLoaderConfig(
            # pyre-ignore[16]
            data_dir=config.data_dir,
            block_size=config.block_size,
            batch_size=config.batch_size,
            # pyre-ignore[16]
            device_type=config.device_type,
            device=config.device_type,
            random_data=config.random_data,
        )


@remote_generator("nanoGPT.data_loader.get_batch_local", max_messages=50)
def data_loader_pipe(p: FakePipe, split: str, config: DataLoaderConfig):
    # this is a metafunction that will just return the right tensors
    # the real function will be invoked on the worker in get_batch_local
    # Keep issuing commands to get data
    while True:
        x = torch.zeros(
            (NanoGPTConfig.batch_size, NanoGPTConfig.block_size),
            dtype=torch.int64,
        )
        y = torch.zeros(
            (NanoGPTConfig.batch_size, NanoGPTConfig.block_size),
            dtype=torch.int64,
        )
        yield x, y


def _get_batch_local(
    split: str, config: DataLoaderConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    if config.random_data:
        data = np.random.randint(0, 65536, size=10000, dtype=np.uint16)
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    elif split == "train":
        data = np.memmap(
            os.path.join(config.data_dir, "train.bin"),
            dtype=np.uint16,
            mode="r",
        )
    else:
        data = np.memmap(
            os.path.join(config.data_dir, "val.bin"),
            dtype=np.uint16,
            mode="r",
        )
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((data[i : i + config.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + config.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    return x, y


def get_batch_local_no_pipe(
    split: str, config: DataLoaderConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Only for use in train.py. We don't want to use pipe abstraction but do want to re-use logic
    """
    x, y = _get_batch_local(split, config)
    if config.device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(config.device, non_blocking=True),
            y.pin_memory().to(config.device, non_blocking=True),
        )
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y


def get_batch_local(p: ProcessPipe, split: str, config: DataLoaderConfig) -> None:
    while True:
        x, y = _get_batch_local(split, config)
        p.send((x, y))
