# pyre-unsafe
import os
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch
from monarch import IN_PAR

from monarch.common.pipe import FakePipe, remote_generator
from monarch.worker.worker import ProcessPipe

from .config import TrainConfig


@dataclass
class DataLoaderConfig:
    data_dir: Optional[str]
    block_size: int
    batch_size: int
    device_type: str
    device: torch.device | str
    xlformers_data: Optional[str]
    xlformers_tokenizer: Optional[str]
    random_data: bool = False

    def __post_init__(self):
        assert (self.data_dir is None) == (
            self.xlformers_data is not None
        ), "Exactly one of data_dir or xlformers_data must be specified"

        if self.xlformers_data is not None:
            assert (
                self.xlformers_tokenizer is not None
            ), "xlformers_tokenizer must be specified if xlformers_data is specified"

    @staticmethod
    def from_config(config: Type[TrainConfig]):
        return DataLoaderConfig(
            data_dir=config.data_dir,
            block_size=config.block_size,
            batch_size=config.batch_size,
            # pyre-ignore[16]
            device_type=config.device_type,
            device=config.device_type,
            xlformers_data=config.xlformers_data,
            xlformers_tokenizer=config.xlformers_tokenizer,
            random_data=config.random_data,
        )


@remote_generator("llama3.data_loader.get_batch_worker", max_messages=50)
def data_loader_pipe(p: FakePipe, split: str, config: DataLoaderConfig):
    # this is a metafunction that will just return the right tensors
    # the real function will be invoked on the worker in get_batch_worker
    # Keep issuing commands to get data
    while True:
        x = torch.zeros(
            (TrainConfig.batch_size, TrainConfig.block_size),
            dtype=torch.int64,
        )
        y = torch.zeros(
            (TrainConfig.batch_size, TrainConfig.block_size),
            dtype=torch.int64,
        )
        yield x, y


def get_batch_from_file(p: ProcessPipe, split: str, config: DataLoaderConfig) -> None:
    while True:
        assert config.data_dir is not None
        data_dir: str = config.data_dir
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if config.random_data:
            data = np.random.randint(0, 65536, size=10000, dtype=np.uint16)
        elif split == "train":
            data = np.memmap(
                os.path.join(data_dir, "train.bin"),
                dtype=np.uint16,
                mode="r",
            )
        else:
            data = np.memmap(
                os.path.join(data_dir, "val.bin"),
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
                torch.from_numpy(
                    (data[i + 1 : i + 1 + config.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        p.send((x, y))


def build_training_iterator(config: DataLoaderConfig, world_size: int, rank: int):
    # Import these here so that they don't interfere with non-xlformers runs.
    # pyre-ignore
    from llama3.xlformers.args import InstructArgs, MultiJSONLIteratorArgs  # @manual

    # pyre-ignore
    from llama3.xlformers.multi_jsonl import MultiJSONLIterator  # @manual

    # pyre-ignore
    from llama3.xlformers.tokenizer import ConfStore, REGISTERED_TOKS  # @manual

    # pyre-ignore
    default_multi_iter_args = MultiJSONLIteratorArgs()
    assert config.xlformers_data is not None
    assert (
        config.xlformers_tokenizer is not None
        # pyre-ignore
        and config.xlformers_tokenizer in REGISTERED_TOKS
    )
    # pyre-ignore
    return MultiJSONLIterator(
        # pyre-ignore
        tokenizer_args=ConfStore[config.xlformers_tokenizer],
        data=str(config.xlformers_data),
        instruct_data="",
        seq_len=config.block_size,
        batch_size=config.batch_size,
        buffer_size=default_multi_iter_args.buffer_size,
        world_rank=rank,
        world_size=world_size,
        # If multiprocess is True, then the data iterator attempts to fork new processes
        # using the multiprocessing module. During creation, the child processes attempt to
        # import the supervisor module, but if running with buck (IN_PAR == True), since __main__
        # is not the worker env and does not depend on the worker env, attempting to find the
        # worker env executable fails. This can't be fixed by adding a dependency on the worker env,
        # because the worker env depends on data_loader.py.
        multiprocess=not IN_PAR,
        max_precompute=default_multi_iter_args.max_precompute,
        ignore_extra_chunks=default_multi_iter_args.ignore_extra_chunks,
        # pyre-ignore
        instruct=InstructArgs(),
        state_dump_freq=1000,
        iterate_chunk_by_chunk=default_multi_iter_args.iterate_chunk_by_chunk,
    )


def get_batch_xlformers(
    p: ProcessPipe, config: DataLoaderConfig, dp_world_size: int, dp_rank: int
) -> None:
    iterator = build_training_iterator(config, dp_world_size, dp_rank)
    for batch in iterator:
        x, y = batch.x, batch.y
        p.send((torch.from_numpy(x), torch.from_numpy(y)))


def get_batch_worker(p: ProcessPipe, split: str, config: DataLoaderConfig) -> None:
    if config.xlformers_data is not None:
        get_batch_xlformers(p, config, p.sizes["dp"], p.ranks["dp"])
    else:
        torch.manual_seed(1337 ^ p.ranks["dp"])
        get_batch_from_file(p, split, config)
