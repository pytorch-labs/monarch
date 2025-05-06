# pyre-unsafe
import itertools
from functools import partial
from typing import Any, Callable, Optional, Tuple

import monarch

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from morpho.config import TrainingConfig
from torchtitan.datasets.tokenizer import TikTokenizer

Cursor = torch.Tensor


# this will be where we can put the pipe for sharded data loading
class Dataloader:
    def __init__(
        self,
        tokenizer_path: str,
        dataset_ctor: Callable[[], Any],
        seq_len: int,
        batch_size: int,
    ):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = TikTokenizer(self.tokenizer_path)
        self.dataset_ctor = dataset_ctor
        self.seq_len = seq_len
        self.batch_size = batch_size

    @property
    def n_words(self):
        return self.tokenizer.n_words

    def generate(self, starting_cursor: Optional[Cursor]):
        pipe = monarch.create_pipe(
            _dataloader_pipe,
            self.dataset_ctor,
            self.seq_len,
            self.batch_size,
            self.tokenizer_path,
            max_messages=3,
        )
        if starting_cursor is None:
            starting_cursor = torch.zeros(2, dtype=torch.int64)
        pipe.send(starting_cursor)
        while True:
            batch, cursor = pipe.recv()
            yield batch[:, :-1], batch[:, 1:], cursor


def _dataloader_pipe_propgate(
    pipe,
    dataset_ctor: Callable[[], Any],
    seq_len: int,
    batch_size: int,
    tokenizer_path: str,
):
    pipe.recv()
    while True:
        yield (
            torch.empty((batch_size, seq_len + 1), dtype=torch.int64),
            torch.empty((2,), dtype=torch.int64),
        )


@monarch.remote(propagate=_dataloader_pipe_propgate)
def _dataloader_pipe(
    pipe,
    dataset_ctor: Callable[[], Any],
    seq_len: int,
    batch_size: int,
    tokenizer_path: str,
):
    rank = pipe.ranks["dp"]
    size = pipe.sizes["dp"]
    tokenizer = TikTokenizer(tokenizer_path)
    dataset = split_dataset_by_node(dataset_ctor(), rank, size)  # type: ignore
    skip_start, offset = pipe.recv()
    all_tokens = batch_size * (seq_len + 1)
    iterator = iter(dataset.skip(skip_start))
    new_tokens = tokenizer.encode(next(iterator)["text"], bos=True, eos=True)
    tokens = new_tokens[offset:]
    for skip in itertools.count(skip_start):
        while len(tokens) >= all_tokens:
            result = torch.tensor(tokens[:all_tokens], dtype=torch.int64)
            tokens = tokens[all_tokens:]
            batch = result.view(batch_size, seq_len + 1)

            # the cursor is where to restart from to resume right after this run
            cursor = torch.tensor([skip, len(new_tokens) - len(tokens)])
            # each time we create a batch, we also yield the cursor.
            # This way we can always restart the dataloader at the right spot,
            # regardless of any layers of data buffering between the the loader
            # and the actual code consuming it.
            pipe.send((batch, cursor))
        new_tokens = tokenizer.encode(next(iterator)["text"], bos=True, eos=True)
        tokens.extend(new_tokens)


def create_dataloader(training: TrainingConfig, tokenizer_path: str):
    # build dataloader
    datasets = {
        "c4": partial(
            load_dataset, "allenai/c4", name="en", split="train", streaming=True
        ),
        "c4_test": partial(load_dataset, training.dataset_path, split="train"),
    }
    dataset_ctor = datasets[training.dataset]
    return Dataloader(
        tokenizer_path, dataset_ctor, training.seq_len, training.batch_size
    )
