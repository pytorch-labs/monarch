import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional, Union

from .cluster import clusterify_data_path

logger = getLogger()


@dataclass
class MultiJSONLIteratorArgs:
    buffer_size: int = 64  # read enough tokens to build `buffer_size` sequences
    multiprocess: bool = True  # enable multiprocess
    max_precompute: int = 20  # maximum number of batches to precompute
    ignore_extra_chunks: bool = True  # useful to debug (e.g. 8-GPU job with 32 chunks)
    # flag to enable iterating over datasets chunk by chunk, instead of reading in parallel from all chunks.
    # Default behaviour is to read in parallel from all chunks.
    iterate_chunk_by_chunk: bool = False


@dataclass
class TokenizerArgs:
    model: str = ""
    directory: str = ""
    tokenizer_cls: Optional[Union[str, type]] = None
    pat_str: str = ""
    num_reserved_special_tokens: int = 8
    basic_special_tokens: list = field(default_factory=list)
    extra_special_tokens: list = field(default_factory=list)

    @property
    def path(self) -> str:
        path = os.path.join(self.directory, self.model)
        return path

    def __post_init__(self):
        self.directory = clusterify_data_path(self.directory)
        _ = self.path

        self._all_special_tokens_with_ids = {}


@dataclass
class InstructArgs:
    no_loss_prompt: bool = False  # remove loss on source
    no_loss_truncated: bool = False  # remove loss on truncated examples
