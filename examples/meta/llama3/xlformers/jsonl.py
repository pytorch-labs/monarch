import itertools
import json
import os
import re
from logging import getLogger
from typing import Any, cast, Dict, Iterator, List, Optional

import numpy as np

from .iterator_base import Batch
from .tokenizer import BaseTokenizer


BEGIN_INST_TAG = "[INST]"
END_INST_TAG = "[/INST]"


logger = getLogger()


def serialize_rng_state(rng: np.random.RandomState) -> Dict[str, Any]:
    # RNG state is a dictionary containing the following fields:
    # the string ‘MT19937’.
    # a 1-D array of 624 unsigned integer keys. (stored in ['state']['key'])
    # an integer pos.
    # an integer has_gauss.
    # a float cached_gaussian.
    # Before calling set_state, we need to convert the list of 624 integer keys to a numpy array.

    rng_state = rng.get_state(legacy=False)
    rng_internal_state: Dict[str, Any] = rng_state["state"]
    assert rng_internal_state["key"].dtype == np.uint32
    rng_internal_state["key"] = rng_internal_state["key"].tolist()
    rng_state["state"] = rng_internal_state

    return rng_state


def deserialize_rng_state(rng_state: Dict[str, Any]) -> Dict[str, Any]:
    rng_internal_state: Dict[str, Any] = rng_state["state"]
    rng_internal_state["key"] = np.array(rng_internal_state["key"], dtype=np.uint32)
    rng_state["state"] = rng_internal_state

    return rng_state


def get_content_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip()
        try:
            x = json.loads(line)
        except UnicodeDecodeError as e:
            print(f"Error when trying to decode '{line}': {str(e)}")
            raise
        for k in ["text", "content"]:
            if k in x:
                return k
        raise RuntimeError(f"Unable to determine key for {path}")


class JSONLIterator:
    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.f = open(fpath, "r", encoding="utf-8")
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank
        self.line_num = 0
        self.iter = iter(self.gen(infinite))
        self.iter_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self, infinite: bool) -> Iterator[Dict]:
        while True:
            logger.info(f"Starting iteration {self.iter_id} over {self.fpath} ...")
            self.iter_id += 1
            while True:
                line, self.line_num = self.f.readline(), self.line_num + 1
                if not line:
                    break
                if (self.line_num - 1) % self.world_size == self.world_rank:
                    yield json.loads(line)
            if not infinite:
                break
            self.set_position(None)
        self.f.close()

    def set_position(self, position: Optional[int]):
        logger.warning(
            f"Setting JSONL position on {self.fpath} "
            f"({self.world_rank}/{self.world_size}): {position}"
        )
        if position is None:
            self.f.seek(0)
            self.line_num = 0
        else:
            assert isinstance(position, int)
            self.f.seek(position)
            self.line_num = (
                self.world_rank + 1
            )  # restore value of line_num (modulo world_size)

    def get_position(self) -> Optional[int]:
        file_pos = self.f.tell()
        if file_pos == 0 and self.line_num == 0:
            return None
        assert (self.line_num - 1) % self.world_size == self.world_rank
        return file_pos

    def get_example_file(self):
        """
        Return the path to a sample file to infer the content key
        """
        return self.fpath

    def get_id(self):
        """
        Return an identifier for the dataset this iterator represents
        """
        return self.fpath


class JSONLDirectoryIterator:
    """
    The JSONLDirectoryIterator is a data wrapper around a dataset folder, which contains
    multiple JSONL files. Internally, it reuses the JSONLIterator class to iterate through
    each individual file, and then wraps onto the next file once the current one is exhausted.

    Once all files in the directory have been iterated over, we wrap back to the first file
    ( if infinite is true ).

    This enables us to iterate over a dataset one chunk at a time.

    Also, note that we open the next chunk file on an ondemand basis, which means that we can
    modify chunks mid training as well to add more data, fix issues, etc.
    """

    def __init__(
        self,
        dirpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.dirpath = dirpath
        self.world_size = world_size
        self.world_rank = world_rank

        fnames = [
            x
            for x in os.listdir(self.dirpath)
            if re.fullmatch(r".*chunk\.\d+.*\.jsonl", x)
        ]
        self.fpaths = [os.path.join(self.dirpath, fname) for fname in sorted(fnames)]
        assert (
            len(self.fpaths) > 0
        ), f"Specified dataset location {self.dirpath} is empty."

        # Generator for cycling through the list of files
        if infinite:
            self.fpaths_generator = cast(Iterator[str], itertools.cycle(self.fpaths))
        else:
            self.fpaths_generator = cast(Iterator[str], iter(self.fpaths))

        self.iter = iter(self.gen(infinite))
        self.jsonl_iterator: Optional[JSONLIterator] = None

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self, infinite: bool) -> Iterator[Dict]:
        # Handle the case when we're reloading from a saved state.
        if self.jsonl_iterator is not None:
            yield from self.jsonl_iterator

        for fpath in self.fpaths_generator:
            # Note that we set infinite to false here, because JSONLDirectoryIterator would take care of infinite looping
            self.jsonl_iterator = JSONLIterator(
                fpath,
                world_size=self.world_size,
                world_rank=self.world_rank,
                infinite=False,
            )

            yield from self.jsonl_iterator

    def set_position(self, state: Dict[str, Any]):
        logger.warning(
            f"Setting JSONL position on {self.dirpath} "
            f"({self.world_rank}/{self.world_size}): {state}"
        )
        fpath: Optional[str] = state["fpath"]
        position: Optional[int] = state["position"]
        if fpath is None or position is None:
            return

        assert isinstance(fpath, str)
        assert isinstance(position, int)

        # Fast forward the generator
        for fpath_candidate in self.fpaths_generator:
            if fpath_candidate == fpath:
                break

        # Create the JSONL iterator and set it's position appropriately
        self.jsonl_iterator = JSONLIterator(
            fpath,
            world_size=self.world_size,
            world_rank=self.world_rank,
            infinite=False,
        )
        self.jsonl_iterator.set_position(position)

    def get_position(self):
        if self.jsonl_iterator is None:
            return {
                "fpath": None,
                "position": None,
            }
        return {
            "fpath": self.jsonl_iterator.fpath,
            "position": self.jsonl_iterator.get_position(),
        }

    def get_example_file(self):
        """
        Return the path to a sample file to infer the content key
        """
        return self.fpaths[0]

    def get_id(self):
        """
        Return an identifier for the dataset this iterator represents
        """
        return self.dirpath


def batch_iterator(
    jsonl_iterator: JSONLIterator,
    tokenizer: BaseTokenizer,
    seq_len: int,
    batch_size: int,
    buffer_size: int,
) -> Iterator[Batch]:
    """
    Take as input a JSONLIterator and return an iterator of batches.
    """
    content_key = get_content_key(jsonl_iterator.fpath)
    n_buffer_toks = (1 + buffer_size * seq_len) * batch_size
    tokens: List[int] = []
    for sample in jsonl_iterator:
        assert len(tokens) < n_buffer_toks
        toks = tokenizer.encode(sample[content_key], bos=True, eos=True)
        tokens.extend(toks)
        while len(tokens) >= n_buffer_toks:
            x = np.array(tokens[:n_buffer_toks]).reshape(batch_size, -1)
            tokens = tokens[n_buffer_toks:]
            assert x.shape[1] == 1 + buffer_size * seq_len
            assert x.shape[1] // seq_len == buffer_size
            for i in range(x.shape[1] // seq_len):
                a, b = i * seq_len, (i + 1) * seq_len
                yield Batch(x=x[:, a:b], y=x[:, a + 1 : b + 1])
