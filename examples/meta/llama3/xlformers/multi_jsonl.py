import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from queue import Empty, Full
from typing import Any, cast, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from .args import InstructArgs, TokenizerArgs
from .cluster import clusterify_data_path
from .iterator_base import Batch, DataIterator
from .jsonl import (
    deserialize_rng_state,
    JSONLDirectoryIterator,
    JSONLIterator,
    serialize_rng_state,
)
from .jsonl_sequence_iterator import JSONLSequenceIterator
from .tokenizer import BaseTokenizer, build_tokenizer
from .utils import merge_seq_masks


logger = getLogger()


Position = List[Optional[int]]


def _combine_seq_iterators(
    seq_iterators: List[JSONLSequenceIterator],
    src_names: List[str],
    weights: np.ndarray,
    seq_len: int,
    batch_size: int,
    rng: np.random.RandomState,
) -> Iterator[Batch]:
    assert len(seq_iterators) == len(src_names) == len(weights)
    while True:
        tokens: List[List[int]] = []
        masks: List[List[bool]] = []
        srcs: List[str] = []
        for _ in range(batch_size):
            src_id = rng.choice(len(weights), p=weights)
            _tokens, _mask = next(seq_iterators[src_id])
            assert len(_tokens) == len(_mask) == seq_len + 1
            tokens.append(_tokens)
            masks.append(_mask)
            srcs.append(src_names[src_id])
        x_tokens = np.array(tokens)
        assert x_tokens.shape == (batch_size, seq_len + 1)
        yield Batch(
            x=x_tokens[:, :-1],
            y=x_tokens[:, 1:],
            mask=merge_seq_masks(batch_size, seq_len, masks),
            src_names=srcs,
        )


@dataclass
class DataAssignmentBase:
    path: str
    rank: int  # rank among workers on this file
    size: int  # number of workers on this file
    weight: float  # weight

    @property
    def name(self) -> str:
        if os.path.isdir(self.path):
            return Path(self.path).name
        else:
            return Path(self.path).parent.name

    def __post_init__(self):
        assert 0 <= self.rank < self.size
        assert self.weight > 0


@dataclass
class DataFileAssignment(DataAssignmentBase):
    def __post_init__(self):
        super().__post_init__()
        assert self.path.endswith(".jsonl"), self.path
        assert os.path.isfile(self.path), self.path


@dataclass
class DataDirectoryAssignment(DataAssignmentBase):
    def __post_init__(self):
        super().__post_init__()
        self.path = self.path.strip()
        assert os.path.isdir(self.path), self.path
        assert (
            len(os.listdir(self.path)) > 0
        ), f"Specified directory {self.path} does not contain any files."


def _assign_data(
    path: str, world_size: int, ignore_extra: bool
) -> List[Tuple[str, int, int]]:
    """
    Given a directory, list .jsonl files, and assign one to a worker.
    """
    path = path.strip()
    assert os.path.isdir(path), path
    fnames = [x for x in os.listdir(path) if re.fullmatch(r".*chunk\.\d+.*\.jsonl", x)]
    if ignore_extra and len(fnames) > world_size:
        logger.warning(f"Removing {len(fnames) - world_size} extra chunks for {path}")
        fnames = fnames[:world_size]
    fpaths = [os.path.join(path, fname) for fname in sorted(fnames)]
    assert world_size % len(fpaths) == 0, (world_size, len(fpaths), path)
    n = world_size // len(fpaths)  # number of workers on the same file
    res = []
    for path in fpaths:
        for i in range(n):
            res.append((path, i, n))
    assert len(res) == world_size
    return res


def _get_data_assignment(
    data: str,
    world_rank: int,
    world_size: int,
    ignore_extra: bool,
    iterate_chunk_by_chunk: bool,
) -> List[DataAssignmentBase]:
    """
    `data` can be either of the form:
        - wiki
    or
        - wiki:2,ccnet:10,github:5
    In the second case, data weights can be arbitrary float values.
    """
    assert len(data) > 0
    assert 0 <= world_rank < world_size

    directories_to_process: List[Tuple[str, ...]] = []

    # same folder for all workers
    if "," not in data:
        dirpath = clusterify_data_path(data)
        directories_to_process.append((dirpath, "1.0"))
    else:
        # otherwise, one folder per dataset
        seen: Set[str] = set()
        for x in data.split(","):
            path, weight = x.split(":")
            path = clusterify_data_path(path)
            assert path not in seen
            assert re.fullmatch(r"\d+(\.\d*)?", weight) and float(weight) > 0
            seen.add(path)
            directories_to_process.append((path, weight))

    assignment: List[DataAssignmentBase] = []
    for dirpath, weight in directories_to_process:
        if iterate_chunk_by_chunk:
            assignment.append(
                DataDirectoryAssignment(dirpath, world_rank, world_size, float(weight))
            )
        else:
            fpath, rank, size = _assign_data(dirpath, world_size, ignore_extra)[
                world_rank
            ]
            assignment.append(DataFileAssignment(fpath, rank, size, float(weight)))

    assert len(assignment) == len(data.split(","))
    return assignment


class MultiJSONLIterator(DataIterator):
    def __init__(
        self,
        data: str,
        instruct_data: str,
        seq_len: int,
        batch_size: int,
        buffer_size: int,
        world_rank: int,
        world_size: int,
        multiprocess: bool,
        max_precompute: int,
        ignore_extra_chunks: bool,
        instruct: InstructArgs,
        state_dump_freq: int,
        iterate_chunk_by_chunk: bool,
        tokenizer: Optional[BaseTokenizer] = None,
        tokenizer_args: Optional[TokenizerArgs] = None,
    ):
        # tokenizer and args
        if multiprocess:
            assert (
                tokenizer_args is not None
            ), "Must provide tokenizer arguments when using multiprocessing instead of tokenizer itself."

        assert (tokenizer is not None) ^ (
            tokenizer_args is not None
        ), "Either tokenizer or tokenizer_args should be provided."

        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args

        # main parameters
        self.data = data
        self.instruct_data = instruct_data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.world_rank = world_rank
        self.world_size = world_size
        self.multiprocess = multiprocess
        self.max_precompute = max_precompute
        self.ignore_extra_chunks = ignore_extra_chunks
        assert data or instruct_data
        assert 0 <= world_rank < world_size
        self.state_counter = 0
        self.state_dump_freq = state_dump_freq
        self.iterate_chunk_by_chunk = iterate_chunk_by_chunk

        # data assigned to worker  / verify paths
        self.data_assignment = _get_data_assignment(
            data=",".join([x for x in [data, instruct_data] if x]),
            world_rank=world_rank,
            world_size=world_size,
            ignore_extra=ignore_extra_chunks,
            iterate_chunk_by_chunk=self.iterate_chunk_by_chunk,
        )
        logger.info(
            f"Starting iteration on {str(self.data_assignment)} "
            f"({world_rank}/{world_size}) ..."
        )

        self.src_names = [x.name for x in self.data_assignment]
        assert len(self.src_names) == len(set(self.src_names)), self.src_names

        # check what is pretraining and instruct data
        self.n_pretrain = len([x for x in data.split(",") if x])
        self.n_instruct = len([x for x in instruct_data.split(",") if x])
        self.is_instruct = [False] * self.n_pretrain + [True] * self.n_instruct
        self.instruct = instruct

        # data source weights
        self.weights = np.array(
            [x.weight for x in self.data_assignment], dtype=np.float64
        )
        self.weights = self.weights / self.weights.sum()
        assert (
            abs(self.weights.sum() - 1) < 1e-6 and min(self.weights) > 0
        ), self.weights
        logger.info(f"Data source weights: {self.weights}")

        # random state for combined iterator
        self.combined_rng = np.random.RandomState((self.world_rank, self.world_size))

        # multiprocessing
        self.batch_queue: Optional[mp.Queue] = None
        self.state_queue: Optional[mp.Queue] = None
        self.mp_position: Optional[List[dict]] = None
        self.process: Optional[mp.process.BaseProcess] = None
        self.stop: Optional[mp.synchronize.Event] = None
        self.position: Optional[Union[List[int], List[Dict[str, Any]]]] = None

    def _init_multi_process(self) -> None:
        logger.info("Initializing multi process ...")
        assert self.multiprocess
        assert self.process is None
        ctx = mp.get_context("forkserver")
        self.stop = ctx.Event()
        self.batch_queue = ctx.Queue(maxsize=self.max_precompute)
        self.state_queue = ctx.Queue(maxsize=self.max_precompute)
        self.process = ctx.Process(
            name="iterator_multi",
            target=self._multiprocess_iterator,
        )
        assert self.process is not None
        self.process.start()

    def build_iterator(self) -> Iterator[Batch]:
        # Create iterators
        if self.iterate_chunk_by_chunk:
            self.iterators: List[Union[JSONLIterator, JSONLDirectoryIterator]] = [
                JSONLDirectoryIterator(
                    dirpath=x.path,
                    world_rank=x.rank,
                    world_size=x.size,
                    infinite=True,
                )
                for x in self.data_assignment
            ]
        else:
            self.iterators = [
                JSONLIterator(
                    fpath=x.path,
                    world_rank=x.rank,
                    world_size=x.size,
                    infinite=True,
                )
                for x in self.data_assignment
            ]
        assert len(self.iterators) == self.n_pretrain + self.n_instruct

        if self.tokenizer_args is not None:
            assert self.tokenizer is None
            self.tokenizer = build_tokenizer(self.tokenizer_args)

        # sequence iterators
        self.seq_iterators: List[JSONLSequenceIterator] = [
            JSONLSequenceIterator(
                iterator=iterator,
                tokenizer=self.tokenizer,  # type: ignore
                slen=self.seq_len + 1,  # +1 for input/output 1-shift
                buffer_size=self.buffer_size,
                rng=np.random.RandomState((self.world_rank, self.world_size)),
                instruct=self.instruct if inst else None,
            )
            for inst, iterator in zip(self.is_instruct, self.iterators)
        ]

        if self.position is not None:
            logger.warning(
                f"Setting JSONL position on {self.data_assignment} "
                f"({self.world_rank}/{self.world_size})"
            )
            self._init_position(self.position, self.seq_iterators, self.iterators)

        return _combine_seq_iterators(
            seq_iterators=self.seq_iterators,
            src_names=self.src_names,
            weights=self.weights,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            rng=self.combined_rng,
        )

    def get_buffer_size(self) -> int:
        return self.batch_queue.qsize()

    def _init_position(
        self,
        position: Union[List[int], List[Dict[str, Any]]],
        seq_iterators: List[JSONLSequenceIterator],
        iterators: List[Union[JSONLIterator, JSONLDirectoryIterator]],
    ) -> None:
        # Check if the position contains a full state of the dataloader
        if all([isinstance(elem, dict) for elem in position]):
            # We have a snapshot of the entire dataloader state
            position = cast(List[Dict[str, Any]], position)
            self._set_full_state(position)
        elif all([isinstance(elem, int) for elem in position]):
            assert len(position) == len(iterators)
            position = cast(List[int], position)

            # In this case, we are sure that the iterators would be of type JSONLIterator
            for x, pos in zip(iterators, position):
                x = cast(JSONLIterator, x)
                x.set_position(pos)  # `pos` can be int or None
        else:
            raise ValueError(
                "position must be either list of integers or a list of dictionaries."
            )

    def _multiprocess_iterator(self) -> None:
        iterator = self.build_iterator()
        assert (
            self.batch_queue is not None
            and self.stop is not None
            and self.state_queue is not None
        )
        try:
            batch: Optional[Batch] = None
            while not self.stop.is_set():
                if batch is None:
                    batch = next(iterator)
                    position = self._get_position()
                    self.state_counter += 1
                try:
                    self.batch_queue.put(batch, timeout=1)
                    if self.state_counter % self.state_dump_freq == 0:
                        self.state_queue.put(position, timeout=1)
                        self.state_counter = 0
                    batch = None
                except Full:
                    pass
        finally:
            self.stop.set()

    def multiprocess_iterator_loop(self) -> Iterator[Batch]:
        assert (
            self.batch_queue is not None
            and self.stop is not None
            and self.state_queue is not None
        )
        try:
            while not self.stop.is_set():
                try:
                    batch = self.batch_queue.get(timeout=1)
                    self.state_counter += 1
                    if self.state_counter % self.state_dump_freq == 0:
                        position = self.state_queue.get(timeout=1)
                        self.mp_position = position
                        self.state_counter = 0
                    yield batch
                except Empty:
                    pass
        finally:
            self.stop.set()

    def __iter__(self) -> Iterator[Batch]:
        if not self.multiprocess:
            return self.build_iterator()
        else:
            self._init_multi_process()
            return self.multiprocess_iterator_loop()

    def set_position(
        self,
        position: Optional[Union[List[int], List[Dict[str, Any]]]],
    ):
        assert self.process is None  # if multiprocessing, position must be set before

        if position is None:
            return

        assert type(position) is list
        self.position = position

    def _set_full_state(self, states: List[Dict[str, Any]]):
        multi_jsonl_state = states.pop()
        self.combined_rng.set_state(
            deserialize_rng_state(multi_jsonl_state["combined_rng_state"])
        )

        id_key_present = all("id" in _ for _ in states)
        if id_key_present:
            # Match back the sequence iterators to their respective states
            iterator_states = {}
            for state in states:
                iterator_states[state["id"]] = state

            for it in self.seq_iterators:
                if it.get_id() in iterator_states:
                    it.set_state(iterator_states[it.get_id()])
                    iterator_states.pop(it.get_id())
                else:
                    # A new dataset was added. Log this.
                    logger.warning(
                        f"A new dataset was added during reloading - {it.get_id()}"
                    )
            if len(iterator_states) > 0:
                logger.warning(
                    f"The following datasets were removed during reloading: {list(iterator_states.keys())}"
                )
        else:
            assert len(states) == len(
                self.seq_iterators
            ), "The number of states does not match the number of datasets!"
            for state, it in zip(states, self.seq_iterators):
                it.set_state(state)

    def _get_position(self) -> List[Dict[str, Any]]:
        return [it.get_state() for it in self.seq_iterators] + [
            {
                "combined_rng_state": serialize_rng_state(self.combined_rng),
            }
        ]

    def get_position(self) -> Optional[Union[List[int], List[Dict[str, Any]]]]:
        if self.multiprocess:
            assert self.state_counter == 0
            return self.mp_position
        else:
            return self._get_position()

    def close(self):
        if self.process is not None:
            print(f"Attempting to close process nicely (I'm process {os.getpid()})")
            if self.stop is not None:
                self.stop.set()
            p = self.process
            p.join(timeout=5)
            if p.exitcode is None:
                print(f"Killing data process {p.pid} ...")
                p.kill()
            else:
                print(f"Data process {p.pid} exited with code {p.exitcode}")
