from copy import copy
from typing import List, Union

import numpy as np

from .jsonl import JSONLDirectoryIterator, JSONLIterator
from .state import TextSequenceIteratorState
from .tokenizer import BaseTokenizer
from .utils import deserialize_rng_state, serialize_rng_state


class TextSequenceIterator:
    """
    Wraps a dataset and returns an iterator over token sequences.
    The underlying iterator returns documents. This class then tokenizes
    the documents and returns a sequence of tokens of length slen.
    """

    def __init__(
        self,
        iterator: Union[JSONLIterator, JSONLDirectoryIterator],
        tokenizer: BaseTokenizer,
        slen: int,
        buffer_size: int,
        rng: np.random.RandomState,
    ):
        self.tokenizer = tokenizer
        self.iterator = iterator
        self.iter = iter(self.gen())

        # State variables
        self.tokens_buffer: List[int] = []
        self.mask_buffer: List[bool] = []
        self.sequence_reservoir = []
        self.mask_reservoir = []
        self.rng = rng

        self.buffer_size = buffer_size
        self.slen = slen
        self.n_buffer_toks = buffer_size * slen

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def get_tokens_and_mask(self, sample):
        _tokens = self.tokenizer.encode(sample, bos=True, eos=True)
        _mask = [True] * len(_tokens)

        return _tokens, _mask

    def gen(self):
        for sample in self.iterator:
            assert len(self.tokens_buffer) < self.n_buffer_toks

            # pre-training data
            _tokens, _mask = self.get_tokens_and_mask(sample)

            assert len(_tokens) == len(_mask)
            self.tokens_buffer.extend(_tokens)
            self.mask_buffer.extend(_mask)

            # Fill in the reservoirs with sequences
            while len(self.tokens_buffer) >= self.n_buffer_toks:
                x_tokens = np.array(self.tokens_buffer[: self.n_buffer_toks]).reshape(
                    self.buffer_size, self.slen
                )
                x_mask = np.array(self.mask_buffer[: self.n_buffer_toks]).reshape(
                    self.buffer_size, self.slen
                )

                self.tokens_buffer = self.tokens_buffer[self.n_buffer_toks :]
                self.mask_buffer = self.mask_buffer[self.n_buffer_toks :]

                # Shuffle the reservoir sequences in groups of self.buffer_size to retain functionality
                permutation = self.rng.permutation(self.buffer_size)
                x_tokens = x_tokens[permutation]
                x_mask = x_mask[permutation]

                seq_tokens: List[List[int]] = x_tokens.tolist()
                seq_mask: List[List[bool]] = x_mask.tolist()

                assert len(seq_tokens) == len(seq_mask) == self.buffer_size

                self.sequence_reservoir.extend(seq_tokens)
                self.mask_reservoir.extend(seq_mask)

            # Yield the reservoir sequences
            while self.sequence_reservoir and self.mask_reservoir:
                yield self.sequence_reservoir.pop(0), self.mask_reservoir.pop(0)

    def set_state(
        self,
        state: TextSequenceIteratorState,
    ):
        """
        Set the state of the dataset for resuming training
        """
        iterator_state = state.text_iterator_state
        self.iterator.set_state(iterator_state)
        self.tokens_buffer = copy(state.tokens_buffer)
        self.mask_buffer = copy(state.mask_buffer)
        self.sequence_reservoir = copy(state.sequence_reservoir)
        self.mask_reservoir = copy(state.mask_reservoir)
        self.rng.set_state(deserialize_rng_state(state.rng_state))

    def get_state(self) -> TextSequenceIteratorState:
        """
        Returns the current state of the dataset which can be used to resume training
        """
        return TextSequenceIteratorState(
            text_iterator_state=self.iterator.get_state(),
            tokens_buffer=copy(self.tokens_buffer),
            mask_buffer=copy(self.mask_buffer),
            sequence_reservoir=copy(self.sequence_reservoir),
            mask_reservoir=copy(self.mask_reservoir),
            rng_state=serialize_rng_state(self.rng),
        )
