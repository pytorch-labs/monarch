from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .args import InstructArgs
from .jsonl import get_content_key, JSONLDirectoryIterator, JSONLIterator
from .text_sequence_iterator import TextSequenceIterator
from .tokenizer import BaseTokenizer
from .utils import deserialize_rng_state, serialize_rng_state


BEGIN_INST_TAG = "[INST]"
END_INST_TAG = "[/INST]"


def get_instruct_tokens(
    sample,
    slen: int,
    n_tokens: int,
    tokenizer: BaseTokenizer,
    instruct: InstructArgs,
) -> Tuple[List[int], List[bool]]:
    """
    Create instruct sequence with mask.
    """
    text = sample["text"]
    assert text.startswith(BEGIN_INST_TAG)

    # split text as instruction generation pairs
    all_utterances = text.split(BEGIN_INST_TAG)
    assert all_utterances.pop(0) == ""
    all_utterances = [BEGIN_INST_TAG + x.rstrip() for x in all_utterances]
    assert len(all_utterances) > 0

    toks: List[int] = []
    valid_ans: List[bool] = []  # 1 if token from a valid answer, 0 otherwise
    trunc_seq: List[bool] = []  # 1 if corresponding sequence is truncated, 0 otherwise

    for index, sentences in enumerate(all_utterances):
        # extract prompt / answer
        prompt, answer = sentences.split(f" {END_INST_TAG} ")
        prompt = prompt + " " + END_INST_TAG + " "

        # tokenize
        toks_src = tokenizer.encode(prompt, bos=True, eos=False)
        toks_tgt = tokenizer.encode(answer, bos=False, eos=sample.get("eos", True))

        # check length
        n_toks = len(toks_src) + len(toks_tgt)
        assert len(toks_src) > 0 and len(toks_tgt) > 0
        if n_toks > slen:  # if exceed slen
            continue

        # update tokens
        toks.extend(toks_src)
        toks.extend(toks_tgt)

        # if bad in sentence remove loss on bad answer
        wrong_answer = index in sample.get("remove_target", [])
        valid_ans.extend([False] * len(toks_src))
        valid_ans.extend([not wrong_answer] * len(toks_tgt))

        # remove loss on sample split
        is_trunc = (n_tokens % slen) + len(toks) > slen
        trunc_seq.extend([is_trunc] * n_toks)

        # sanity check
        assert len(toks) == len(valid_ans)
        assert len(toks) == len(trunc_seq)

    # mask
    mask = np.ones((len(toks),), dtype=bool)
    if instruct.no_loss_prompt:
        mask = mask & np.array(valid_ans, dtype=bool)
    if instruct.no_loss_truncated:
        mask = mask & ~np.array(trunc_seq, dtype=bool)

    assert len(toks) == len(valid_ans) == len(trunc_seq) == len(mask)
    return toks, mask.tolist()


class JSONLSequenceIterator(TextSequenceIterator):
    """
    Wraps a JSONLIterator and returns an iterator over token sequences.
    To enable reloading, we do the following steps -

    1. Keep track of the current sample of the wrapped JSONLIterator.
    2. Keep track of how many sequences we have consumed so far from this Sequence Iterator.

    During reloading -
    1. If we have picked up all sequences from the SequenceIterator, we just iterate over the next sample from JSONLIterator
        (i.e. just reload the state and proceed as usual).
    2. If we haven't picked up all sequences from the SequenceIterator, we set the JSONLIterator state to the sample we were processing.
        And then fast forward the Iterator to ignore already processed sequences.
    """

    def __init__(
        self,
        iterator: Union[JSONLIterator, JSONLDirectoryIterator],
        tokenizer: BaseTokenizer,
        slen: int,
        buffer_size: int,
        rng: np.random.RandomState,
        instruct: Optional[InstructArgs],
    ):
        super().__init__(
            iterator=iterator,
            tokenizer=tokenizer,
            slen=slen,
            buffer_size=buffer_size,
            rng=rng,
        )
        assert (
            isinstance(self.iterator, JSONLIterator)
            or isinstance(self.iterator, JSONLDirectoryIterator)
        ), "JSONLSequenceIterator is supposed to be used with JSONLIterator or JSONLDirectoryIterator"
        self.instruct = instruct
        self.content_key = get_content_key(self.iterator.get_example_file())

    def get_tokens_and_mask(self, sample):
        # pre-training data
        if self.instruct is None:
            _tokens = self.tokenizer.encode(
                sample[self.content_key], bos=True, eos=True
            )
            _mask = [True] * len(_tokens)

        # instruct data
        else:
            _tokens, _mask = get_instruct_tokens(
                sample=sample,
                slen=self.slen,
                n_tokens=len(self.tokens),
                tokenizer=self.tokenizer,
                instruct=self.instruct,
            )

        return _tokens, _mask

    def get_id(self):
        return self.iterator.get_id()

    def set_state(
        self,
        state: Dict[str, Any],
    ):
        """
        Set the state of the dataset for resuming training
        """
        self.iterator.set_position(state["position"])
        self.tokens_buffer = copy(state["tokens_buffer"])
        self.mask_buffer = copy(state["mask_buffer"])
        self.sequence_reservoir = copy(state["sequence_reservoir"])
        self.mask_reservoir = copy(state["mask_reservoir"])
        self.rng.set_state(deserialize_rng_state(state["rng_state"]))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the dataset which can be used to resume training
        """
        return {
            "id": self.get_id(),
            "position": self.iterator.get_position(),
            "tokens_buffer": copy(self.tokens_buffer),
            "mask_buffer": copy(self.mask_buffer),
            "sequence_reservoir": copy(self.sequence_reservoir),
            "mask_reservoir": copy(self.mask_reservoir),
            "rng_state": serialize_rng_state(self.rng),
        }
