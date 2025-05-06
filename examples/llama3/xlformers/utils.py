from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .state import NumpyRandomNumberGeneratorState


def serialize_rng_state(rng: np.random.RandomState) -> NumpyRandomNumberGeneratorState:
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

    return NumpyRandomNumberGeneratorState(**rng_state)


def deserialize_rng_state(rng_state: NumpyRandomNumberGeneratorState) -> Dict[str, Any]:
    rng_state: Dict[str, Any] = asdict(rng_state)
    rng_internal_state: Dict[str, Any] = rng_state["state"]
    rng_internal_state["key"] = np.array(rng_internal_state["key"], dtype=np.uint32)
    rng_state["state"] = rng_internal_state

    return rng_state


def merge_seq_masks(
    batch_size: int, seq_len: int, mask_seqs: List[List[bool]]
) -> Optional[np.ndarray]:
    """
    Merge a list of sequence masks (corresponding to a data batch) into a single 2D mask.
    """
    assert len(mask_seqs) == batch_size
    assert all(len(m) == seq_len + 1 for m in mask_seqs)
    if all(all(m) for m in mask_seqs):
        # All masks are all True, so no need to merge.
        return None
    mask = np.ones((batch_size, seq_len), dtype=bool)
    for i, m in enumerate(mask_seqs):
        if m is not None:
            mask[i] = mask_seqs[i][1:]
    return mask
