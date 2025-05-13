from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from .state import TextAIRStoreCompositeDatasetMixerAggregatedState


@dataclass
class Batch:
    """
    aggregated_state_map contains a mapping from data_parallel_rank to the aggregated state
    """

    x: np.ndarray
    y: np.ndarray
    aggregated_state_map: Optional[
        Dict[int, TextAIRStoreCompositeDatasetMixerAggregatedState]
    ] = None
    mask: Optional[np.ndarray] = None
    src_names: Optional[List[str]] = None

    def __post_init__(self):
        assert self.x.ndim == 2
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert self.mask is None or self.mask.shape == self.x.shape
        assert self.src_names is None or len(self.src_names) == len(self.x)


class DataIterator:
    @abstractmethod
    def __iter__(self) -> Iterator[Batch]: ...

    @abstractmethod
    def get_position(
        self,
    ) -> Optional[Union[List[int], List[Dict[str, Any]], Dict[str, Any]]]: ...

    @abstractmethod
    def set_position(
        self, position: Optional[Union[List[int], List[Dict[str, Any]], Dict[str, Any]]]
    ): ...

    def get_buffer_size(self) -> int:
        return 0

    def close(self):
        pass
