from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class RngInternalState:
    key: List[int]
    pos: int


@dataclass
class NumpyRandomNumberGeneratorState:
    bit_generator: str
    state: RngInternalState
    has_gauss: int
    gauss: float

    def __post_init__(self):
        if isinstance(self.state, dict):
            self.state = RngInternalState(**self.state)


@dataclass
class AIRStoreDatasetState:
    processed_sample_count: int
    current_epoch: int
    dataloader_world_size: int


@dataclass
class TextSequenceIteratorState:
    text_iterator_state: Union[AIRStoreDatasetState, Optional[int]]
    tokens_buffer: List[int]
    mask_buffer: List[bool]
    sequence_reservoir: List[List[int]]
    mask_reservoir: List[List[bool]]
    rng_state: NumpyRandomNumberGeneratorState


@dataclass
class TextAIRStoreCompositeDatasetMixerAggregatedState:
    combined_rng_state: NumpyRandomNumberGeneratorState
    sequence_iterator_states: Dict[str, TextSequenceIteratorState]


def deserialize_text_airstore_composite_dataset_mixer_aggregated_state(
    state: Dict[str, Any],
) -> TextAIRStoreCompositeDatasetMixerAggregatedState:
    combined_rng_state = NumpyRandomNumberGeneratorState(**state["combined_rng_state"])
    converted_states = {}

    for (
        dataset_name,
        dataset_sequence_iterator_state,
    ) in state["sequence_iterator_states"].items():
        if isinstance(dataset_sequence_iterator_state, dict):
            converted_states[dataset_name] = TextSequenceIteratorState(
                text_iterator_state=AIRStoreDatasetState(
                    **dataset_sequence_iterator_state["text_iterator_state"]
                ),
                tokens_buffer=dataset_sequence_iterator_state["tokens_buffer"],
                mask_buffer=dataset_sequence_iterator_state["mask_buffer"],
                sequence_reservoir=dataset_sequence_iterator_state[
                    "sequence_reservoir"
                ],
                mask_reservoir=dataset_sequence_iterator_state["mask_reservoir"],
                rng_state=NumpyRandomNumberGeneratorState(
                    **dataset_sequence_iterator_state["rng_state"]
                ),
            )
        else:
            assert isinstance(
                dataset_sequence_iterator_state, TextSequenceIteratorState
            )
            converted_states[dataset_name] = dataset_sequence_iterator_state

    return TextAIRStoreCompositeDatasetMixerAggregatedState(
        combined_rng_state=combined_rng_state,
        sequence_iterator_states=converted_states,
    )
