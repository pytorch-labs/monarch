"""Generator instances, modeled as a UDF."""

import logging
from typing import Dict

# Required for OpaqueRef
import monarch.common.tree  # noqa: F401

import torch
from monarch.common.opaque_ref import OpaqueRef
from rl import model
from rl.config import Config
from rl.data import Trajectory


class Generator:
    def __init__(self, config: Config):
        self.model = model.create_model(config).to("cuda")
        self.config = config

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state_dict)

    def generate(self, prompt: torch.Tensor) -> Trajectory:
        # TODO - replace this with a data loader
        response_logits = []
        prompt = prompt.to("cuda")

        for _ in range(self.config.response_length):
            with torch.no_grad():
                response_logits.append(self.model(prompt)[-1])

        # in this case, response is a sequence of logits
        # In practice we would want a tokenized version of response,
        # but it doesn't really matter here
        query_and_response = torch.empty(
            (
                self.config.prompt_length + self.config.response_length,
                self.config.input_shape,
            ),
            dtype=torch.float32,
            device="cuda",
        )
        response = torch.stack(response_logits)
        query_and_response[: self.config.prompt_length,] = prompt
        query_and_response[self.config.prompt_length :,] = response
        reward = torch.sum(query_and_response)
        return query_and_response, response, reward


def create(config: Config) -> OpaqueRef:
    return OpaqueRef(Generator(config))


def load_state_dict(ref: OpaqueRef, state_dict: Dict[str, torch.Tensor]) -> None:
    ref.value.load_state_dict(state_dict)


def generate(ref: OpaqueRef, prompt: torch.Tensor) -> Trajectory:
    return ref.value.generate(prompt)
