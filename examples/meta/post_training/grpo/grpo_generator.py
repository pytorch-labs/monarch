import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from post_training.lib.executor import Executor

from .config import GRPOGeneratorConfig
from .grpo_trainer import _build_model

logger: logging.Logger = logging.getLogger(__name__)


class GRPOGenerator(Executor):
    def __init__(
        self,
        name: str,
        num_processes: int,
        params: GRPOGeneratorConfig,
    ):
        super().__init__(name, num_processes)
        self.num_processes = num_processes
        self.params = params
        self.model = _build_model(self.params)
        self.output_dict: Dict[str, Any] = {}
        self.test = params.model_type == "test"

    def step(self, step, input_dict) -> None:
        if self.test and step > 0:
            state_dict = self.model.state_dict()
            logger.debug(
                f"Executor_Name: {self.name} Global_step={step} Local_step={self._state.step} {input_dict=} {state_dict=}"
            )
            # The trainer increments all its weights by 0.1 at each step.
            # The trainer sends weights to the generator first, then starts the next step,
            # making the generator's weights one version older.
            expected_value = (step - 1) / 10.0
            for param_name, param_tensor in state_dict.items():
                assert torch.allclose(
                    param_tensor,
                    torch.full_like(param_tensor, expected_value),
                    atol=1e-5,
                ), f"Parameter {param_name} = {param_tensor} does not match expected value {expected_value} at step {step}."

        self._state.step += 1

    def get_output(self) -> Dict[str, Any]:
        if self.test:
            self.output_dict["generations"] = torch.tensor(
                [[self._state.step] * 4, [self._state.step] * 4]
            )
            return self.output_dict
        else:
            return {}

    def get_model(self) -> nn.Module:
        return self.model
