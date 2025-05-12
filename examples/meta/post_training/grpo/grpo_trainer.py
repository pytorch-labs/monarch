import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from monarch.common import opaque_ref
from post_training.lib.executor import Executor

from .config import GRPOTrainerConfig

logger: logging.Logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, device="cuda"):
        """
        Initializes the neural network with one hidden layer.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
            hidden_dim (int, optional): The number of neurons in the hidden layer. Defaults to 128.
        """
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, output_dim, device=device)


class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, delta=0.5):
        self.delta = delta
        defaults = {"lr": lr}
        super(CustomOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                p.data.add_(group["lr"] * self.delta)
        return loss


def _build_model(params) -> torch.nn.Module:
    if params.model_type == "test":
        model = Net(
            input_dim=params.input_dim,
            output_dim=params.output_dim,
            hidden_dim=params.hidden_dim,
        )
        if params.init_param_value is not None:
            init_params(model, params.init_param_value)
    else:
        raise ValueError(f"Unknown model type: {params.model_type}")
    return model


def _build_optimizer(
    model,
    params,
) -> torch.optim.Optimizer:
    if params.model_type == "test":
        return CustomOptimizer(model.parameters(), lr=params.lr, delta=params.delta)
    else:
        raise ValueError(f"Unknown model type: {params.model_type}")


def init_params(model, x):
    for param in model.parameters():
        param.data.fill_(x)


class GRPOTrainer(Executor):
    def __init__(
        self,
        name: str,
        num_processes: int,
        params: GRPOTrainerConfig,
        reward_funcs: Optional[list[Callable]] = None,
    ):
        super().__init__(name, num_processes)
        self.num_processes = num_processes
        self.params = params
        self.model = _build_model(self.params)
        self.optimizer = _build_optimizer(self.model, self.params)
        self.reward_funcs = reward_funcs
        self.output_dict: Dict[str, Any] = {}
        self.test = params.model_type == "test"

    def step(self, step, input_dict) -> None:
        if self.test and step > 0:
            input_dict_local = {
                k: v.value if isinstance(v, opaque_ref.OpaqueRef) else v
                for k, v in input_dict.items()
            }
            logger.debug(
                f"Executor_Name: {self.name} Global_step={step} Local_step={self._state.step} {input_dict_local=}"
            )
            self.optimizer.zero_grad()
            self.optimizer.step()

            # Check the input_dict
            assert (
                "generations" in input_dict_local
            ), "Expected 'generations' in input_dict but it was not found"
            generations = input_dict_local["generations"]
            # Verify the shape and value
            expected_shape = torch.Size([2, 4])
            expected_value = step

            assert (
                generations.shape == expected_shape
            ), f"Expected generations shape {expected_shape}, got {generations.shape}"

            assert torch.allclose(
                generations,
                torch.full_like(generations, expected_value),
                atol=1e-5,
            ), f"Expected all values in generations to be {expected_value}, got {generations}"

        self._state.step += 1

    def get_model(self) -> torch.nn.Module:
        return self.model
