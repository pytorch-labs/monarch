# pyre-unsafe
from typing import Union

import torch
from rl.config import Config


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_shape: int, output_shape: int, dim: int, depth: int = 3):
        super().__init__()
        layers = [torch.nn.Linear(input_shape, dim), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([torch.nn.Linear(dim, dim), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(dim, output_shape))
        self.model = torch.nn.Sequential(*layers)

    def __call__(self, inputs: Union[torch.Tensor, int]):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32, device="cuda")
        return self.model(inputs)


def create_model(config: Config) -> torch.nn.Module:
    return SimpleMLP(
        input_shape=config.input_shape,
        output_shape=config.output_shape,
        dim=config.model_dim,
        depth=config.model_depth,
    )
