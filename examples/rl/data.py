from typing import Tuple

import torch

Trajectory = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]  # [query_and_response, response, reward]
