# pyre-unsafe
from typing import Any, Optional

import torch

class GradientGenerator:
    def __init__(
        self,
        roots_list: Any,
        with_respect_to: Any,
        grad_roots: Any,
        context_restorer: Any,
    ): ...
    # pyre-ignore[11]: Annotation `torch.Tensor` is not defined as a type.
    def __next__(self) -> Optional[torch.Tensor]: ...
    def __iter__(self) -> "GradientGenerator": ...
