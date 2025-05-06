# pyre-unsafe
import math
import pickle
import tempfile
from builtins import ValueError
from contextlib import contextmanager

from typing import Any, Dict

import torch

from morpho.config import DebugConfig
from morpho.debug.observer import Observer
from morpho.random_context import RandomContext, RandomContextCreate


@contextmanager
def inject_debug_behavior(conf: DebugConfig, randomness_source: RandomContextCreate):
    if not conf.observe:
        yield None, None
        return

    observer = Observer(conf.run, conf.variant, conf.sample)
    outputfile = tempfile.NamedTemporaryFile("wb", delete=False)
    filename = outputfile.name
    randomness = randomness_source(hash((conf.run, conf.variant, conf.sample)))
    match conf.variant:
        case "reference":
            yield observer, filename
        case "experiment":
            yield observer, filename
        case "noise":
            import torch
            import torch.nn.functional as F
            from monarch.gradient_generator import grad_function
            from torch.nn import Linear

            @grad_function
            def nextafter(input: torch.Tensor):
                with randomness:
                    direction = (torch.randint_like(input, 2) * 2 - 1) * math.inf
                return torch.nextafter(input, direction), lambda do: do

            original_forward = Linear.forward
            try:

                def forward(self, input: torch.Tensor) -> torch.Tensor:
                    result = original_forward(self, input)
                    return nextafter(result)

                Linear.forward = forward
                yield observer, filename
            finally:
                Linear.forward = original_forward

        case "reassociated":
            import torch
            import torch.nn.functional as F
            from torch.nn import Linear

            original_forward = Linear.forward
            try:

                def forward(self, input: torch.Tensor) -> torch.Tensor:
                    with randomness:
                        perm = torch.randperm(self.in_features)
                    a = F.linear(input[..., perm], self.weight[:, perm], self.bias)
                    return a

                Linear.forward = forward
                yield observer, filename
            finally:
                Linear.forward = original_forward
        case _:
            raise ValueError(f"Unknown variant {conf.variant}")

    with outputfile:
        pickle.dump(observer.run, outputfile)
