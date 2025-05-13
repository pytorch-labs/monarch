# pyre-unsafe
from typing import Callable, TypeVar

import torch

State = TypeVar("State")


def _get_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state())


def _set_state(state):
    cpu, device = state
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(device)


def _new_state(seed: int):
    orig = _get_state()
    torch.manual_seed(seed)
    mine = _get_state()
    _set_state(orig)
    return mine


class RandomContext:
    """
    A reusable context manager that starts with the rng set to  the given state,
    and when it is active generates numbers along the sequence of seed.

    Random numbers outside of this context are generated as normal.

    This is intended to provide a separate stream of random numbers
    needed for a separate concern while not affecting the builtin random number generator.

    It is written with state, set and get left abstract since pretty much each training library
    will handle the random number state slightly differently.
    """

    def __init__(
        self,
        state: State,
        get_state: Callable[[], State],
        set_state: Callable[[State], None],
    ):
        self.saved_state = state
        self._get_state = get_state
        self._set_state = set_state

    def __enter__(self):
        self._swap_state()

    def __exit__(self, typ, value, traceback):
        self._swap_state()

    def _swap_state(self):
        self.saved_state, oldstate = self._get_state(), self.saved_state
        self._set_state(oldstate)


RandomContextCreate = Callable[[int], "RandomContext"]


def new_random_context(seed: int) -> RandomContext:
    return RandomContext(_new_state(seed), _get_state, _set_state)
