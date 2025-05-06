"""Common monarch related abstractions for RL."""

import contextlib
import logging
from typing import Dict, Generator, List, Optional, Tuple

# Required for OpaqueRef
import monarch.common.tree  # noqa: F401
import torch
from monarch import remote

from monarch.common.device_mesh import DeviceMesh
from monarch.common.opaque_ref import OpaqueRef
from monarch.rust_local_mesh import local_mesh, LoggingLocation, SocketType
from rl.config import Config
from rl.data import Trajectory
from rl.model import create_model

_ref_config: Optional[Config] = None
logger: logging.Logger = logging.getLogger(__name__)


set_worker_random_seed = remote(
    "monarch.worker.worker.set_random_seed_impl", propagate="inspect"
)

timer_start = remote(
    "monarch.timer.execution_timer.execution_timer_start", propagate="inspect"
)

timer_stop = remote(
    "monarch.timer.execution_timer.execution_timer_stop", propagate="inspect"
)

get_latest_time = remote(
    "monarch.timer.execution_timer.get_latest_timer_measurement",
    propagate=lambda: torch.tensor(0.0, dtype=torch.float64),
)


@contextlib.contextmanager
def create_mesh(
    mesh_type: str, num_hosts: int, gpus_per_host: int
) -> Generator[DeviceMesh, None, None]:
    logger.info(
        "Creating mesh (%s) with %d hosts and %d GPUs per host",
        mesh_type,
        num_hosts,
        gpus_per_host,
    )
    match mesh_type:
        # only local mesh is supported for now
        case "local":
            with local_mesh(
                hosts=num_hosts,
                gpus_per_host=gpus_per_host,
                logging_location=LoggingLocation.DEFAULT,
                socket_type=SocketType.UNIX,
            ) as mesh:
                yield mesh
        case _:
            raise ValueError(f"Unknown mesh type: {mesh_type}")


def set_global_config(config: Config) -> None:
    """Sets global variables for correct monarch tensor propagation."""
    global _ref_config
    _ref_config = config


# ---- learner
create_learner = remote("rl.learner.create", propagate=lambda x: OpaqueRef(None))


def _get_learner_state_dict(ref: OpaqueRef) -> Dict[str, torch.Tensor]:
    # the model state dict is bogus, but we need it to pass
    # the correct model keys and tensor shapes.
    assert _ref_config is not None, "Must call set_global_config() first"
    model = create_model(_ref_config)
    return {k: torch.zeros_like(v) for k, v in model.state_dict().items()}


get_learner_state_dict = remote(
    "rl.learner.get_state_dict", propagate=_get_learner_state_dict
)

learner_step = remote(
    "rl.learner.step",
    propagate=lambda ref, trajectories: (
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    ),
)


# ---- generator


create_generator = remote("rl.generator.create", propagate=lambda x: OpaqueRef(None))

load_generator_state_dict = remote("rl.generator.load_state_dict", propagate="inspect")


def _generate(ref: OpaqueRef, prompt: torch.Tensor) -> Trajectory:
    assert _ref_config is not None, "Must call _set_global_config() first"
    prompt_length, response_length, input_shape = (
        _ref_config.prompt_length,
        _ref_config.response_length,
        _ref_config.input_shape,
    )
    return (
        torch.zeros(
            (
                prompt_length + response_length,
                input_shape,
            ),
            dtype=torch.float32,
        ),
        torch.zeros((response_length, input_shape), dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
    )


generate = remote("rl.generator.generate", propagate=_generate)


class RemoteLearner:
    """A representation of a learner as an actor."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ref = create_learner(config)

    def step(self, trajectories: List[Trajectory]) -> Tuple[torch.Tensor, torch.Tensor]:
        return learner_step(self.ref, trajectories)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return get_learner_state_dict(self.ref)


class RemoteGenerator:
    """A representation of a generator as an actor."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ref = create_generator(config)

    def generate(self, prompt: torch.Tensor) -> Trajectory:
        return generate(self.ref, prompt)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        load_generator_state_dict(self.ref, state_dict)
