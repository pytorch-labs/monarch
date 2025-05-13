import importlib
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from monarch.common import opaque_ref
from monarch.opaque_object import opaque_method, OpaqueObject

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class State:
    step: int = 0


@dataclass
class Executor:
    """
    The `Executor` class serves as the main entry point for a model in the online learning framework.
    For example, we can define the policy model training as an training executor and completions generating as a generator executor.
    Please refer to ../examples for model details.
    Args:
        name: The name of the executor which will be used in defining the communications among different executors.
        num_processes: The number of GPU processes to run the executor.
        dump_dir: The directory to dump the metrics and checkpoints.
    """

    name: str
    num_processes: int
    dump_dir: Optional[str] = None
    _ranks: Optional[List[int]] = None
    _state: State = State()
    cfg: Optional[Dict[str, Any]] = None

    def init(self):
        """
        Any torch distributed related setup and initialization should be done here.
        """
        pass

    def step_with_validation(
        self, step, input_dict, profiler: Optional[torch.profiler.profile] = None
    ) -> None:
        # TODO: add profiler support for specific executor
        if self.validate_input(input_dict):
            self.step(step, input_dict)
        else:
            logger.warning(f"Skipping {step=}")

    def validate_input(self, input_dict) -> bool:
        """
        Add any validation logic here to control whether to run the step function or not.
        """
        pass

    def step(self, step, input_dict) -> None:
        """
        The actual logic of a single training step.
        Args:
            step: The current step in the training process.
            input_dict: A dictionary containing the input data.
        """
        pass

    def maybe_save_checkpoint(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_output(self) -> Dict[str, Any]:
        pass

    def get_model(self) -> torch.nn.Module:
        """
        The logic to get the model which will be used for weights update if specified.
        """
        pass

    def get_current_step(self) -> int:
        return self._state.step

    def is_master(self) -> bool:
        # global_rank = get_global_rank()
        # TODO: fix this
        global_rank = 0
        return (
            True
            if self._ranks is not None
            and len(self._ranks) > 0
            and global_rank == self._ranks[0]
            else False
        )

    def post_training_step(self) -> None:
        """To be called after all steps are completed."""
        pass

    def get_step(self) -> torch.Tensor:
        return torch.tensor(self._state.step, dtype=torch.int64)

    def get_model_output_dict_key_to_shape(self) -> torch.Tensor:
        output_dict = self.get_output()
        _model_state_dict_key_to_shape = {
            k: v.shape for k, v in output_dict.items() if isinstance(v, torch.Tensor)
        }
        serialized_shape_map = pickle.dumps(_model_state_dict_key_to_shape)  # noqa: PYTHONPICKLEISBAD
        shape_tensor = torch.tensor(list(serialized_shape_map), dtype=torch.uint8)
        return shape_tensor

    def get_output_tensor(self, name, shape) -> torch.Tensor:
        output_dict = self.get_output()
        if name not in output_dict:
            return torch.tensor([])
        return output_dict[name]

    def get_model_state_dict_key_to_shape(self) -> torch.Tensor:
        model = self.get_model()
        if model is None:
            return torch.tensor([])
        state_dict = model.state_dict()
        _model_state_dict_key_to_shape = {
            k: v.shape for k, v in state_dict.items() if isinstance(v, torch.Tensor)
        }
        serialized_shape_map = pickle.dumps(_model_state_dict_key_to_shape)  # noqa: PYTHONPICKLEISBAD
        shape_tensor = torch.tensor(list(serialized_shape_map), dtype=torch.uint8)
        return shape_tensor

    def get_model_weight(self, name, shape) -> torch.Tensor:
        model = self.get_model()
        if model is None:
            return torch.tensor([])
        state_dict = model.state_dict()
        if name not in state_dict:
            return torch.tensor([])
        return state_dict[name]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        model = self.get_model()
        state_dict_local = {
            k: v.value if isinstance(v, opaque_ref.OpaqueRef) else v
            for k, v in state_dict.items()
        }
        logger.debug(f"{self.name}: model loading state dict {state_dict_local=}")
        model.load_state_dict(state_dict_local)


def log_ref(ref):
    logger.info("ref: " + str(ref.value))


class MonarchExecutor(OpaqueObject):
    @opaque_method
    def init(self):
        pass

    @opaque_method
    def step(self, step, input_dict):
        return torch.tensor(1.0, dtype=torch.float)

    @opaque_method
    def get_output(self):
        return opaque_ref.OpaqueRef({})
        # return torch.tensor(0, dtype=torch.int64)

    @opaque_method
    def get_step(self):
        return torch.tensor(0, dtype=torch.int64)

    @opaque_method
    def get_output_tensor(self, name, shape) -> torch.Tensor:
        # TODO: Verify the return shape is accurate
        # The output is a Monarch Tensor. The return output should not
        # require an output buffer size equal to the shape of the tensor.
        return torch.rand(shape)

    @opaque_method
    def get_model_weight(self, name, shape) -> torch.Tensor:
        # TODO: Verify the return shape is accurate
        # The output is a Monarch Tensor. The return output should not
        # require an output buffer size equal to the shape of the tensor.
        return torch.rand(shape)

    @opaque_method
    def get_model(self):
        return opaque_ref.OpaqueRef(None)

    @opaque_method
    def load_state_dict(self, state_dict) -> torch.Tensor:
        return torch.tensor(1.0, dtype=torch.float)


class MonarchExecutorWrapper:
    def __init__(self, path_to_object, *args, **kwargs) -> None:
        module_name, class_name = path_to_object.rsplit(".", 1)
        _class = getattr(importlib.import_module(module_name), class_name)
        self.executor = _class(*args, **kwargs)

    def init(self):
        self.executor.init()

    def step(self, step, input_dict):
        if type(input_dict) is opaque_ref.OpaqueRef:
            input = input_dict.value
        else:
            input = input_dict
        self.executor.step(step, input)
        return torch.tensor(1.0, dtype=torch.float)

    def get_output(self) -> opaque_ref.OpaqueRef:
        return self.executor.get_output()

    def get_model(self) -> opaque_ref.OpaqueRef:
        return self.executor.get_model()

    def get_step(self) -> torch.Tensor:
        return self.executor.get_step()

    def get_model_state_dict_key_to_shape(self) -> torch.Tensor:
        return self.executor.get_model_state_dict_key_to_shape()

    def get_model_output_dict_key_to_shape(self) -> torch.Tensor:
        return self.executor.get_model_output_dict_key_to_shape()

    def get_model_weight(self, name, shape) -> torch.Tensor:
        return self.executor.get_model_weight(name, shape)

    def get_output_tensor(self, name, shape) -> torch.Tensor:
        return self.executor.get_output_tensor(name, shape)

    def load_state_dict(self, state_dict) -> torch.Tensor:
        self.executor.load_state_dict(state_dict)
        return torch.tensor(1.0, dtype=torch.float)

    def get_output_state_dict_key_to_shape(self) -> torch.Tensor:
        return self.executor.get_model_output_dict_key_to_shape()


def make_monarch_executor(path_to_object, *args, **kwargs):
    return MonarchExecutor(
        "post_training.lib.executor.MonarchExecutorWrapper",
        path_to_object,
        *args,
        **kwargs,
    )
