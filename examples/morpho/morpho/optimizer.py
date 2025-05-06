# pyre-unsafe
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import monarch

import torch
from torch.optim.adam import adam

Step = torch.Tensor

# before we know how the parameters will be arranged in distributed training,
# we still need to specify

# which parameters we optimize: this is done by listing the fully qualified names
# of the parameters we will optimize. Helper method just registers all model parameters for common case.

# what optimization algorithm to use for each parameter: this is specified when adding parameters,
# each call to add_parameters has an associated Algorithm object where the algorithm and its hyperparameters are speified,'

# What hyperparameters are used for each timestep: things like Learning rate schedules


# the above needs to happen _before_ we choose parallelization for parameters and their placement.


class Algorithm(Protocol):
    @abstractmethod
    def new_state(self, parameter: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def update_(
        self,
        step: Step,  # step lets algorithms take something like a learning rate schedule and turn it into a learning rate.
        states: Sequence[Any],
        parameters: Sequence[torch.Tensor],
        gradients: Sequence[torch.Tensor],
    ):
        pass


class Optimizer:
    def __init__(self):
        self.algorithms = []
        self.algorithms_index: Dict[str, int] = {}
        self.states: Dict[str, Any] = {}

    def add_parameters(self, parameters: Iterable[str], algorithm: Algorithm):
        for param in parameters:
            self.algorithms_index[param] = len(self.algorithms)
        self.algorithms.append(algorithm)

    def add_model(self, model: torch.nn.Module, algorithm: Algorithm):
        self.add_parameters([name for name, _ in model.named_parameters()], algorithm)

    # create new state (e.g. for a new run)
    def initialize_state(self, parameters_values: Iterable[Tuple[str, torch.Tensor]]):
        for name, value in parameters_values:
            algo_index = self.algorithms_index[name]
            self.states[name] = self.algorithms[algo_index].new_state(value)

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(self.algorithms_index.keys())

    def step(
        self,
        step: Step,
        parameters: Iterable[Tuple[str, torch.Tensor]],
        gradients: Iterable[torch.Tensor],
    ):
        algos = defaultdict(lambda: ([], [], []))
        for (name, param), grad in zip(parameters, gradients):
            state = self.states[name]
            states, params, grads = algos[self.algorithms_index[name]]
            params.append(param)
            states.append(state)
            grads.append(grad)

        for idx, (states, params, grads) in algos.items():
            algorithm = self.algorithms[idx]
            algorithm.update_(step, states, params, grads)

    def state_dict(self):
        return {"states": self.states, "algorithms_index": self.algorithms_index}

    def load_state_dict(self, d: Dict[str, Any]):
        self.states = d["states"]
        self.algorithms_index = d["algorithms_index"]
        for index in self.algorithms_index.values():
            if index >= len(self.algorithms):
                raise ValueError("Algorithm index out of range")


# Callable takes the step (as a tensor on the parameter's device),
# and returns the hyperparameter as a singleton tensor.
HyperParam = Union[float, Callable[[Step], torch.Tensor]]


def resolve_hyperparam(
    step: torch.Tensor, param: HyperParam
) -> Union[float, torch.Tensor]:
    if callable(param):
        return param(step)
    return param


class AdamW:
    @dataclass
    class State:
        exp_avg: torch.Tensor
        exp_avg_sq: torch.Tensor
        max_exp_avg_sq: Optional[torch.Tensor]

    def __init__(
        self,
        lr: HyperParam = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def new_state(self, parameter: torch.Tensor) -> State:
        return self.State(
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
            max_exp_avg_sq=torch.zeros_like(parameter) if self.amsgrad else None,
        )

    def update_(
        self,
        step: Step,
        states: Sequence[State],
        parameters: Sequence[torch.Tensor],
        gradients: Sequence[torch.Tensor],
    ):
        if not parameters:
            return
        lr = resolve_hyperparam(step, self.lr)
        exp_avgs = [state.exp_avg for state in states]
        exp_avg_sqs = [state.exp_avg_sq for state in states]
        max_exp_avg_sqs: List[torch.Tensor] = [
            state.max_exp_avg_sq for state in states if state.max_exp_avg_sq is not None
        ]
        # because these will be mutated we need on tensor for each.
        # for performance we should probably just use a modified version of the fusible
        # adamw anyway where state is the appropriate tensor.
        steplist = step.expand(len(states)).to(parameters[0].device, copy=True)
        steps = [steplist[i] for i, _ in enumerate(states)]

        remote_adam(
            params=parameters,  # type: ignore
            grads=gradients,  # type: ignore
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=max_exp_avg_sqs,
            state_steps=steps,
            amsgrad=self.amsgrad,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
            maximize=False,
            decoupled_weight_decay=True,
            differentiable=False,
            capturable=False,
            has_complex=False,
            grad_scale=None,
            found_inf=None,
            foreach=True,
        )


def _remote_atom(*, params, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, **kwargs):
    def mutate(xs):
        for x in xs:
            x += 1

    mutate(params)
    mutate(exp_avgs)
    mutate(exp_avg_sqs)
    mutate(max_exp_avg_sqs)


# we have to do this wrapper because directly calling adam
# with a lr tensor is not the bitwise same result.
@monarch.remote(propagate=_remote_atom)
def remote_adam(*, lr: torch.Tensor, **kwargs):
    if isinstance(lr, torch.Tensor):
        # XXX: we can't do this in monarch if we want easy compilation,
        # but it is required to match torch titan bitwise accuracy.
        # for now we will put the entire implementation of this thingy into a udf.
        lr = lr.item()

    return adam(lr=lr, **kwargs)
