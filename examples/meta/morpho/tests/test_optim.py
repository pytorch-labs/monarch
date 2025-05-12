from typing import Generator

import pytest
import torch
from morpho.optimizer import AdamW, Optimizer
from torch import nn
from torch.optim import AdamW as TorchAdamW


# until we have better remote scalar support we need tensor range
# so that step can be passed
def tensorrange(*args, **kwargs) -> Generator[torch.Tensor, None, None]:
    the_range = range(*args, **kwargs)
    i_t = torch.full((), the_range.start, dtype=torch.double)
    for _ in the_range:
        yield i_t
        i_t += the_range.step


@pytest.mark.parametrize("amsgrad", [True, False])
def test_adamw(amsgrad: bool):
    # Define a simple model with two parameters
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.param1 = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
            self.param2 = nn.Parameter(torch.tensor([3.0, 4.0], requires_grad=True))

        def forward(self, x):
            return self.param1 * x + self.param2

    # Initialize the model
    model = SimpleModel()
    # Create a copy of the model for the custom optimizer
    model_custom = SimpleModel()
    model_custom.param1.data = model.param1.data.clone()
    model_custom.param2.data = model.param2.data.clone()
    # Define the loss function
    criterion = nn.MSELoss()
    # Input and target
    x = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 3.0])
    # Initialize PyTorch's AdamW optimizer
    optimizer_torch = TorchAdamW(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=amsgrad,
    )
    # Initialize custom AdamW optimizer
    optimizer = Optimizer()
    optimizer.add_model(
        model_custom,
        AdamW(
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=amsgrad,
        ),
    )
    optimizer.initialize_state(model.named_parameters())
    # Perform a few optimization steps

    for step in range(10):
        # Zero gradients
        optimizer_torch.zero_grad()
        model_custom.zero_grad()
        # Forward pass
        output_torch = model(x)
        output_custom = model_custom(x)
        # Compute loss
        loss_torch = criterion(output_torch, target)
        loss_custom = criterion(output_custom, target)
        # Backward pass
        loss_torch.backward()
        loss_custom.backward()
        # Update parameters using PyTorch's AdamW
        optimizer_torch.step()
        with torch.no_grad():
            # Update parameters using custom AdamW
            optimizer.step(
                torch.tensor(step, dtype=torch.double),
                model_custom.named_parameters(),
                [p.grad for p in model_custom.parameters() if p.grad is not None],
            )
        # Compare parameters
        for p_torch, p_custom in zip(model.parameters(), model_custom.parameters()):
            assert torch.allclose(
                p_torch, p_custom, atol=0, rtol=0
            ), f"Parameters diverged at step {step}"


def test_example_optimizer():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.param1 = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
            self.param2 = nn.Parameter(torch.tensor([3.0, 4.0], requires_grad=True))

        def forward(self, x):
            return self.param1 * x + self.param2

    model = SimpleModel()
    criterion = nn.MSELoss()
    x = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 3.0])

    # optimizer object can have multiple different optimization
    # algorithms entirely, which lets you use different hyperparameters
    # or just different 'Algorithm' objects. The Algorithm objects
    # implement any optimizer that works per parameter. If you
    # have some optimizer that is global, then it will use the same
    # interface as this optimizer for training, but have its own implementation.
    optimizer = Optimizer()

    max_lr = torch.tensor(0.01)

    def warmup_rate(step, warmup_steps=4):
        warmup_lr = max_lr * (step / warmup_steps)
        learning_rate = torch.where(step < warmup_steps, warmup_lr, max_lr)
        return learning_rate

    # common case is to just add one algorithm for all parameters
    # in the model, but you can also configure differently for each parameter.
    # we name parameters with their state_dict name since this is pretty good
    # universal name in the torch world.
    optimizer.add_parameters(
        (name for name, _ in model.named_parameters()),
        # The algorithm object specifies how to (1) create a new state given a parameter,
        # (2) update that state given a parameter and a gradient
        AdamW(
            lr=warmup_rate,  # HyperParam can be either a float, or a Callable[[step], torch.Tensor]
            beta1=0.9,  # to allow for schedules.
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
        ),
    )

    # The intent is that the above call are made _before_ passing the optimizer and model into
    # a 'Trainer' object. The API calls below are intended for the Trainer object to call.
    # In particular the trainer will be the one apply parallelisms and choosing shardings.
    # So we cannot have initialized the optimizer state before now.

    # state initialization is explicit, and can be done for subsets of parameters.
    # this lets the trainer first load, and partitition the parameters via any sharding scheme
    # before asking the optimizer to initialize (or load) state for them.
    optimizer.initialize_state(model.named_parameters())

    # The organization of the optimizer, model, and trainer estabishes clear ownership:
    #    the trainer creates then owns the gradients
    #    the optimizer owns the optimizer state
    #    the model owns the parameters (but sharding schemes like FSDP might
    #     steal them and replace them with temprary local ones, we are limited by the exist torch.nn.Module API)
    # The other objects explicitly avoid holding references to things they do not own to remove
    # surprise 'action at a distance' interactions.

    for step in tensorrange(10):
        output = model(x)
        loss = criterion(output, target)

        gradients = torch.autograd.grad(loss, model.parameters())
        with torch.no_grad():
            # step the optimizer. You explicitly pass the step number,
            # and also specify which parameters to update. This allows
            # updating some parameters at different times.
            # notice that because the optimizer doesn't have a reference to the
            # parameters, they must be explicitly passed.
            optimizer.step(
                step,
                model.named_parameters(),
                gradients,
            )
