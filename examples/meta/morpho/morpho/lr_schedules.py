# pyre-unsafe
import torch


def linear_warmup_linear_decay(
    lr: float, warmup_steps: int, decay_steps: int, current_step: torch.Tensor
) -> torch.Tensor:
    """Computes linear warmup followed by linear decay using torch operations.
    Returns a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    # Calculate linear warmup adjustment
    warmup_adjustment = (current_step + 1) / (warmup_steps + 1)
    # Calculate linear decay adjustment
    normalized_step = decay_steps - (current_step - warmup_steps)
    decay_adjustment = 1 - (decay_steps - normalized_step) / decay_steps
    # Use torch.where to select between warmup and decay adjustments
    curr_adjustment = torch.where(
        current_step < warmup_steps, warmup_adjustment, decay_adjustment
    )
    return lr * curr_adjustment
