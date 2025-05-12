# pyre-unsafe
from typing import Tuple

import torch


@torch.no_grad()
def clip_grad_norm_(
    grads: Tuple[torch.Tensor, ...],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    dims: Tuple[str, ...] = (),
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, True
    )
    total_norm.reduce_(dims, "sum")
    clip_grads_with_norm_(grads, max_norm, total_norm)
    return total_norm


# torch.util... version of this takes the parameters rather than the grads for some reason.
def clip_grads_with_norm_(
    grads: Tuple[torch.Tensor, ...],
    max_norm: float,
    total_norm: torch.Tensor,
) -> None:
    r"""Scale the gradients of an iterable of parameters given a pre-calculated total norm and desired max norm.

    The gradients will be scaled by the following calculation

    .. math::
        grad = grad * \frac{max\_norm}{total\_norm + 1e-6}

    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.clip_grad_norm_` with a pre-calculated
    total norm.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        total_norm (Tensor): total norm of the gradients to use for clipping
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        None
    """
    max_norm = float(max_norm)
    if len(grads) == 0:
        return

    clip_coef = max_norm / (total_norm + 1e-6)  # type: ignore
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)  # type: ignore
    torch._foreach_mul_(grads, clip_coef_clamped)
