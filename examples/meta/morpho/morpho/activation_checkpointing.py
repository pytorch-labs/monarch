# pyre-unsafe
import logging

import torch

from torchtitan.parallelisms.parallelize_llama import ptd_checkpoint_wrapper

logger = logging.getLogger(__name__)


def _apply_ac_to_transformer_block(module: torch.nn.Module, ac_freq):
    # Checkpoint every `ac_freq` of the modules passed to this function
    ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
    ptd_checkpoint_wrapper._count += 1
    if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
    else:
        return module


def apply_ac(model, ac_freq: int):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(transformer_block, ac_freq)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Applied selective activation checkpointing to the model")
