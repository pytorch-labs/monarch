# pyre-unsafe
from time import time

import torch


def worker_load_weights(state_dict, path):
    return torch.load(path, weights_only=True)


def worker_save_checkpoint(path, model_dict, opt_dict, it, rank, extra_state=None):
    import os
    import tempfile

    os.makedirs(path, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(dir=path, mode="w", delete=False)
    fname = path + f"/rank_{rank.item()}.pth"
    checkpoint = {
        "opt": opt_dict,
        "model": model_dict,
        "iter": it,
        "extra_state": extra_state,
    }
    torch.save(checkpoint, tmp_file.name)
    os.replace(tmp_file.name, fname)
