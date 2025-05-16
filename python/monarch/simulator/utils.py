# pyre-unsafe
import os

import numpy as np


def file_path_with_iter(file_path: str, iter_count: int) -> str:
    dir_path = os.path.dirname(file_path)
    file_name, file_postfix = os.path.basename(file_path).split(".")
    file_name = f"{file_name}_{iter_count}.{file_postfix}"
    return os.path.join(dir_path, file_name)


def compress_workers_range(workers) -> str:
    regions = []
    start = workers[0]
    end = workers[0]
    sorted_workers = np.sort(workers)
    for i in range(1, len(sorted_workers)):
        if workers[i] == end + 1:
            end = workers[i]
        else:
            regions.append(f"[{start}-{end}]")
            start = workers[i]
            end = workers[i]
    regions.append(f"[{start}-{end}]")
    return " ".join(regions)


def clean_name(name: str) -> str:
    if name.startswith("torch.ops.aten."):
        name = name[len("torch.ops.") :]  # noqa: whitespace before ':'
    if name.endswith(".default"):
        name = name[: -len(".default")]
    return name
