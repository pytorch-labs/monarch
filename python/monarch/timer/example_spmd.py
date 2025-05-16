"""An example that demonstrates how to use ExecutionTimer in a SPMD style program.

Run this with:
buck run //monarch/python/monarch/timer:example_spmd
"""

import time

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import torch
from monarch.timer import ExecutionTimer


def main() -> None:
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    device = torch.device("cuda")

    num_iterations = 5

    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # Warmup
    torch.matmul(a, b)
    torch.cuda.synchronize()

    cpu_timings = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        cpu_timings.append(time.perf_counter() - t0)

    for _ in range(num_iterations):
        with ExecutionTimer.time("matrix_multiply"):
            torch.matmul(a, b)

    mean_cuda_ms = ExecutionTimer.summary()["matrix_multiply"]["mean_ms"]
    mean_perfcounter_ms = sum(cpu_timings) / len(cpu_timings) * 1000
    print("mean perf counter times: ", mean_perfcounter_ms)
    print("mean cuda times: ", mean_cuda_ms)


if __name__ == "__main__":
    main()
