"""An example that demonstrates how to use ExecutionTimer with a Monarch program.

Run this with
buck run //monarch/python/monarch/timer:example_monarch

"""
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-unsafe

import logging

import torch

from monarch import inspect, remote
from monarch.rust_local_mesh import local_mesh

logger = logging.getLogger(__name__)


execution_timer_start = remote(
    "monarch.timer.remote_execution_timer.execution_timer_start", propagate="inspect"
)

execution_timer_stop = remote(
    "monarch.timer.remote_execution_timer.execution_timer_stop", propagate="inspect"
)

get_execution_timer_average_ms = remote(
    "monarch.timer.remote_execution_timer.get_execution_timer_average_ms",
    propagate=lambda: torch.tensor(0.0, dtype=torch.float64),
)

get_time_perfcounter = remote(
    "monarch.timer.remote_execution_timer.get_time_perfcounter",
    propagate=lambda: torch.tensor(0.0, dtype=torch.float64),
)


def main() -> None:
    with local_mesh(hosts=1, gpus_per_host=1) as mesh:
        with mesh.activate():
            num_iterations = 5

            a = torch.randn(1000, 1000, device="cuda")
            b = torch.randn(1000, 1000, device="cuda")
            torch.matmul(a, b)

            total_dt = torch.zeros(1, dtype=torch.float64)

            for _ in range(num_iterations):
                t0 = get_time_perfcounter()
                torch.matmul(a, b)
                total_dt += get_time_perfcounter() - t0

            for _ in range(num_iterations):
                execution_timer_start()
                torch.matmul(a, b)
                execution_timer_stop()

            cuda_average_ms = get_execution_timer_average_ms()
            local_total_dt = inspect(total_dt)
            local_cuda_avg_ms = inspect(cuda_average_ms)

        local_total_dt = local_total_dt.item()
        local_cuda_avg_ms = local_cuda_avg_ms.item()
        mesh.exit()
    avg_perfcounter_ms = local_total_dt / num_iterations * 1000
    print(f"average time w/ perfcounter: {avg_perfcounter_ms:.4f} (ms)")
    print(f"average time w/ ExecutionTimer:   {local_cuda_avg_ms:.4f} (ms)")


if __name__ == "__main__":
    main()
