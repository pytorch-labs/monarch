#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for measuring message throughput in Monarch actor mesh.
"""

import asyncio
import time
from typing import Any, Dict

from monarch.actor import Actor, endpoint, proc_mesh

from windtunnel.benchmarks.python_benchmark_runner.benchmark import (
    main,
    register_benchmark,
    UserCounters,
    UserMetric,
)

FILE_PATH: str = "monarch/python/benches/actor_mesh_benchmark.py"


class SleepActor(Actor):
    @endpoint
    async def sleep(self, sleep_secs: float, _: bytes) -> None:
        await asyncio.sleep(sleep_secs)


async def run_actor_scaling_benchmark(
    actor_mesh: Any,
    message_size: int,
    duration_seconds: int = 10,
    sleep_secs: float = 0.1,
) -> Dict[str, float]:
    """
    Run a benchmark with a specific number of actors and message size.
    Returns statistics about the benchmark run including:
    - avg_time_ms: average time per iteration in milliseconds
    - median_time_ms: median time per iteration in milliseconds
    - min_time_ms: minimum time per iteration in milliseconds
    - max_time_ms: maximum time per iteration in milliseconds
    - throughput_mbps: throughput in megabits per second
    - iterations: number of iterations completed
    """
    payload = bytes(message_size)
    times = []

    start_benchmark = time.time()
    iteration_count = 0

    while time.time() - start_benchmark < duration_seconds:
        start_time = time.time()
        await actor_mesh.sleep.call(sleep_secs, payload)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        iteration_count += 1

    if iteration_count == 0:
        raise ValueError("No iterations completed")

    times_ms = [t * 1000 for t in times]
    avg_time_ms = sum(times_ms) / iteration_count
    sorted_times = sorted(times_ms)
    median_time_ms = (
        sorted_times[iteration_count // 2]
        if iteration_count % 2 == 1
        else (
            sorted_times[iteration_count // 2 - 1] + sorted_times[iteration_count // 2]
        )
        / 2
    )

    return {
        "avg_time_ms": avg_time_ms,
        "median_time_ms": median_time_ms,
        "min_time_ms": min(times_ms),
        "max_time_ms": max(times_ms),
        "throughput_mbps": (message_size * 8) / (avg_time_ms / 1000) / 1_000_000,
        "iterations": iteration_count,
    }


@register_benchmark(FILE_PATH, use_counters=True)
async def bench_actor_scaling(counters: UserCounters) -> None:
    """
    Benchmark how long it takes to process 1KB message on different numbers of actors.
    Reports average, median, min, and max times.
    """
    host_counts = [1, 10, 100]
    message_sizes = [1024]
    duration_seconds = 10

    for host_count in host_counts:
        for message_size in message_sizes:
            mesh = await proc_mesh(hosts=host_count)
            await mesh.logging_option(stream_to_client=False, aggregate_window_sec=None)
            actor_mesh = await mesh.spawn("actor", SleepActor)
            # Allow Actor init to finish
            await asyncio.sleep(1)

            stats = await run_actor_scaling_benchmark(
                actor_mesh, message_size, duration_seconds, sleep_secs=0.1
            )
            await mesh.stop()

            counters[f"actor_count_{host_count}_median_ms"] = UserMetric(
                value=int(stats["median_time_ms"])
            )
            counters[f"actor_count_{host_count}_min_ms"] = UserMetric(
                value=int(stats["min_time_ms"])
            )
            counters[f"actor_count_{host_count}_max_ms"] = UserMetric(
                value=int(stats["max_time_ms"])
            )


if __name__ == "__main__":
    asyncio.run(main())
