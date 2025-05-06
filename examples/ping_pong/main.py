#!/usr/bin/env python3
"""
Main execution logic for the Monarch ping_pong example.
"""

import asyncio
import importlib.resources
import itertools
import json
import os
import signal
import tempfile
import time
import uuid
from asyncio.subprocess import Process
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import psutil

from libfb.py.build_info import BuildInfo

from monarch.examples.ping_pong.cli import parse_args
from monarch.examples.ping_pong.goldenfile import (
    BuildMode,
    display_comparison_tables,
    display_goldenfiles_table,
    GoldenFile,
    LogLevel,
    TracingLayer,
)
from monarch.examples.ping_pong.process_management import (
    run_command_with_output,
    wait_for_sync_point,
)


def get_grandparent_process() -> int:
    """Get the parent process ID of the parent process."""
    parent: psutil.Process = psutil.Process(os.getppid())
    return parent.ppid()


RUNNERS = asyncio.Semaphore(2)


def retry_n_times(n):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if i >= n:
                        raise e
                    await asyncio.sleep(1)

        return wrapper

    return decorator


async def handle_workers(procs: Sequence[tuple[Process, Path]]) -> list[float]:
    procs = list(procs)
    procs.reverse()
    # restart the process that is currently sigstopped
    for p, _ in procs:
        p.send_signal(signal.SIGCONT)
        await asyncio.sleep(0.5)

    codes = await asyncio.gather(*[p.wait() for p, _ in procs])
    if any(c > 0 for c in codes):
        raise RuntimeError("Worker failed")

    reports = [json.loads(path.read_text()) for _, path in procs]

    return [float(r["duration"]) / float(r["iterations"]) for r in reports]


async def start_pingpong_worker(
    n_messages: int,
    output_dir: str,
    worker_name: str,
    env: Dict[str, str],
    pingpong_worker_bin: str,
    hyperactor_bootstrap_addr: str,
) -> tuple[Process, Path]:
    worker_file = worker_name.replace("[", ".").replace("]", ".")

    stdoutput_file = f"{output_dir}/{worker_file}_logs.txt"
    report_file = f"{output_dir}/{worker_file}_report.json"

    pingpong_worker_cmd = [
        pingpong_worker_bin,
        "-i",
        str(n_messages),
        "-p",
        worker_name,
        "-a",
        hyperactor_bootstrap_addr,
        "-o",
        report_file,
    ]

    pingpong_process = await run_command_with_output(
        pingpong_worker_cmd, stdoutput_file, env
    )
    await wait_for_sync_point(Path(stdoutput_file), 10)
    return (pingpong_process, Path(report_file))


async def run_pingpong(
    n_messages: int,
    timeout: int,
    output_dir: str,
    build_mode: BuildMode,
    log_level: LogLevel,
    tracing_layers: Set[TracingLayer],
    hyper_bin: str,
    pingpong_worker_bin: str,
    iteration_idx: int,
    enable_profile: bool = False,
) -> GoldenFile:
    """Run a single iteration of the ping-pong test and return a GoldenFile with the results."""
    # Generate a unique execution ID
    exec_id = str(uuid.uuid4())

    # Define output files for this iteration
    iteration_dir = os.path.join(output_dir, f"iteration_{iteration_idx}")
    os.makedirs(iteration_dir, exist_ok=True)

    hyper_output = os.path.join(iteration_dir, "hyper_output.txt")

    # Create environment dictionary with tracing settings
    env = os.environ.copy()

    # Set log level based on the LogLevel enum
    env["RUST_LOG"] = log_level.value
    env["MONARCH_OTEL_LOG"] = log_level.value
    env["HYPERACTOR_EXECUTION_ID"] = exec_id

    # Disable tracing logging depending on what's set in args
    if TracingLayer.OTEL not in tracing_layers:
        env["DISABLE_OTEL_TRACING"] = "1"
    if TracingLayer.GLOG not in tracing_layers:
        env["DISABLE_GLOG_TRACING"] = "1"
    if TracingLayer.RECORDING not in tracing_layers:
        env["DISABLE_RECORDER_TRACING"] = "1"
    socket = f"/{iteration_dir}/monarch.sock"
    # Remove existing socket if it exists
    if os.path.exists(socket):
        os.remove(socket)
    hyperactor_bootstrap_addr = f"unix!{socket}"

    print(f"Iteration {iteration_idx}: exec id: {exec_id}")

    # Start the hyper server
    print(
        f"Iteration {iteration_idx}: Starting hyper server on {hyperactor_bootstrap_addr}..."
    )
    hyper_cmd = [
        hyper_bin,
        "serve",
        "-a",
        hyperactor_bootstrap_addr,
    ]
    hyper_process = await run_command_with_output(hyper_cmd, hyper_output, env)

    # Give the server a moment to start
    await asyncio.sleep(2)
    # Start timing

    # Start the processes and collect them
    processes = [
        start_pingpong_worker(
            n_messages=n_messages,
            output_dir=iteration_dir,
            worker_name=f"ping[{i}]",
            env=env,
            pingpong_worker_bin=pingpong_worker_bin,
            hyperactor_bootstrap_addr=hyperactor_bootstrap_addr,
        )
        for i in range(2)
    ]

    # Wait for pingpong workers to complete
    print(
        f"Iteration {iteration_idx}: Waiting for pingpong workers to complete (timeout: {timeout}s)..."
    )
    # Start profiling or monitoring if requested
    profile_process = None
    if enable_profile:
        # Define profile output file
        profile_output = os.path.join(iteration_dir, "profile_output.txt")

        print(f"Iteration {iteration_idx}: Starting profiling...")
        profile_cmd = [
            "strobe",
            "bpf",
            "--children",
            "--pids",
            str(get_grandparent_process()),
            "--event",
            "cpu_cycles",
            "--sample-interval",
            "1000000",
        ]
        profile_process = await run_command_with_output(
            profile_cmd, profile_output, env
        )
    duration_per_message = await handle_workers(await asyncio.gather(*processes))

    # Stop the hyper server
    print(f"Iteration {iteration_idx}: Stopping hyper server...")
    if hyper_process.returncode is None:
        hyper_process.terminate()
        try:
            await asyncio.wait_for(hyper_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            hyper_process.kill()

    # Stop profiling or monitoring
    if profile_process and profile_process.returncode is None:
        # Print profile output for iteration_idx==0
        print(f"Iteration {iteration_idx}: Stopping profiling...")
        profile_process.terminate()
        try:
            await asyncio.wait_for(profile_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            profile_process.kill()

    # Print profile output if it exists and this is the first iteration
    if enable_profile:
        print(f"Iteration {iteration_idx}: Profile output:")
        try:
            with open(os.path.join(iteration_dir, "profile_output.txt"), "r") as f:
                profile_content = f.read()
                print(profile_content)
        except Exception as e:
            print(f"Error reading profile output: {e}")

    fastests = min(duration_per_message)
    qps = 1.0 / fastests

    # Create GoldenFile instance with the current run data
    golden_file = GoldenFile(
        n_messages=n_messages,
        iterations=1,
        build_mode=build_mode,
        log_level=log_level,
        tracing_layers=tracing_layers,
        messages_per_second=qps,
    )

    print(f"Iteration {iteration_idx}: Messages per second: {qps:.2f}")
    return golden_file


def generate_tracing_combinations() -> List[Set[TracingLayer]]:
    """Generate all possible combinations of tracing layers.

    Returns:
        A list of sets, each containing a unique combination of tracing layers.
    """
    tracing_layers = list(TracingLayer)
    tracing_combinations = []

    # Generate all possible combinations of tracing layers (including empty set)
    for i in range(len(tracing_layers) + 1):
        for combo in itertools.combinations(tracing_layers, i):
            tracing_combinations.append(set(combo))

    return tracing_combinations


def generate_combinations(
    log_level: Optional[LogLevel] = None,
) -> List[Tuple[LogLevel, Set[TracingLayer]]]:
    """Generate combinations of log levels and tracing layers.

    Args:
        log_level: If provided, only generate combinations for this log level.
                  If None, generate combinations for all log levels.

    Returns:
        A list of tuples containing (log_level, tracing_layers) for all combinations.
    """
    tracing_combinations = generate_tracing_combinations()

    if log_level is not None:
        # Generate combinations for the specified log level only
        return [(log_level, tracing_combo) for tracing_combo in tracing_combinations]
    else:
        # Generate combinations for all log levels
        all_combinations = []
        for level in list(LogLevel):
            for tracing_combo in tracing_combinations:
                all_combinations.append((level, tracing_combo))
        return all_combinations


async def run_single_configuration(
    n_messages: int,
    iterations: int,
    timeout: int,
    output_dir: str,
    build_mode: BuildMode,
    log_level: LogLevel,
    tracing_layers: Set[TracingLayer],
    hyper_bin: str,
    pingpong_worker_bin: str,
    enable_profile: bool = False,
) -> GoldenFile:
    """Run pingpong for a single configuration and return the average golden file."""

    config_dir = os.path.join(
        output_dir,
        f"{log_level.value}__{'+'.join(sorted([layer.value for layer in tracing_layers])) if tracing_layers else 'no_tracing'}",
    )
    os.makedirs(config_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(
        f"Running configuration: Log Level={log_level.value}, Tracing Layers={tracing_layers}"
    )
    print(f"{'='*80}")

    # Run iterations of the test
    print(f"Running {iterations} iterations...")

    # Gather all results
    golden_files = [
        await run_pingpong(
            n_messages=n_messages,
            timeout=timeout,
            output_dir=config_dir,
            build_mode=build_mode,
            log_level=log_level,
            tracing_layers=tracing_layers,
            hyper_bin=hyper_bin,
            pingpong_worker_bin=pingpong_worker_bin,
            iteration_idx=i,
            enable_profile=enable_profile
            and i == 0,  # Only enable profiling for the first iteration
        )
        for i in range(iterations)
    ]

    avg_mps = sum(g.messages_per_second for g in golden_files) / len(golden_files)
    # Create a new GoldenFile with the average times
    avg_golden_file = GoldenFile(
        n_messages=n_messages,
        iterations=iterations,
        build_mode=build_mode,
        log_level=log_level,
        tracing_layers=tracing_layers,
        messages_per_second=avg_mps,
    )

    # Print the average results
    print("\nAverage results across all iterations:")
    # pyre-fixme[16]: `GoldenFile` has no attribute `to_json`.
    print(avg_golden_file.to_json(indent=2))

    # Load the existing golden file for comparison
    previous_golden = GoldenFile.from_file(log_level, tracing_layers)

    # Compare with previous golden file if available
    if previous_golden:
        prev_mps = previous_golden.messages_per_second
        percent_change = ((avg_mps - prev_mps) / prev_mps) * 100

        if percent_change < 0:
            comparison_msg = f"REGRESSION: Current run is {percent_change:.2f}% worse than previous run"
        else:
            comparison_msg = f"IMPROVEMENT: Current run is {abs(percent_change):.2f}% better than previous run"

        print("\nComparison with previous run:")
        print(f"  Previous mps: {prev_mps:.2f}")
        print(f"  Current mps:  {avg_mps:.2f}")
        print(f"  {comparison_msg}")

    else:
        print("No previous golden file found for comparison.")

    return avg_golden_file


async def async_main() -> Optional[int]:
    """Main async function to set up and run the Monarch ping_pong example."""
    args = parse_args()

    # Display all GoldenFiles in a table format
    display_goldenfiles_table()

    # Configuration
    build_mode: BuildMode = BuildMode(BuildInfo.get_build_mode())
    n_messages: int = args.n_messages
    iterations: int = args.iterations
    timeout: int = args.timeout

    # Get hyper and pingpong binaries from buck resources using importlib.resources
    try:
        with importlib.resources.path(__package__, "hyper") as hyper_path:
            hyper_bin = str(hyper_path)
        with importlib.resources.path(
            __package__, "pingpong_worker"
        ) as pingpong_wroker_path:
            pingpong_worker_bin = str(pingpong_wroker_path)
    except (ImportError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to load resources: {e}")

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp()

    print(f"Using output directory: {output_dir}")

    # Check for incompatible combinations
    if args.run_all and args.enable_profile:
        raise ValueError("--run-all and --enable-profile cannot be used together")

    if args.run_all and args.tracing_layers:
        raise ValueError("--run-all and --tracing-layers cannot be used together")

    # Validate log level requirements
    if not args.run_all and args.log_level is None:
        raise ValueError("--log-level is required when --run-all is not specified")

    # Generate combinations based on args
    all_combinations = []
    all_golden_files = []

    if args.run_all:
        print(
            "Running all combinations for LogLevel: " + args.log_level.value
            if args.log_level
            else "ALL"
        )
        all_combinations = generate_combinations(args.log_level)
    else:
        # Use tracing layers directly from args (already TracingLayer enum values)
        tracing_layers: Set[TracingLayer] = set(args.tracing_layers)

        # Use log level directly from args (already LogLevel enum value)
        print(
            f"Running with log level: {args.log_level.value}, tracing layers: {tracing_layers}"
        )
        all_combinations = [(args.log_level, tracing_layers)]

    # Run each combination
    all_golden_files = [
        await run_single_configuration(
            n_messages=n_messages,
            iterations=iterations,
            timeout=timeout,
            output_dir=output_dir,
            build_mode=build_mode,
            log_level=log_level,
            tracing_layers=tracing_layers,
            hyper_bin=hyper_bin,
            pingpong_worker_bin=pingpong_worker_bin,
            enable_profile=args.enable_profile,
        )
        for log_level, tracing_layers in all_combinations
    ]

    # Create a dictionary mapping (log_level, frozenset(tracing_layers)) to the new golden files
    new_golden_files_map = {}
    for golden_file in all_golden_files:
        key = (golden_file.log_level, frozenset(golden_file.tracing_layers))
        new_golden_files_map[key] = golden_file

    # Display comparison tables (current in red, new in green)
    display_comparison_tables(new_golden_files_map)

    # If --update is set, ask before rewriting goldenfiles with new times
    if args.update:
        prompt = "\nDo you want to save the new timing data? (y/n) "

        if input(prompt).lower().strip() == "y":
            for golden_file in all_golden_files:
                saved_path = golden_file.to_file()
                print(f"Golden file written to {saved_path}")

    print(f"Output directory: {output_dir}")


def main() -> Optional[int]:
    """Entry point for the script."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    main()
