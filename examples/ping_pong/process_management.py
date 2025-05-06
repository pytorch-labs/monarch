#!/usr/bin/env python3
"""
Process management functions for the Monarch ping_pong example.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


async def wait_for_sync_point(path: Path, timeout: float) -> None:
    end = time.monotonic() + timeout
    while end > time.monotonic():
        if "SYNC_POINT" in path.read_text():
            return
        await asyncio.sleep(0.1)

    raise TimeoutError(f"Timed out waiting for sync point after {timeout}s")


async def run_command_with_output(
    command: List[str], output_file: str, env: Optional[Dict[str, str]] = None
) -> asyncio.subprocess.Process:
    """
    Run a command asynchronously and redirect its output to a file.

    Args:
        command: The command to run as a list of strings
        output_file: The file to write the output to
        env: Optional environment variables

    Returns:
        The process object
    """
    output_fd = open(output_file, "w")

    try:
        # Create the subprocess
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=output_fd,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        # Store the file descriptor in the process object for later cleanup
        process._output_fd = output_fd  # type: ignore

        return process
    except Exception:
        # Ensure the file is closed if an exception occurs during subprocess creation
        output_fd.close()
        raise


async def monitor_process(
    process: asyncio.subprocess.Process,
    start_time: float,
    timeout: float = 300.0,
) -> float:
    """
    Monitor a process for completion and update completion times.

    Args:
        process: The process to monitor
        name: The name of the process
        output_file: The output file to check for completion markers
        start_time: The start time of the process
        completion_event: Event to signal when the process completes
        completion_times: Dictionary to store completion times
    """
    # Wait for the process to complete
    await asyncio.wait_for(process.wait(), timeout=timeout)
    return time.monotonic() - start_time


async def wait_for_completion(
    processes: Sequence[asyncio.subprocess.Process], timeout: float
) -> Sequence[float]:
    """
    Wait for the processes to complete and return their execution times.

    Args:
        processes: List of (process, name, output_file) tuples
        timeout: Maximum time to wait in seconds

    Returns:
        Dictionary with process names as keys and execution times in seconds as values
    """
    start_time = time.monotonic()
    return await asyncio.gather(
        *[monitor_process(p, start_time, timeout) for p in processes]
    )


async def async_run_subprocess(
    command: List[str], check: bool = True, env: Optional[Dict[str, str]] = None
) -> asyncio.subprocess.Process:
    """
    Run a subprocess asynchronously and optionally check its return code.

    Args:
        command: The command to run as a list of strings
        check: Whether to check the return code
        env: Optional environment variables

    Returns:
        The completed process
    """
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        env=env,
    )

    _, stderr = await process.communicate()

    if check and process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(
            f"Command {command} failed with return code {process.returncode}: {error_msg}"
        )

    return process
