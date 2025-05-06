#!/usr/bin/env python3
# pyre-strict
from __future__ import annotations

import enum
import json
import os
import subprocess
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Set, Tuple

from dataclasses_json import dataclass_json


class BuildMode(enum.Enum):
    """Build modes for the Monarch ping_pong example."""

    DEV = "dev"
    DEVO_NOSAN = "dev-nosan"
    OPT = "opt"
    DBG = "dbg"


class LogLevel(enum.Enum):
    """Log levels for the Monarch ping_pong example."""

    ERROR = "error"
    INFO = "info"
    DEBUG = "debug"
    TRACE = "trace"


class TracingLayer(enum.Enum):
    """Configuration for tracing layers."""

    OTEL = "otel"
    GLOG = "glog"
    RECORDING = "recording"


def gen_filename(log_level: LogLevel, tracing_layers: Set[TracingLayer]) -> str:
    """Generate a filename based on log level and tracing layers.

    Returns:
        A filename string
    """
    # Format the tracing layers as a sorted, comma-separated string
    tracing_str = (
        "_".join(sorted(layer.value for layer in tracing_layers))
        if tracing_layers
        else "no_tracing"
    )

    # Format the filename with log level and tracing layers
    return f"pingpong.golden.{log_level.value}__{tracing_str}.json"


def display_goldenfiles_table(
    golden_files_map: Optional[
        Dict[Tuple[LogLevel, frozenset[TracingLayer]], GoldenFile]
    ] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Display GoldenFiles in a table format.

    Args:
        golden_files_map: Optional map of (log_level, tracing_layers) to GoldenFile.
                         If None, will load from disk.
        color: Optional color for the table (e.g., "red", "green").
        title: Optional title for the table.

    Left axis is log level, top axis is tracing layers.
    """
    # ANSI color codes
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    # Set color code based on parameter
    color_code = ""
    if color == "red":
        color_code = RED
    elif color == "green":
        color_code = GREEN
    # If no golden_files_map provided, load from disk
    if golden_files_map is None:
        golden_dir = get_default_goldenfile_dir()
        if not os.path.exists(golden_dir):
            print(f"{color_code}No golden files directory found at {golden_dir}{RESET}")
            return

        # Get all golden files
        golden_files = []

        # Iterate through build mode subdirectories
        for build_mode_dir in os.listdir(golden_dir):
            build_mode_path = os.path.join(golden_dir, build_mode_dir)
            if not os.path.isdir(build_mode_path):
                continue

            try:
                build_mode = BuildMode(build_mode_dir)
            except ValueError:
                # Skip directories that don't match a build mode
                continue

            # Look for golden files in this build mode directory
            for filename in os.listdir(build_mode_path):
                if filename.startswith("pingpong.golden.") and filename.endswith(
                    ".json"
                ):
                    file_path = os.path.join(build_mode_path, filename)
                    try:
                        with open(file_path, "r") as f:
                            data = json.loads(f.read())
                            # Create GoldenFile instance directly from the data
                            golden_file = GoldenFile(
                                n_messages=data.get("n_messages", 0),
                                iterations=data.get("iterations", 0),
                                build_mode=build_mode,
                                log_level=LogLevel(data.get("log_level", "debug")),
                                tracing_layers={
                                    TracingLayer(layer)
                                    for layer in data.get("tracing_layers", [])
                                },
                                messages_per_second=data.get(
                                    "messages_per_second", 0.0
                                ),
                            )
                            golden_files.append(golden_file)
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        print(
                            f"{color_code}Error loading golden file {filename}: {e}{RESET}"
                        )
    else:
        # Use provided golden_files_map
        golden_files = list(golden_files_map.values())

    if not golden_files:
        print(f"{color_code}No golden files found{RESET}")
        return

    # Get all unique log levels and tracing layer combinations
    log_levels = sorted(LogLevel, key=lambda x: x.value)
    build_modes = sorted(BuildMode, key=lambda x: x.value)

    # Get all unique tracing layer combinations
    tracing_combinations = set()
    for gf in golden_files:
        tracing_combinations.add(frozenset(gf.tracing_layers))

    # Sort tracing combinations by number of layers and then alphabetically
    sorted_tracing_combinations = sorted(
        tracing_combinations,
        key=lambda x: (len(x), sorted([layer.value for layer in x])),
    )

    # For each build mode, create a mapping of (log_level, tracing_layers) -> golden_file
    for build_mode in build_modes:
        # Filter golden files for this build mode
        build_mode_files = [gf for gf in golden_files if gf.build_mode == build_mode]

        if not build_mode_files:
            continue

        golden_file_map = {}
        for gf in build_mode_files:
            key = (gf.log_level, frozenset(gf.tracing_layers))
            golden_file_map[key] = gf

        # Print the table header for this build mode
        header_text = f"\nGolden Files Table for Build Mode: {build_mode.value} (Messages per second)"
        if title:
            header_text = f"\n{title} - Build Mode: {build_mode.value} (Messages Per Second (Higher is better))"
        print(f"{color_code}{header_text}")
        print(f"{color_code}{'-' * 80}")

        # Print the column headers (tracing layer combinations)
        header = "Log Level"
        header_width = max(
            len("Log Level"), max(len(level.value) for level in log_levels)
        )
        header = header.ljust(header_width + 2)

        for tracing_combo in sorted_tracing_combinations:
            if not tracing_combo:
                header += "| No Tracing ".ljust(15)
            else:
                combo_str = "+".join(sorted([layer.value for layer in tracing_combo]))
                header += f"| {combo_str} ".ljust(15)

        print(f"{color_code}{header}")
        print(f"{color_code}{'-' * 80}")

        # Print each row (log level)
        for log_level in log_levels:
            row = log_level.value.ljust(header_width + 2)

            for tracing_combo in sorted_tracing_combinations:
                key = (log_level, frozenset(tracing_combo))
                if key in golden_file_map:
                    gf = golden_file_map[key]
                    row += f"| {gf.messages_per_second:.2f} ".ljust(15)
                else:
                    row += "| --- ".ljust(15)

            print(f"{color_code}{row}")

    print(f"{color_code}{'-' * 80}{RESET}")
    print()


def display_comparison_tables(
    new_golden_files: Dict[Tuple[LogLevel, frozenset[TracingLayer]], GoldenFile],
) -> None:
    """Display two tables side by side - current goldenfiles (red) and new goldenfiles (green).

    Args:
        new_golden_files: Dictionary mapping (log_level, tracing_layers) to new GoldenFile instances
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF GOLDEN FILES")
    print("=" * 80)

    # Display current goldenfiles from disk (in red)
    display_goldenfiles_table(None, "red", "CURRENT GOLDEN FILES")

    # Display new goldenfiles from the run (in green)
    display_goldenfiles_table(new_golden_files, "green", "NEW GOLDEN FILES")


def get_default_goldenfile_dir() -> str:
    """Get the default path for timing output based on Buck root."""
    try:
        # Run 'buck root' command and capture its output
        buck_root = subprocess.check_output(["buck", "root"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        # If 'buck root' fails, fall back to the current directory
        buck_root = os.getcwd()

    # Set the default timing output path relative to the Buck root
    return os.path.join(buck_root, "monarch/examples/ping_pong/goldenfiles")


@dataclass_json
@dataclass(frozen=True)
class GoldenFile:
    """Dataclass for golden file in the Monarch ping_pong example."""

    # Configuration
    n_messages: int
    iterations: int
    build_mode: BuildMode
    log_level: LogLevel
    tracing_layers: Set[TracingLayer]
    messages_per_second: float
    __comment__: str = (
        "@" + "generated by `buck run //monarch/examples/ping_pong:run_script`"
    )

    # Default directory for golden files
    DEFAULT_GOLDEN_DIR: ClassVar[str] = get_default_goldenfile_dir()

    @classmethod
    def from_file(
        cls, log_level: LogLevel, tracing_layers: Set[TracingLayer]
    ) -> Optional[GoldenFile]:
        """Load a GoldenFile from a file, if it exists.

        Returns:
            A new GoldenFile instance or None if the file doesn't exist or is invalid
        """
        # Get the current build mode
        from libfb.py.build_info import BuildInfo

        build_mode = BuildMode(BuildInfo.get_build_mode())

        # Create a temporary GoldenFile to generate the filename
        temp_golden = GoldenFile(
            n_messages=0,
            iterations=0,
            build_mode=build_mode,
            log_level=log_level,
            tracing_layers=tracing_layers,
            messages_per_second=0.0,
        )

        # Get the build mode directory and filename
        build_mode_dir = os.path.join(cls.DEFAULT_GOLDEN_DIR, build_mode.value)
        filename = temp_golden.generate_filename()
        file_path = os.path.join(build_mode_dir, filename)

        try:
            with open(
                file_path,
                "r",
            ) as f:
                data = json.loads(f.read())
                # pyre-fixme[16]: `GoldenFile` has no attribute `from_dict`.
                return cls.from_dict(data)

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Error loading previous golden file from {file_path}. {e}")
            print("Create it using --update")
            return None

    def to_file(self) -> str:
        """Save the GoldenFile to a file.

        Args:
            file_path: Path to save the file to. If None, a default path will be generated.

        Returns:
            The path where the file was saved
        """
        # Create the build mode directory if it doesn't exist
        build_mode_dir = os.path.join(
            self.__class__.DEFAULT_GOLDEN_DIR, self.build_mode.value
        )
        os.makedirs(build_mode_dir, exist_ok=True)

        # Generate the file path
        file_path = os.path.join(build_mode_dir, self.generate_filename())

        with open(file_path, "w") as f:
            # pyre-fixme[16]: `GoldenFile` has no attribute `to_json`.
            f.write(self.to_json())

        return str(file_path)

    def generate_filename(self) -> str:
        """Generate a filename based on log level and tracing layers."""
        return gen_filename(self.log_level, self.tracing_layers)
