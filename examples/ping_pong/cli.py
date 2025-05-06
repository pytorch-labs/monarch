#!/usr/bin/env python3
# pyre-strict
"""
Command-line interface for the Monarch ping_pong example.
"""

import argparse
import enum
from typing import Type, TypeVar

from monarch.examples.ping_pong.goldenfile import LogLevel, TracingLayer

# Define a generic type variable for enum types
E = TypeVar("E", bound=enum.Enum)


def enum_validator(value: str, enum_class: Type[E], error_prefix: str) -> E:
    """Generic validator for enum values."""
    try:
        return enum_class(value)
    except ValueError:
        valid_values = [item.value for item in enum_class]
        raise argparse.ArgumentTypeError(
            f"Invalid {error_prefix}: {value}. Valid values are: {', '.join(valid_values)}"
        )


def log_level_validator(value: str) -> LogLevel:
    """Validate and convert a string to a LogLevel enum value."""
    return enum_validator(value, LogLevel, "log level")


def tracing_layer_validator(value: str) -> TracingLayer:
    """Validate and convert a string to a TracingLayer enum value."""
    return enum_validator(value, TracingLayer, "tracing layer")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Get the default timing output path
    parser = argparse.ArgumentParser(
        description="Run the Monarch ping_pong example and write output to files."
    )
    # Create a mutually exclusive group for run-all vs specific configuration options
    run_all_group = parser.add_mutually_exclusive_group()
    run_all_group.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Run against all log levels and all combinations of tracing layers",
    )

    # Regular arguments (not in the run-all mutually exclusive group)
    parser.add_argument(
        "--enable-profile",
        action="store_true",
        default=False,
        help="Enable profiling (default: False)",
    )
    parser.add_argument(
        "--n-messages",
        type=int,
        default=10000,
        help="Number of messages (default: 10000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store output files (default: temporary directory)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for pingpong workers (default: 300)",
    )

    # Add log level options (not in the run-all mutually exclusive group)
    parser.add_argument(
        "--log-level",
        type=log_level_validator,
        default=None,  # No default, will be handled in main.py
        help="Set the log level. Required if --run-all is not specified.",
    )

    # Add tracing layer options (not in the run-all mutually exclusive group)
    parser.add_argument(
        "--tracing-layers",
        type=tracing_layer_validator,
        nargs="+",
        default=[],  # Default to no tracing
        help="Specify tracing layers to enable (default: no tracing)",
    )

    # Add update flag for golden file
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Update the golden file with new timing data (default: False)",
    )

    args = parser.parse_args()

    return args
