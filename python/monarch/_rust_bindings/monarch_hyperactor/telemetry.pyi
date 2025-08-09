# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

def forward_to_tracing(record: logging.LogRecord) -> None:
    """
    Log a message with the given metadata using the tracing system.

    This function forwards Python log messages to the Rust tracing system,
    preserving the original source location and log level.

    Args:
    - record (logging.LogRecord): The log record containing message, file, lineno, and level information.
      The function extracts:
        - message: The log message content via record.getMessage()
        - file: The filename via record.filename
        - lineno: The line number via record.lineno
        - level: The log level via record.levelno:
            - 10: DEBUG
            - 20: INFO
            - 30: WARN
            - 40: ERROR
            - other values default to INFO
    """
    ...

def enter_span(module_name: str, method_name: str, actor_id: str) -> None:
    """
    Enter a tracing span for a Python actor method.

    Creates and enters a new tracing span for the current thread that tracks
    execution of a Python actor method. The span is stored in thread-local
    storage and will be active until exit_span() is called.

    If a span is already active for the current thread, this function will
    preserve that span and not create a new one.

    Args:
    - module_name (str): The name of the module containing the actor (used as the target).
    - method_name (str): The name of the method being called (used as the span name).
    - actor_id (str): The ID of the actor instance (included as a field in the span).
    """
    ...

def exit_span() -> None:
    """
    Exit the current tracing span for a Python actor method.

    Exits and drops the tracing span that was previously created by enter_span().
    This should be called when the actor method execution is complete.

    If no span is currently active for this thread, this function has no effect.
    """
    ...

def get_current_span_id() -> int:
    """
    Get the current span ID from the active span.

    Returns the span ID of the current active span. If no span is active,
    returns 0 (invalid span ID).

    Returns:
    - int: The span ID as an integer.
    """
    ...

def use_real_clock() -> None:
    """
    Convenience function to switch to real-time clock.

    This switches the telemetry system to use real system time.
    """
    ...

def use_sim_clock() -> None:
    """
    Convenience function to switch to simulated clock.

    This switches the telemetry system to use simulated time, which is useful for
    testing and simulation environments where you want deterministic time control.
    """
    ...

def create_tracing_db_file() -> str:
    """
    Create a tracing database in a temporary file and return the file path.

    This function creates a SQLite database file in the system temporary directory
    with a unique name based on the process ID. The database is initialized with
    all the necessary tables (events, messages, actor_lifecycle, actors) and the
    custom assert() SQL function. The tracing subscriber is also initialized to
    capture events to this database.

    The returned file path can be used with Python's sqlite3.connect() to access
    the same database from Python code, enabling analysis of distributed system
    behavior, replica consistency checking, and custom SQL assertions.

    Returns:
    - str: The absolute path to the created SQLite database file.

    Raises:
    - RuntimeError: If the database file cannot be created or initialized.

    Example:
        >>> import sqlite3
        >>> db_path = create_tracing_db_file()
        >>> conn = sqlite3.connect(db_path)
        >>> # Now you can query the tracing data from Python
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT COUNT(*) FROM events")
        >>> count = cursor.fetchone()[0]
    """
    ...

class PySpan:
    def __init__(self, name: str) -> None:
        """
        Create a new PySpan.

        Args:
        - name (str): The name of the span.
        """
        ...

    def exit(self) -> None:
        """
        Exit the span.
        """
        ...

class PyCounter:
    def __init__(self, name: str) -> None:
        """
        Create a new PyCounter.

        Args:
        - name (str): The name of the counter metric.
        """
        ...

    def add(self, value: int) -> None:
        """
        Add a value to the counter.

        Args:
        - value (int): The value to add to the counter (must be non-negative).
        """
        ...

class PyHistogram:
    def __init__(self, name: str) -> None:
        """
        Create a new PyHistogram.

        Args:
        - name (str): The name of the histogram metric.
        """
        ...

    def record(self, value: float) -> None:
        """
        Record a value in the histogram.

        Args:
        - value (float): The value to record in the histogram.
        """
        ...

class PyUpDownCounter:
    def __init__(self, name: str) -> None:
        """
        Create a new PyUpDownCounter.

        Args:
        - name (str): The name of the up-down counter metric.
        """
        ...

    def add(self, value: int) -> None:
        """
        Add a value to the up-down counter.

        Args:
        - value (int): The value to add to the counter (can be positive or negative).
        """
        ...
