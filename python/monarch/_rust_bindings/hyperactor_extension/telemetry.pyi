# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def forward_to_tracing(message: str, file: str, lineno: int, level: int) -> None:
    """
    Log a message with the given metadata using the tracing system.

    This function forwards Python log messages to the Rust tracing system,
    preserving the original source location and log level.

    Args:
    - message (str): The log message content.
    - file (str): The file where the log message originated.
    - lineno (int): The line number where the log message originated.
    - level (int): The log level:
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
