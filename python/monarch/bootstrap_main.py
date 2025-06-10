# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is the main function for the boostrapping a new process using a ProcessAllocator.
"""

import asyncio
import importlib.resources
import logging
import os
import sys

# Import torch to avoid import-time races if a spawned actor tries to import torch.
import torch  # noqa[F401]


async def main():
    from monarch._rust_bindings.monarch_hyperactor.bootstrap import bootstrap_main

    await bootstrap_main()


def invoke_main():
    # if this is invoked with the stdout piped somewhere, then print
    # changes its buffering behavior. So we default to the standard
    # behavior of std out as if it were a terminal.
    sys.stdout.reconfigure(line_buffering=True)
    global bootstrap_main
    from monarch._rust_bindings.hyperactor_extension.telemetry import (  # @manual=//monarch/monarch_extension:monarch_extension  # @manual=//monarch/monarch_extension:monarch_extension
        forward_to_tracing,
    )

    # TODO: figure out what from worker_main.py we should reproduce here.

    class TracingForwarder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                forward_to_tracing(
                    record.getMessage(),
                    record.filename or "",
                    record.lineno or 0,
                    record.levelno,
                )
            except AttributeError:
                forward_to_tracing(
                    record.__str__(),
                    record.filename or "",
                    record.lineno or 0,
                    record.levelno,
                )

    if os.environ.get("MONARCH_ERROR_DURING_BOOTSTRAP_FOR_TESTING") == "1":
        raise RuntimeError("Error during bootstrap for testing")

    # forward logs to rust tracing. Defaults to on.
    if os.environ.get("MONARCH_PYTHON_LOG_TRACING", "1") == "1":
        logging.root.addHandler(TracingForwarder(level=logging.DEBUG))

    try:
        with (
            importlib.resources.path("monarch", "py-spy") as pyspy,
        ):
            if pyspy.exists():
                os.environ["PYSPY_BIN"] = str(pyspy)
            # fallback to using local py-spy
    except Exception as e:
        logging.warning(f"Failed to set up py-spy: {e}")

    # Start an event loop for PythonActors to use.
    asyncio.run(main())


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
