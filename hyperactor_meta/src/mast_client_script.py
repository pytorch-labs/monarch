# Wrapper script for MAST to catch uncaught exceptions in the training script
# and report them to the reply file. This is bundled in the monarch fbpkg using
# the buck_filegroup rule.
import json
import logging
import runpy
import sys
import time
import traceback
from os import getenv
from typing import cast

from monarch.common.invocation import DeviceException, RemoteException

logger = logging.getLogger(__name__)


def _report_error(error_reason: RemoteException | DeviceException | Exception):
    file_path = getenv("PY_CLIENT_SCRIPT_REPLY_FILE")
    logger.info(f"writing reply file to {file_path}")
    if file_path is not None:
        error_message = {
            RemoteException: lambda e: cast(RemoteException, e).message,
            DeviceException: lambda e: cast(DeviceException, e).message,
            Exception: lambda e: f"{type(e).__name__}: {e}",
        }.get(type(error_reason), lambda e: f"{type(e).__name__}: {e}")(error_reason)
        py_callstack = "".join(
            {
                RemoteException: lambda e: traceback.format_list(
                    cast(RemoteException, e).worker_frames
                ),
                DeviceException: lambda e: traceback.format_list(
                    cast(DeviceException, e).frames
                ),
                Exception: lambda e: traceback.extract_tb(e.__traceback__).format(),
            }.get(
                type(error_reason),
                lambda e: traceback.extract_tb(e.__traceback__).format(),
            )(error_reason)
        )
        reply_message = json.dumps(
            {
                "message": error_message,
                "timestamp": int(time.time() * 1000),
                "pyCallStack": py_callstack,
            }
        )
        with open(file_path, "w") as reply_file:
            reply_file.write(reply_message)

        logger.info(f"done writing reply file to {file_path}")


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No target script provided"}), file=sys.stderr)
        sys.exit(1)

    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    sys.argv = [script_path] + script_args

    try:
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        _report_error(e)
        raise e


if __name__ == "__main__":
    main()
