"""
This is the main function for the boostrapping a new process using a ProcessAllocator.
"""

import asyncio
import importlib.resources
import os


async def main():
    await hyperactor.bootstrap_main()


def invoke_main() -> None:
    global hyperactor
    # TODO: figure out what from worker_main.py we should reproduce here.

    # pyre-ignore[21]
    from ..._rust_bindings import (  # @manual=//monarch/monarch_extension:monarch_extension
        hyperactor,
    )

    with (
        importlib.resources.path("monarch", "py-spy") as pyspy,
    ):
        if pyspy.exists():
            os.environ["PYSPY_BIN"] = str(pyspy)
        # fallback to using local py-spy

    # Start an event loop for PythonActors to use.
    asyncio.run(main())


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    invoke_main()  # pragma: no cover
