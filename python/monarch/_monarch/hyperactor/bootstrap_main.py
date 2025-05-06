"""
This is the main function for the boostrapping a new process using a ProcessAllocator.
"""

import asyncio
import importlib.resources
import os


async def main():
    hyperactor.init_asyncio_loop()
    await hyperactor.bootstrap_main()


if __name__ == "__main__":
    # TODO: figure out what from worker_main.py we should reproduce here.

    # pyre-ignore[21]
    from .._lib import (  # @manual=//monarch/monarch_extension:monarch_extension
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
