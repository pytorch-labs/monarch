"""
Simplified version of worker_main.py for testing the monarch_worker standalone.

We want a Python entrypoint here because we want to initialize the Monarch
Python extension on the main thread.
"""


def main() -> None:
    # torch is import to make sure all the dynamic types are registered
    import torch  # noqa

    # Force CUDA initialization early on. CUDA init is lazy, and Python CUDA
    # APIs are guarded to init CUDA if necessary. But our worker calls
    # raw libtorch APIs which are not similarly guarded. So just initialize here
    # to avoid issues with potentially using uninitialized CUDA state.
    torch.cuda.init()

    # pyre-ignore[21]
    from monarch._monarch._lib import (  # @manual=//monarch/monarch_extension:monarch_extension
        worker,
    )

    # pyre-ignore[16]
    worker.worker_main()


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
