"""
This is the main function for the worker / pipe processes. It expects the args to
the process to be passed in on the command line and accessible in `sys.argv`.

To see the supported arguments checkout `monarch_worker::bootstrap`.
"""

import importlib.resources
import os

import pdb  # noqa

from .debugger import _set_trace
from .logging import initialize_logging

if __name__ == "__main__":
    # torch is import to make sure all the dynamic types are registered
    import torch  # noqa

    if torch.cuda.is_available():
        # Force CUDA initialization early on. CUDA init is lazy, and Python CUDA
        # APIs are guarded to init CUDA if necessary. But our worker calls
        # raw libtorch APIs which are not similarly guarded. So just initialize here
        # to avoid issues with potentially using uninitialized CUDA state.
        torch.cuda.init()

    # pyre-ignore[21]
    from .._lib import worker  # @manual=//monarch/monarch_extension:monarch_extension

    initialize_logging()

    def check_set_device(device):
        import os

        if str(device) not in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","):
            raise ValueError(
                f"Only devices {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')} are available to monarch worker, "
                f"but torch.cuda.set_device({device}) was called"
            )

    torch.cuda.set_device = check_set_device

    with (
        importlib.resources.path("monarch", "py-spy") as pyspy,
    ):
        if pyspy.exists():
            os.environ["PYSPY_BIN"] = str(pyspy)
        # fallback to using local py-spy

    pdb.set_trace = _set_trace
    # pyre-ignore[16]
    worker.worker_main()
