import importlib.resources
import os
import sys

try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
except ImportError:
    IN_PAR = False

PYTHON_EXECUTABLE: str
if IN_PAR:
    # The worker bootstrap binary will import this supervisor lib. When that
    # happens don't try to search for the bootstrap binary again, just use the
    # current executable.
    import __main__ as main_module  # @manual

    if hasattr(main_module, "__MONARCH_WORKER_ENV__"):
        PYTHON_EXECUTABLE = os.environ["FB_XAR_INVOKED_NAME"]
    else:
        try:
            with importlib.resources.path("monarch_worker_env", "worker_env") as path:
                if not path.exists():
                    raise ImportError()
                PYTHON_EXECUTABLE = str(path)
        except ImportError:
            raise ImportError(
                "Monarch worker env not found, please define a custom 'monarch_worker_env' or "
                "add '//monarch/python/monarch_supervisor/worker:default_worker_env' "
                "to your binary dependencies in TARGETS"
            )
else:
    PYTHON_EXECUTABLE = sys.executable
