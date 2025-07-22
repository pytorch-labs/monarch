# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources
import os
import sys

try:
    from __manifest__ import fbmake  # noqa

    # simply checking for the existence of __manifest__ is not enough to tell if we are in a PAR
    # because monarch wheels include a dummy __manifest__ (see fbcode//monarch/python/monarch/session/meta/__manifest__.py)
    # so that we can use libfb programmatically. Hence additionally check if the `par_style` key is not null/empty
    IN_PAR = bool(fbmake.get("par_style"))
except ImportError:
    IN_PAR = False

PYTHON_EXECUTABLE: str
if IN_PAR:
    # The worker bootstrap binary will import this supervisor lib. When that
    # happens don't try to search for the bootstrap binary again, just use the
    # current executable.
    import __main__ as main_module  # @manual

    if hasattr(main_module, "__MONARCH_TENSOR_WORKER_ENV__"):
        PYTHON_EXECUTABLE = os.environ["FB_XAR_INVOKED_NAME"]
    else:
        try:
            with importlib.resources.as_file(
                importlib.resources.files("monarch_tensor_worker_env") / "worker_env"
            ) as path:
                if not path.exists():
                    raise ImportError()
                PYTHON_EXECUTABLE = str(path)
        except ImportError:
            raise ImportError(
                "Monarch worker env not found, please define a custom 'monarch_tensor_worker_env' or "
                "add '//monarch/python/monarch_supervisor/worker:default_worker_env' "
                "to your binary dependencies in TARGETS"
            )
else:
    PYTHON_EXECUTABLE = sys.executable
