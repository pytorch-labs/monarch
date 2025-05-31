# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os
import time
from contextlib import contextmanager

import monarch

import torch
from monarch import remote
from monarch._rust_bindings.hyperactor_extension.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    AllocConstraints,
    AllocSpec,
)
from monarch.common import messages
from monarch.proc_mesh import ProcMesh
from monarch.python_local_mesh import _local_device_count
from monarch.rust_local_mesh import LoggingLocation, SocketType
from monarch_supervisor.logging import initialize_logging

initialize_logging()


@contextmanager
def local_device_mesh(
    hosts,
    gpu_per_host,
):
    with monarch.local_mesh(
        hosts=hosts,
        gpus_per_host=gpu_per_host,
        socket_type=SocketType.UNIX,
        logging_location=LoggingLocation.FILE,
    ) as dm:
        try:
            with dm.activate():
                yield dm
            dm.exit()
        except Exception:
            dm.client._shutdown = True
            raise


def python_throws_error_on_worker():
    with local_device_mesh(1, 1):
        throws = remote(
            "monarch.worker._testing_function.throw_python_exception",
            propagate=lambda: torch.tensor(1),
        )
        throws()
        time.sleep(10)
        torch.ones(())


def python_returns_unexpected_value_to_rust_on_worker():
    with local_device_mesh(1, 1):
        actually_returns_a_string = remote(
            "monarch.worker._testing_function.returns_a_string",
            propagate=lambda: torch.tensor(1),
        )
        client_thinks_im_a_tensor = actually_returns_a_string()
        client_thinks_im_a_tensor += 1
        time.sleep(10)
        torch.ones(())


def rust_error_on_worker():
    with local_device_mesh(1, 1) as mesh:

        class _Recording:
            def __init__(self):
                self.ref = 0

        # Trying to call a recording that doesn't exist will cause a rust error
        # on the worker.
        mesh._send(
            messages.CallRecording(
                ident=0, recording=_Recording(), results=[], actuals=[]
            )
        )
        time.sleep(10)
        torch.rand(3, 4)


def rust_panic_on_worker():
    with local_device_mesh(1, 1):
        panic = remote("__test_panic", propagate=lambda: torch.ones(()))
        panic()
        time.sleep(10)
        torch.rand(3, 4)


def worker_startup_import_error():
    # Monarch expects cuda 12.0, and setting this will cause a torch
    # import error on worker startup. Attempting to create the device
    # mesh will hang.
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64"
    with local_device_mesh(1, 1):
        pass


def worker_remote_function_import_error():
    with local_device_mesh(1, 1):
        import_error = remote(
            "monarch.worker._testing_throws_on_import._an_unusable_function",
            propagate=lambda: torch.ones(()),
        )
        import_error()
        time.sleep(10)
        torch.ones(())


def client_rust_error():
    gpus = _local_device_count()
    spec = AllocSpec(AllocConstraints(), gpus=gpus, hosts=1)
    allocator = monarch.LocalAllocator()
    alloc = allocator.allocate(spec).get()
    ProcMesh.from_alloc(alloc).get()
    # Attempting to reuse alloc for a new proc mesh will cause a rust error.
    ProcMesh.from_alloc(alloc).get()
