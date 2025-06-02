# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import pytest
import torch
from monarch import fetch_shard, no_mesh

from monarch._testing import BackendType, TestingContext
from monarch.builtins.misc import get_default_dtype_remote, set_default_dtype_remote


@pytest.fixture(scope="module", autouse=True)
def testing_context():
    global local
    with TestingContext() as local:
        yield


@pytest.mark.timeout(120)
@pytest.mark.parametrize("backend_type", [BackendType.PY, BackendType.RS])
class TestMiscFunctions:
    @classmethod
    def local_device_mesh(cls, num_hosts, gpu_per_host, backend_type, activate=True):
        return local.local_device_mesh(
            num_hosts,
            gpu_per_host,
            activate,
            rust=backend_type == BackendType.RS,
        )

    def test_set_default_dtype_remote(self, backend_type):
        with self.local_device_mesh(1, 1, backend_type) as device_mesh:
            with device_mesh.activate():
                set_default_dtype_remote(torch.float64)

                t1 = torch.tensor([1.0])

                set_default_dtype_remote(torch.float32)

                t2 = torch.tensor([1.0])

                results = fetch_shard((t1, t2)).result()
                with no_mesh.activate():
                    assert results[0].dtype == torch.float64
                    assert results[1].dtype == torch.float32

    def test_get_default_dtype_remote(self, backend_type):
        with self.local_device_mesh(1, 1, backend_type) as device_mesh:
            with device_mesh.activate():
                original_dtype = get_default_dtype_remote()

                set_default_dtype_remote(torch.float64)
                middle_dtype = get_default_dtype_remote()

                set_default_dtype_remote(original_dtype)
                final_dtype = get_default_dtype_remote()

                results = fetch_shard(
                    (original_dtype, middle_dtype, final_dtype)
                ).result()
                with no_mesh.activate():
                    assert results[0] == results[2] == torch.float32
                    assert results[1] == torch.float64
