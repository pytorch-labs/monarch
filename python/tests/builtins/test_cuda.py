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
from monarch.builtins.cuda import (
    device_count,
    empty_cache,
    is_available,
    memory_allocated,
    memory_reserved,
)


@pytest.fixture(scope="module", autouse=True)
def testing_context():
    global local
    with TestingContext() as local:
        yield


@pytest.mark.timeout(120)
@pytest.mark.parametrize("backend_type", [BackendType.PY, BackendType.RS])
class TestCudaFunctions:
    @classmethod
    def local_device_mesh(cls, num_hosts, gpu_per_host, backend_type, activate=True):
        return local.local_device_mesh(
            num_hosts,
            gpu_per_host,
            activate,
            rust=backend_type == BackendType.RS,
        )

    def test_empty_cache(self, backend_type):
        with self.local_device_mesh(1, 1, backend_type) as device_mesh:
            with device_mesh.activate():
                _tensor = torch.rand(1000, 1000, device="cuda")
                initial_memory_allocation = memory_allocated()
                res1 = fetch_shard((initial_memory_allocation)).result()

                del _tensor
                empty_cache()
                after_clear_allocation = memory_allocated()
                res2 = fetch_shard((after_clear_allocation)).result()

                with no_mesh.activate():
                    print(res1, res2)
                    assert res1.item() > res2.item()

    def test_memory_allocated(self, backend_type):
        with self.local_device_mesh(1, 1, backend_type) as device_mesh:
            with device_mesh.activate():
                initial_memory_allocation = memory_allocated()
                res1 = fetch_shard((initial_memory_allocation)).result()
                _ = torch.rand(1000, 1000, device="cuda")
                new_memory_allocation = memory_allocated()
                res2 = fetch_shard((new_memory_allocation)).result()

                with no_mesh.activate():
                    print(res1, res2)
                    assert res1.item() < res2.item()

    def test_memory_reserved(self, backend_type):
        with self.local_device_mesh(1, 1, backend_type) as device_mesh:
            with device_mesh.activate():
                _ = torch.rand(1000, 1000, device="cuda")
                new_memory = memory_reserved()
                res = fetch_shard((new_memory)).result()

                with no_mesh.activate():
                    print(res)
                    assert res.item() > 0

    def test_device_count(self, backend_type):
        NUM_GPU = 1
        with self.local_device_mesh(1, NUM_GPU, backend_type) as device_mesh:
            with device_mesh.activate():
                count = device_count()
                result = fetch_shard(count).result()

                with no_mesh.activate():
                    assert result.item() == NUM_GPU

    def test_is_available(self, backend_type):
        with self.local_device_mesh(1, 1, backend_type) as device_mesh:
            with device_mesh.activate():
                available = is_available()
                result = fetch_shard(available).result()

                with no_mesh.activate():
                    assert result.item() == 1
