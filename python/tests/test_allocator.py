# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import importlib.resources
import math
import os
import subprocess
import sys
import unittest
from datetime import timedelta
from typing import Generator

import cloudpickle

import torch
import torch.distributed as dist
import torch.nn.functional as F

from monarch._rust_bindings.hyperactor_extension.alloc import (
    AllocConstraints,
    AllocSpec,
)

from monarch._rust_bindings.monarch_hyperactor.channel import (
    ChannelAddr,
    ChannelTransport,
)
from monarch.actor_mesh import Actor, current_rank, current_size, endpoint, ValueMesh

from monarch.allocator import RemoteAllocator, StaticRemoteAllocInitializer
from monarch.proc_mesh import ProcMesh

from torch.distributed.elastic.utils.distributed import get_free_port

_100_MILLISECONDS = timedelta(milliseconds=100)


class TestActor(Actor):
    """Silly actor that computes the world size by all-reducing rank-hot tensors"""

    def __init__(self) -> None:
        self.rank: int = current_rank().rank
        self.world_size: int = math.prod(current_size().values())

    @endpoint
    async def compute_world_size(self, master_addr: str, master_port: int) -> int:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size)

        try:
            t = F.one_hot(torch.tensor(self.rank), num_classes=dist.get_world_size())
            dist.all_reduce(t)
            return int(torch.sum(t).item())
        finally:
            dist.destroy_process_group()


@contextlib.contextmanager
def remote_process_allocator() -> Generator[str, None, None]:
    with importlib.resources.path(__package__, "") as package_path:
        addr = ChannelAddr.any(ChannelTransport.Unix)

        process_allocator = subprocess.Popen(
            args=[
                "process_allocator",
                f"--addr={addr}",
            ],
            env={
                # prefix PATH with this test module's directory to
                # give 'process_allocator' and 'monarch_bootstrap' binary resources
                # in this test module's directory precedence over the installed ones
                # useful in BUCK where these binaries are added as 'resources' of this test target
                "PATH": f"{package_path}:{os.getenv('PATH', '')}",
                "RUST_LOG": "debug",
            },
        )
        try:
            yield addr
        finally:
            process_allocator.terminate()
            try:
                five_seconds = 5
                process_allocator.wait(timeout=five_seconds)
            except subprocess.TimeoutExpired:
                process_allocator.kill()


class TestRemoteAllocator(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cloudpickle.register_pickle_by_value(sys.modules[TestActor.__module__])

    @classmethod
    def tearDownClass(cls) -> None:
        cloudpickle.unregister_pickle_by_value(sys.modules[TestActor.__module__])

    def assert_computed_world_size(
        self, computed: ValueMesh[int], expected_world_size: int
    ) -> None:
        expected_world_sizes = {
            rank: expected_world_size for rank in range(0, expected_world_size)
        }
        computed_world_sizes = {p.rank: v for p, v in list(computed.flatten("rank"))}
        self.assertDictEqual(expected_world_sizes, computed_world_sizes)

    async def test_call_allocate_twice(self) -> None:
        class DeletingAllocInitializer(StaticRemoteAllocInitializer):
            """test initializer that removes the last address from the list each time initialize_alloc() is called
            used to test that the state of the initializer is preserved across calls to allocate()
            """

            async def initialize_alloc(self) -> list[str]:
                alloc = await super().initialize_alloc()
                self.addrs.pop(-1)
                return alloc

        with remote_process_allocator() as host1, remote_process_allocator() as host2:
            initializer = DeletingAllocInitializer(host1, host2)

            allocator = RemoteAllocator(
                world_id="test_remote_allocator",
                initializer=initializer,
                heartbeat_interval=_100_MILLISECONDS,
            )

            spec = AllocSpec(AllocConstraints(), host=1, gpu=1)

            await allocator.allocate(spec)
            self.assertEqual([host1], initializer.addrs)

            await allocator.allocate(spec)
            self.assertEqual([], initializer.addrs)

    async def test_throws_when_initializer_returns_empty_addrs(self) -> None:
        class EmptyAllocInitializer(StaticRemoteAllocInitializer):
            """test initializer that returns an empty list of addresses"""

            async def initialize_alloc(self) -> list[str]:
                return []

        empty_initializer = EmptyAllocInitializer()
        with self.assertRaisesRegex(
            RuntimeError, r"initializer must return non-empty list of addresses"
        ):
            allocator = RemoteAllocator(
                world_id="test_remote_allocator",
                initializer=empty_initializer,
                heartbeat_interval=_100_MILLISECONDS,
            )
            await allocator.allocate(AllocSpec(AllocConstraints(), host=1, gpu=1))

    async def test_allocate_2d_mesh(self) -> None:
        hosts = 2
        gpus = 4
        world_size = hosts * gpus
        spec = AllocSpec(AllocConstraints(), host=hosts, gpu=gpus)

        # create 2x process-allocators (on their own bind addresses) to simulate 2 hosts
        with remote_process_allocator() as host1, remote_process_allocator() as host2:
            allocator = RemoteAllocator(
                world_id="test_remote_allocator",
                initializer=StaticRemoteAllocInitializer(host1, host2),
                heartbeat_interval=_100_MILLISECONDS,
            )
            alloc = await allocator.allocate(spec)
            proc_mesh = await ProcMesh.from_alloc(alloc)
            actor = await proc_mesh.spawn("test_actor", TestActor)

            values = await actor.compute_world_size.call(
                master_addr="0.0.0.0",
                master_port=get_free_port(),
            )

            self.assert_computed_world_size(values, world_size)

    async def test_stacked_1d_meshes(self) -> None:
        # create two stacked actor meshes on the same host
        # each actor mesh running on separate process-allocators

        with remote_process_allocator() as host1_a, remote_process_allocator() as host1_b:
            allocator_a = RemoteAllocator(
                world_id="a",
                initializer=StaticRemoteAllocInitializer(host1_a),
                heartbeat_interval=_100_MILLISECONDS,
            )
            allocator_b = RemoteAllocator(
                world_id="b",
                initializer=StaticRemoteAllocInitializer(host1_b),
                heartbeat_interval=_100_MILLISECONDS,
            )

            spec_a = AllocSpec(AllocConstraints(), host=1, gpu=2)
            spec_b = AllocSpec(AllocConstraints(), host=1, gpu=6)

            proc_mesh_a = await ProcMesh.from_alloc(await allocator_a.allocate(spec_a))
            proc_mesh_b = await ProcMesh.from_alloc(await allocator_b.allocate(spec_b))

            actor_a = await proc_mesh_a.spawn("actor_a", TestActor)
            actor_b = await proc_mesh_b.spawn("actor_b", TestActor)

            results_a = await actor_a.compute_world_size.call(
                master_addr="0.0.0.0", master_port=get_free_port()
            )
            results_b = await actor_b.compute_world_size.call(
                master_addr="0.0.0.0", master_port=get_free_port()
            )

            self.assert_computed_world_size(results_a, 2)  # a is a 1x2 mesh
            self.assert_computed_world_size(results_b, 6)  # b is a 1x6 mesh
