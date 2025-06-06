# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from types import ModuleType

import monarch

import pytest

import torch

from monarch.actor_mesh import (
    Accumulator,
    Actor,
    current_actor_name,
    current_rank,
    current_size,
    endpoint,
)

from monarch.mesh_controller import spawn_tensor_engine

from monarch.proc_mesh import local_proc_mesh, proc_mesh
from monarch.rdma import RDMABuffer


class Counter(Actor):
    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        self.v += 1

    @endpoint
    async def value(self) -> int:
        return self.v


class Indirect(Actor):
    @endpoint
    async def call_value(self, c: Counter) -> int:
        return await c.value.choose()


class ParameterServer(Actor):
    def __init__(self):
        self.params = torch.rand(10, 10)
        self.grad_buffer = torch.rand(10, 10)

    @endpoint
    async def grad_handle(self) -> RDMABuffer:
        byte_tensor = self.grad_buffer.view(torch.uint8).flatten()
        return RDMABuffer(byte_tensor)

    @endpoint
    async def update(self):
        self.params += 0.01 * self.grad_buffer

    @endpoint
    async def get_grad_buffer(self) -> torch.Tensor:
        # just used for testing
        return self.grad_buffer


async def test_choose():
    proc = await local_proc_mesh(gpus=2)
    v = await proc.spawn("counter", Counter, 3)
    i = await proc.spawn("indirect", Indirect)
    v.incr.broadcast()
    result = await v.value.choose()
    result2 = await i.call_value.choose(v)

    assert result == result2


async def test_stream():
    proc = await local_proc_mesh(gpus=2)
    v = await proc.spawn("counter2", Counter, 3)
    v.incr.broadcast()

    assert 8 == sum([x async for x in v.value.stream()])


class ParameterClient(Actor):
    def __init__(self, server, buffer):
        self.server = server
        byte_tensor = buffer.view(torch.uint8).flatten()
        self.buffer = byte_tensor

    @endpoint
    async def upload(self, tensor):
        gh = await self.server.grad_handle.call_one()
        await gh.write(tensor)

    @endpoint
    async def download(self):
        gh = await self.server.grad_handle.call_one()
        await gh.read_into(self.buffer)

    @endpoint
    async def get_buffer(self):
        return self.buffer


async def test_proc_mesh_rdma():
    proc = await proc_mesh(gpus=1)
    server = await proc.spawn("server", ParameterServer)

    # --- CPU TESTS ---
    client_cpu = await proc.spawn(
        "client_cpu", ParameterClient, server, torch.ones(10, 10)
    )
    x = await client_cpu.get_buffer.call_one()
    assert torch.sum(x.view(torch.float32).view(10, 10)) == 100
    zeros = torch.zeros(10, 10)
    await client_cpu.upload.call_one(zeros.view(torch.uint8).flatten())
    await client_cpu.download.call_one()
    x = await client_cpu.get_buffer.call_one()
    assert torch.sum(x.view(torch.float32).view(10, 10)) == 0

    # --- Modify server's backing buffer directly ---
    await server.update.call_one()

    # Should reflect updated values
    await client_cpu.download.call_one()

    buffer = await client_cpu.get_buffer.call_one()
    remote_grad = await server.get_grad_buffer.call_one()
    assert torch.allclose(buffer.view(torch.float32).view(10, 10), remote_grad)

    # --- GPU TESTS ---
    client_gpu = await proc.spawn(
        "client_gpu", ParameterClient, server, torch.ones(10, 10, device="cuda")
    )
    x = await client_gpu.get_buffer.call_one()
    buffer = x.view(torch.float32).view(10, 10)
    assert torch.sum(buffer) == 100
    zeros = torch.zeros(10, 10, device="cuda")
    await client_gpu.upload.call_one(zeros.view(torch.uint8).flatten())
    await client_gpu.download.call_one()
    x = await client_gpu.get_buffer.call_one()
    buffer_gpu = x.view(torch.float32).view(10, 10)
    assert torch.sum(buffer_gpu) == 0
    assert buffer_gpu.device.type == "cuda"

    # Modify server state again
    await server.update.call_one()
    await client_gpu.download.call_one()
    x = await client_gpu.get_buffer.call_one()
    buffer_gpu = x.view(torch.float32).view(10, 10)
    remote_grad = await server.get_grad_buffer.call_one()
    assert torch.allclose(buffer_gpu.cpu(), remote_grad)


class To(Actor):
    @endpoint
    async def whoami(self):
        return current_actor_name()


class From(Actor):
    @endpoint
    async def get(self, to: To):
        return [x async for x in to.whoami.stream()]


async def test_mesh_passed_to_mesh():
    proc = await local_proc_mesh(gpus=2)
    f = await proc.spawn("from", From)
    t = await proc.spawn("to", To)
    all = [y async for x in f.get.stream(t) for y in x]
    assert len(all) == 4
    assert all[0] != all[1]


async def test_mesh_passed_to_mesh_on_different_proc_mesh():
    proc = await local_proc_mesh(gpus=2)
    proc2 = await local_proc_mesh(gpus=2)
    f = await proc.spawn("from", From)
    t = await proc2.spawn("to", To)
    all = [y async for x in f.get.stream(t) for y in x]
    assert len(all) == 4
    assert all[0] != all[1]


async def test_actor_slicing():
    proc = await local_proc_mesh(gpus=2)
    proc2 = await local_proc_mesh(gpus=2)

    f = await proc.spawn("from", From)
    t = await proc2.spawn("to", To)

    assert await t.slice(gpus=0).whoami.call() != await t.slice(gpus=1).whoami.call()

    result = [y async for x in f.get.stream(t.slice(gpus=0)) for y in x]
    assert len(result) == 2

    assert result[0] == result[1]


async def test_aggregate():
    proc = await local_proc_mesh(gpus=2)
    counter = await proc.spawn("counter", Counter, 1)
    counter.incr.broadcast()
    acc = Accumulator(counter.value, 0, operator.add)
    r = await acc.accumulate()
    assert r == 4


class RunIt(Actor):
    @endpoint
    async def run(self, fn):
        return fn()


async def test_rank_size():
    proc = await local_proc_mesh(gpus=2)
    r = await proc.spawn("runit", RunIt)

    acc = Accumulator(r.run, 0, operator.add)

    assert 1 == await acc.accumulate(lambda: current_rank()["gpus"])
    assert 4 == await acc.accumulate(lambda: current_size()["gpus"])


class TrainerActor(Actor):
    def __init__(self):
        super().__init__()
        self.trainer = torch.nn.Linear(10, 10).to("cuda")
        self.trainer.weight.data.zero_()

    @endpoint
    async def init(self, gen):
        ranks = current_rank()
        self.gen = gen.slice(**ranks)

    @endpoint
    async def exchange_metadata(self):
        byte_tensor = self.trainer.weight.data.view(torch.uint8).flatten()
        self.handle = RDMABuffer(byte_tensor)
        await self.gen.attach_weight_buffer.call(self.handle)

    @endpoint
    async def weights_ready(self):
        self.trainer.weight.data.add_(1.0)


class GeneratorActor(Actor):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(10, 10).to("cuda")
        self.step = 0

    @endpoint
    async def init(self, trainer):
        ranks = current_rank()
        self.trainer = trainer.slice(**ranks)

    @endpoint
    async def attach_weight_buffer(self, handle):
        self.handle = handle

    @endpoint
    async def update_weights(self):
        self.step += 1
        byte_tensor = self.generator.weight.data.view(torch.uint8).flatten()
        await self.handle.read_into(byte_tensor)
        assert (
            torch.sum(self.generator.weight.data) == self.step * 100
        ), f"{torch.sum(self.generator.weight.data)=}, {self.step=}"


async def test_gpu_trainer_generator():
    trainer_proc = await proc_mesh(gpus=1)
    gen_proc = await proc_mesh(gpus=1)
    trainer = await trainer_proc.spawn("trainer", TrainerActor)
    generator = await gen_proc.spawn("gen", GeneratorActor)

    await generator.init.call(trainer)
    await trainer.init.call(generator)
    await trainer.exchange_metadata.call()

    for _ in range(3):
        await trainer.weights_ready.call()
        await generator.update_weights.call()


class SyncActor(Actor):
    @endpoint
    def sync_endpoint(self, a_counter: Counter):
        return a_counter.value.choose().get()


async def test_sync_actor():
    proc = await local_proc_mesh(gpus=2)
    a = await proc.spawn("actor", SyncActor)
    c = await proc.spawn("counter", Counter, 5)
    r = await a.sync_endpoint.choose(c)
    assert r == 5


def test_gpu_trainer_generator_sync() -> None:
    trainer_proc = proc_mesh(gpus=1).get()
    gen_proc = proc_mesh(gpus=1).get()
    trainer = trainer_proc.spawn("trainer", TrainerActor).get()
    generator = gen_proc.spawn("gen", GeneratorActor).get()

    generator.init.call(trainer).get()
    trainer.init.call(generator).get()
    trainer.exchange_metadata.call().get()

    for _ in range(3):
        trainer.weights_ready.call().get()
        generator.update_weights.call().get()


def test_sync_actor_sync_client():
    proc = local_proc_mesh(gpus=2).get()
    a = proc.spawn("actor", SyncActor).get()
    c = proc.spawn("counter", Counter, 5).get()
    r = a.sync_endpoint.choose(c).get()
    assert r == 5


def test_proc_mesh_size() -> None:
    proc = local_proc_mesh(gpus=2).get()
    assert 2 == proc.size("gpus")


def test_rank_size_sync() -> None:
    proc = local_proc_mesh(gpus=2).get()
    r = proc.spawn("runit", RunIt).get()

    acc = Accumulator(r.run, 0, operator.add)
    assert 1 == acc.accumulate(lambda: current_rank()["gpus"]).get()
    assert 4 == acc.accumulate(lambda: current_size()["gpus"]).get()


def test_accumulate_sync() -> None:
    proc = local_proc_mesh(gpus=2).get()
    counter = proc.spawn("counter", Counter, 1).get()
    counter.incr.broadcast()
    acc = Accumulator(counter.value, 0, operator.add)
    r = acc.accumulate().get()
    assert r == 4


class CastToCounter(Actor):
    @endpoint
    def doit(self, c: Counter):
        return list(c.value.call().get())


def test_value_mesh() -> None:
    proc = local_proc_mesh(gpus=2).get()
    counter = proc.spawn("counter", Counter, 0).get()
    counter.slice(hosts=0, gpus=1).incr.broadcast()
    x = counter.value.call().get()
    assert 0 == x.item(hosts=0, gpus=0)
    assert 1 == x.item(hosts=0, gpus=1)
    assert 1 == x.slice(hosts=0, gpus=1).item()
    n = proc.spawn("ctc", CastToCounter).get()
    assert list(x) == n.slice(gpus=0).doit.call_one(counter).get()


def test_rust_binding_modules_correct() -> None:
    import monarch._rust_bindings as bindings

    def check(module, path):
        for name, value in module.__dict__.items():
            if name.startswith("__"):
                continue
            if isinstance(value, ModuleType):
                check(value, f"{path}.{name}")
            elif hasattr(value, "__module__"):
                assert value.__name__ == name
                assert value.__module__ == path

    check(bindings, "monarch._rust_bindings")


def test_tensor_engine() -> None:
    pm = proc_mesh(gpus=2).get()

    dm = spawn_tensor_engine(pm)
    with dm.activate():
        r = monarch.inspect(2 * torch.zeros(3, 4))

    fm = dm.flatten("all")
    with fm.activate():
        f = monarch.inspect(2 * torch.zeros(3, 4), all=1)

    assert torch.allclose(torch.zeros(3, 4), r)
    assert torch.allclose(torch.zeros(3, 4), f)

    dm.exit()
