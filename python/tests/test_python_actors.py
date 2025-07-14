# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import asyncio
import logging
import operator
import os
import sys
import tempfile
import threading
import time
from logging import INFO
from types import ModuleType

import pytest

import torch

from monarch.actor import (
    Accumulator,
    Actor,
    current_actor_name,
    current_rank,
    current_size,
    endpoint,
    Future,
    local_proc_mesh,
    proc_mesh,
)
from monarch.rdma import RDMABuffer
from typing_extensions import assert_type


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


class Counter(Actor):
    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        self.v += 1

    @endpoint
    async def value(self) -> int:
        return self.v

    @endpoint
    def value_sync_endpoint(self) -> int:
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

    # Test that Pyre derives the correct type for result (int, not Any)
    assert_type(result, int)
    result2 = await i.call_value.choose(v)

    assert result == result2

    result3 = await v.value_sync_endpoint.choose()
    assert_type(result, int)
    assert result2 == result3


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


@needs_cuda
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
    # copying a tensor across hosts moves it to CPU
    assert buffer_gpu.device.type == "cpu"

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


@needs_cuda
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


@needs_cuda
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


def test_proc_mesh_liveness() -> None:
    mesh = proc_mesh(gpus=2).get()
    counter = mesh.spawn("counter", Counter, 1).get()
    del mesh
    # Give some time for the mesh to have been shut down.
    # (It only would if there were a bug.)
    time.sleep(0.5)
    counter.value.call().get()


class TLSActor(Actor):
    """An actor that manages thread-local state."""

    def __init__(self):
        self.local = threading.local()
        self.local.value = 0

    @endpoint
    def increment(self):
        self.local.value += 1

    @endpoint
    async def increment_async(self):
        self.local.value += 1

    @endpoint
    def get(self):
        return self.local.value

    @endpoint
    async def get_async(self):
        return self.local.value


async def test_actor_tls() -> None:
    """Test that thread-local state is respected."""
    pm = await proc_mesh(gpus=1)
    am = await pm.spawn("tls", TLSActor)
    await am.increment.call_one()
    await am.increment_async.call_one()
    await am.increment.call_one()
    await am.increment_async.call_one()

    assert 4 == await am.get.call_one()
    assert 4 == await am.get_async.call_one()


class TLSActorFullSync(Actor):
    """An actor that manages thread-local state."""

    def __init__(self):
        self.local = threading.local()
        self.local.value = 0

    @endpoint
    def increment(self):
        self.local.value += 1

    @endpoint
    def get(self):
        return self.local.value


async def test_actor_tls_full_sync() -> None:
    """Test that thread-local state is respected."""
    pm = await proc_mesh(gpus=1)
    am = await pm.spawn("tls", TLSActorFullSync)
    await am.increment.call_one()
    await am.increment.call_one()
    await am.increment.call_one()
    await am.increment.call_one()

    assert 4 == await am.get.call_one()


class AsyncActor(Actor):
    def __init__(self):
        self.should_exit = False

    @endpoint
    async def sleep(self) -> None:
        while True and not self.should_exit:
            await asyncio.sleep(1)

    @endpoint
    async def no_more(self) -> None:
        self.should_exit = True


@pytest.mark.timeout(15)
async def test_async_concurrency():
    """Test that async endpoints will be processed concurrently."""
    pm = await proc_mesh(gpus=1)
    am = await pm.spawn("async", AsyncActor)
    fut = am.sleep.call()
    # This call should go through and exit the sleep loop, as long as we are
    # actually concurrently processing messages.
    await am.no_more.call()
    await fut


async def awaitit(f):
    return await f


def test_actor_future():
    v = 0

    async def incr():
        nonlocal v
        v += 1
        return v

    # can use async implementation from sync
    # if no non-blocking is provided
    f = Future(incr)
    assert f.get() == 1
    assert v == 1
    assert f.get() == 1
    assert asyncio.run(awaitit(f)) == 1

    f = Future(incr)
    assert asyncio.run(awaitit(f)) == 2
    assert f.get() == 2

    def incr2():
        nonlocal v
        v += 2
        return v

    # Use non-blocking optimization if provided
    f = Future(incr, incr2)
    assert f.get() == 4
    assert asyncio.run(awaitit(f)) == 4

    async def nope():
        nonlocal v
        v += 1
        raise ValueError("nope")

    f = Future(nope)

    with pytest.raises(ValueError):
        f.get()

    assert v == 5

    with pytest.raises(ValueError):
        f.get()

    assert v == 5

    with pytest.raises(ValueError):
        asyncio.run(awaitit(f))

    assert v == 5

    def nope2():
        nonlocal v
        v += 1
        raise ValueError("nope")

    f = Future(incr, nope2)

    with pytest.raises(ValueError):
        f.get()

    assert v == 6

    with pytest.raises(ValueError):
        f.result()

    assert f.exception() is not None

    assert v == 6

    with pytest.raises(ValueError):
        asyncio.run(awaitit(f))

    assert v == 6

    async def seven():
        return 7

    f = Future(seven)

    assert 7 == f.get(timeout=0.001)

    async def neverfinish():
        f = asyncio.Future()
        await f

    f = Future(neverfinish)

    with pytest.raises(asyncio.exceptions.TimeoutError):
        f.get(timeout=0.1)


class Printer(Actor):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(INFO)

    @endpoint
    async def print(self, content: str):
        print(f"{os.getpid()} {content}")

    @endpoint
    async def log(self, content: str):
        self.logger.info(f"{os.getpid()} {content}")


async def test_actor_log_streaming() -> None:
    # Save original file descriptors
    original_stdout_fd = os.dup(1)  # stdout
    original_stderr_fd = os.dup(2)  # stderr

    try:
        # Create temporary files to capture output
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as stdout_file, tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as stderr_file:
            stdout_path = stdout_file.name
            stderr_path = stderr_file.name

            # Redirect file descriptors to our temp files
            # This will capture both Python and Rust output
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            # Also redirect Python's sys.stdout/stderr for completeness
            original_sys_stdout = sys.stdout
            original_sys_stderr = sys.stderr
            sys.stdout = stdout_file
            sys.stderr = stderr_file

            try:
                pm = await proc_mesh(gpus=2)
                am = await pm.spawn("printer", Printer)

                await am.print.call("hello 1")
                await am.log.call("hello 2")

                pm.logging_option(stream_to_client=True)

                await am.print.call("hello 3")
                await am.log.call("hello 4")

                # Give it sometime to send log back
                time.sleep(5)

                # Flush all outputs
                stdout_file.flush()
                stderr_file.flush()
                os.fsync(stdout_file.fileno())
                os.fsync(stderr_file.fileno())

            finally:
                # Restore Python's sys.stdout/stderr
                sys.stdout = original_sys_stdout
                sys.stderr = original_sys_stderr

        # Restore original file descriptors
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)

        # Read the captured output
        with open(stdout_path, "r") as f:
            stdout_content = f.read()

        # Clean up temp files
        os.unlink(stdout_path)
        os.unlink(stderr_path)

        # TODO: (@jamessun) we need to disable logging forwarder for python logger
        # assert "hello 1" not in stdout_content
        assert "hello 2" not in stdout_content

        assert "hello 3" in stdout_content
        # assert "hello 4" in stdout_content

    finally:
        # Ensure file descriptors are restored even if something goes wrong
        try:
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
        except OSError:
            pass


@pytest.mark.timeout(15)
async def test_same_actor_twice() -> None:
    pm = await proc_mesh(gpus=1)
    await pm.spawn("dup", Counter, 0)

    # The second spawn with the same name should fail with a specific error
    with pytest.raises(Exception) as exc_info:
        await pm.spawn("dup", Counter, 0)

    # Assert that the error message contains the expected text about duplicate actor name
    error_msg = str(exc_info.value)
    assert (
        "gspawn failed: an actor with name 'dup' has already been spawned" in error_msg
    ), f"Expected error message about duplicate actor name, got: {error_msg}"
