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
import unittest
from logging import INFO
from types import ModuleType
from typing import cast

import pytest

import torch

from monarch._src.actor.actor_mesh import ActorMeshRef, Port, PortTuple

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


def test_actor_future() -> None:
    v = 0

    async def incr():
        nonlocal v
        v += 1
        return v

    # can use async implementation from sync
    # if no non-blocking is provided
    f = Future(impl=incr, requires_loop=False)
    assert f.get() == 1
    assert v == 1
    assert f.get() == 1
    assert asyncio.run(awaitit(f)) == 1

    f = Future(impl=incr, requires_loop=False)
    assert asyncio.run(awaitit(f)) == 2
    assert f.get() == 2

    async def incr2():
        nonlocal v
        v += 2
        return v

    # Use non-blocking optimization if provided
    f = Future(impl=incr2)
    assert f.get() == 4

    async def nope():
        nonlocal v
        v += 1
        raise ValueError("nope")

    f = Future(impl=nope, requires_loop=False)

    with pytest.raises(ValueError):
        f.get()

    assert v == 5

    with pytest.raises(ValueError):
        f.get()

    assert v == 5

    with pytest.raises(ValueError):
        asyncio.run(awaitit(f))

    assert v == 5

    async def nope2():
        nonlocal v
        v += 1
        raise ValueError("nope")

    f = Future(impl=nope2)

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

    f = Future(impl=seven, requires_loop=False)

    assert 7 == f.get(timeout=0.001)

    async def neverfinish():
        f = asyncio.Future()
        await f

    f = Future(impl=neverfinish, requires_loop=True)

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

                await pm.logging_option(stream_to_client=True)

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


class SendAlot(Actor):
    @endpoint
    async def send(self, port: Port[int]):
        for i in range(100):
            port.send(i)


def test_port_as_argument():
    proc_mesh = local_proc_mesh(gpus=1).get()
    s = proc_mesh.spawn("send_alot", SendAlot).get()
    send, recv = PortTuple.create(proc_mesh._mailbox)

    s.send.broadcast(send)

    for i in range(100):
        assert i == recv.recv().get()


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


class TestActorMeshStop(unittest.IsolatedAsyncioTestCase):
    async def test_actor_mesh_stop(self) -> None:
        pm = await proc_mesh(gpus=2)
        am_1 = await pm.spawn("printer", Printer)
        am_2 = await pm.spawn("printer2", Printer)
        await am_1.print.call("hello 1")
        await am_1.log.call("hello 2")
        await cast(ActorMeshRef, am_1).stop()

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="`ActorMesh` has been stopped"
        ):
            await am_1.print.call("hello 1")

        await am_2.print.call("hello 3")
        await am_2.log.call("hello 4")


class PortedActor(Actor):
    @endpoint(explicit_response_port=True)
    def add(self, port: "Port[int]", b: int) -> None:
        port.send(3 + b)


def test_ported_actor():
    proc_mesh = local_proc_mesh(gpus=1).get()
    a = proc_mesh.spawn("port_actor", PortedActor).get()
    assert 5 == a.add.call_one(2).get()
