# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import asyncio
import importlib.resources
import logging
import operator
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import ModuleType
from typing import cast

import pytest

import torch
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask

from monarch._src.actor.actor_mesh import ActorMesh, Channel, Port

from monarch.actor import (
    Accumulator,
    Actor,
    current_actor_name,
    current_rank,
    current_size,
    endpoint,
    local_proc_mesh,
    proc_mesh,
)
from monarch.tools.config import defaults
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


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
async def test_stream():
    proc = await local_proc_mesh(gpus=2)
    v = await proc.spawn("counter2", Counter, 3)
    v.incr.broadcast()

    assert 8 == sum([await x for x in v.value.stream()])


class To(Actor):
    @endpoint
    async def whoami(self):
        return current_actor_name()


class From(Actor):
    @endpoint
    async def fetch(self, to: To):
        return [await x for x in to.whoami.stream()]


@pytest.mark.timeout(60)
async def test_mesh_passed_to_mesh():
    proc = await local_proc_mesh(gpus=2)
    f = await proc.spawn("from", From)
    t = await proc.spawn("to", To)
    all = [y for x in f.fetch.stream(t) for y in await x]
    assert len(all) == 4
    assert all[0] != all[1]


@pytest.mark.timeout(60)
async def test_mesh_passed_to_mesh_on_different_proc_mesh():
    proc = await local_proc_mesh(gpus=2)
    proc2 = await local_proc_mesh(gpus=2)
    f = await proc.spawn("from", From)
    t = await proc2.spawn("to", To)
    all = [y for x in f.fetch.stream(t) for y in await x]
    assert len(all) == 4
    assert all[0] != all[1]


@pytest.mark.timeout(60)
def test_actor_slicing():
    proc = local_proc_mesh(gpus=2)
    proc2 = local_proc_mesh(gpus=2)

    f = proc.spawn("from", From)
    t = proc2.spawn("to", To)

    assert t.slice(gpus=0).whoami.call().get() != t.slice(gpus=1).whoami.call().get()

    result = [y for x in f.fetch.stream(t.slice(gpus=0)) for y in x.get()]
    assert len(result) == 2

    assert result[0] == result[1]


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
async def test_sync_actor():
    proc = await local_proc_mesh(gpus=2)
    a = await proc.spawn("actor", SyncActor)
    c = await proc.spawn("counter", Counter, 5)
    r = await a.sync_endpoint.choose(c)
    assert r == 5


@pytest.mark.timeout(60)
def test_sync_actor_sync_client() -> None:
    proc = local_proc_mesh(gpus=2)
    a = proc.spawn("actor", SyncActor).get()
    c = proc.spawn("counter", Counter, 5).get()
    r = a.sync_endpoint.choose(c).get()
    assert r == 5


@pytest.mark.timeout(60)
def test_proc_mesh_size() -> None:
    proc = local_proc_mesh(gpus=2)
    assert 2 == proc.size("gpus")


@pytest.mark.timeout(60)
def test_rank_size_sync() -> None:
    proc = local_proc_mesh(gpus=2)
    r = proc.spawn("runit", RunIt).get()

    acc = Accumulator(r.run, 0, operator.add)
    assert 1 == acc.accumulate(lambda: current_rank()["gpus"]).get()
    assert 4 == acc.accumulate(lambda: current_size()["gpus"]).get()


@pytest.mark.timeout(60)
def test_accumulate_sync() -> None:
    proc = local_proc_mesh(gpus=2)
    counter = proc.spawn("counter", Counter, 1).get()
    counter.incr.broadcast()
    acc = Accumulator(counter.value, 0, operator.add)
    r = acc.accumulate().get()
    assert r == 4


class CastToCounter(Actor):
    @endpoint
    def doit(self, c: Counter):
        return list(c.value.call().get())


@pytest.mark.timeout(60)
def test_value_mesh() -> None:
    proc = local_proc_mesh(gpus=2)
    counter = proc.spawn("counter", Counter, 0).get()
    counter.slice(hosts=0, gpus=1).incr.broadcast()
    x = counter.value.call().get()
    assert 0 == x.item(hosts=0, gpus=0)
    assert 1 == x.item(hosts=0, gpus=1)
    assert 1 == x.slice(hosts=0, gpus=1).item()
    n = proc.spawn("ctc", CastToCounter).get()
    assert list(x) == n.slice(gpus=0).doit.call_one(counter).get()


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
def test_proc_mesh_liveness() -> None:
    mesh = proc_mesh(gpus=2)
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
    def get_value(self):
        return self.local.value

    @endpoint
    async def get_async(self):
        return self.local.value


@pytest.mark.timeout(60)
async def test_actor_tls() -> None:
    """Test that thread-local state is respected."""
    pm = proc_mesh(gpus=1)
    am = await pm.spawn("tls", TLSActor)
    await am.increment.call_one()
    await am.increment_async.call_one()
    await am.increment.call_one()
    await am.increment_async.call_one()

    assert 4 == await am.get_value.call_one()
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
    def get_value(self):
        return self.local.value


@pytest.mark.timeout(60)
async def test_actor_tls_full_sync() -> None:
    """Test that thread-local state is respected."""
    pm = proc_mesh(gpus=1)
    am = await pm.spawn("tls", TLSActorFullSync)
    await am.increment.call_one()
    await am.increment.call_one()
    await am.increment.call_one()
    await am.increment.call_one()

    assert 4 == await am.get_value.call_one()


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


# def test_actor_future() -> None:
#     v = 0

#     async def incr():
#         nonlocal v
#         v += 1
#         return v

#     # can use async implementation from sync
#     # if no non-blocking is provided
#     f = Future(impl=incr, requires_loop=False)
#     assert f.get() == 1
#     assert v == 1
#     assert f.get() == 1
#     assert asyncio.run(awaitit(f)) == 1

#     f = Future(impl=incr, requires_loop=False)
#     assert asyncio.run(awaitit(f)) == 2
#     assert f.get() == 2

#     async def incr2():
#         nonlocal v
#         v += 2
#         return v

#     # Use non-blocking optimization if provided
#     f = Future(impl=incr2)
#     assert f.get() == 4

#     async def nope():
#         nonlocal v
#         v += 1
#         raise ValueError("nope")

#     f = Future(impl=nope, requires_loop=False)

#     with pytest.raises(ValueError):
#         f.get()

#     assert v == 5

#     with pytest.raises(ValueError):
#         f.get()

#     assert v == 5

#     with pytest.raises(ValueError):
#         asyncio.run(awaitit(f))

#     assert v == 5

#     async def nope2():
#         nonlocal v
#         v += 1
#         raise ValueError("nope")

#     f = Future(impl=nope2)

#     with pytest.raises(ValueError):
#         f.get()

#     assert v == 6

#     with pytest.raises(ValueError):
#         f.result()

#     assert f.exception() is not None

#     assert v == 6

#     with pytest.raises(ValueError):
#         asyncio.run(awaitit(f))

#     assert v == 6

#     async def seven():
#         return 7

#     f = Future(impl=seven, requires_loop=False)

#     assert 7 == f.get(timeout=0.001)

#     async def neverfinish():
#         f = asyncio.Future()
#         await f

#     f = Future(impl=neverfinish, requires_loop=True)

#     with pytest.raises(asyncio.exceptions.TimeoutError):
#         f.get(timeout=0.1)


class Printer(Actor):
    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger()

    @endpoint
    async def print(self, content: str) -> None:
        print(f"{content}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

    @endpoint
    async def log(self, content: str) -> None:
        self._logger.error(f"{content}")
        for handler in self._logger.handlers:
            handler.flush()
        sys.stdout.flush()
        sys.stderr.flush()


@pytest.mark.timeout(60)
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
                pm = proc_mesh(gpus=2)
                am = await pm.spawn("printer", Printer)

                # Disable streaming logs to client
                await pm.logging_option(
                    stream_to_client=False, aggregate_window_sec=None
                )
                await asyncio.sleep(1)

                # These should not be streamed to client initially
                for _ in range(5):
                    await am.print.call("no print streaming")
                    await am.log.call("no log streaming")
                await asyncio.sleep(1)

                # Enable streaming logs to client
                await pm.logging_option(
                    stream_to_client=True, aggregate_window_sec=1, level=logging.FATAL
                )
                # Give it some time to reflect
                await asyncio.sleep(1)

                # These should be streamed to client
                for _ in range(5):
                    await am.print.call("has print streaming")
                    await am.log.call("no log streaming due to level mismatch")
                await asyncio.sleep(1)

                # Enable streaming logs to client
                await pm.logging_option(
                    stream_to_client=True, aggregate_window_sec=1, level=logging.ERROR
                )
                # Give it some time to reflect
                await asyncio.sleep(1)

                # These should be streamed to client
                for _ in range(5):
                    await am.print.call("has print streaming too")
                    await am.log.call("has log streaming as level matched")

                # Give it some time to reflect and aggregate
                await asyncio.sleep(1)

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

        with open(stderr_path, "r") as f:
            stderr_content = f.read()

        # Clean up temp files
        os.unlink(stdout_path)
        os.unlink(stderr_path)

        # Assertions on the captured output
        # Has a leading context so we can distinguish between streamed log and
        # the log directly printed by the child processes as they share the same stdout/stderr
        assert not re.search(
            r"similar log lines.*no print streaming", stdout_content
        ), stdout_content
        assert not re.search(
            r"similar log lines.*no print streaming", stderr_content
        ), stderr_content
        assert not re.search(
            r"similar log lines.*no log streaming", stdout_content
        ), stdout_content
        assert not re.search(
            r"similar log lines.*no log streaming", stderr_content
        ), stderr_content
        assert not re.search(
            r"similar log lines.*no log streaming due to level mismatch", stdout_content
        ), stdout_content
        assert not re.search(
            r"similar log lines.*no log streaming due to level mismatch", stderr_content
        ), stderr_content

        assert re.search(
            r"similar log lines.*has print streaming", stdout_content
        ), stdout_content
        assert not re.search(
            r"similar log lines.*has print streaming", stderr_content
        ), stderr_content
        assert re.search(
            r"similar log lines.*has print streaming too", stdout_content
        ), stdout_content
        assert not re.search(
            r"similar log lines.*has print streaming too", stderr_content
        ), stderr_content
        assert not re.search(
            r"similar log lines.*log streaming as level matched", stdout_content
        ), stdout_content
        assert re.search(
            r"similar log lines.*log streaming as level matched",
            stderr_content,
        ), stderr_content

    finally:
        # Ensure file descriptors are restored even if something goes wrong
        try:
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
        except OSError:
            pass


@pytest.mark.timeout(60)
async def test_logging_option_defaults() -> None:
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

                for _ in range(5):
                    await am.print.call("print streaming")
                    await am.log.call("log streaming")
                await asyncio.sleep(4)

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

        with open(stderr_path, "r") as f:
            stderr_content = f.read()

        # Clean up temp files
        os.unlink(stdout_path)
        os.unlink(stderr_path)

        # Assertions on the captured output
        assert re.search(
            r"similar log lines.*print streaming", stdout_content
        ), stdout_content
        assert not re.search(
            r"similar log lines.*print streaming", stderr_content
        ), stderr_content
        assert not re.search(
            r"similar log lines.*log streaming", stdout_content
        ), stdout_content
        assert re.search(
            r"similar log lines.*log streaming", stderr_content
        ), stderr_content

    finally:
        # Ensure file descriptors are restored even if something goes wrong
        try:
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
        except OSError:
            pass


# oss_skip: importlib not pulling resource correctly in git CI, needs to be revisited
@pytest.mark.oss_skip
async def test_flush_logs_fast_exit() -> None:
    # We use a subprocess to run the test so we can handle the flushed logs at the end.
    # Otherwise, it is hard to restore the original stdout/stderr.

    test_bin = importlib.resources.files(str(__package__)).joinpath("test_bin")

    # Run the binary in a separate process and capture stdout and stderr
    cmd = [str(test_bin), "flush-logs"]

    process = subprocess.run(cmd, capture_output=True, timeout=60, text=True)

    # Check if the process ended without error
    if process.returncode != 0:
        raise RuntimeError(f"{cmd} ended with error code {process.returncode}. ")

    # Assertions on the captured output, 160 = 32 procs * 5 logs per proc
    # 32 and 5 are specified in the test_bin flush-logs.
    assert (
        len(
            re.findall(
                r"160 similar log lines.*has print streaming",
                process.stdout,
            )
        )
        == 1
    ), process.stdout


@pytest.mark.timeout(60)
async def test_flush_on_disable_aggregation() -> None:
    """Test that logs are flushed when disabling aggregation.

    This tests the corner case: "Make sure we flush whatever in the aggregators before disabling aggregation."
    """
    # Save original file descriptors
    original_stdout_fd = os.dup(1)  # stdout

    try:
        # Create temporary files to capture output
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as stdout_file:
            stdout_path = stdout_file.name

            # Redirect file descriptors to our temp files
            os.dup2(stdout_file.fileno(), 1)

            # Also redirect Python's sys.stdout
            original_sys_stdout = sys.stdout
            sys.stdout = stdout_file

            try:
                pm = await proc_mesh(gpus=2)
                am = await pm.spawn("printer", Printer)

                # Set a long aggregation window to ensure logs aren't flushed immediately
                await pm.logging_option(stream_to_client=True, aggregate_window_sec=60)

                # Generate some logs that will be aggregated but not flushed immediately
                for _ in range(5):
                    await am.print.call("aggregated log line")
                await asyncio.sleep(1)

                # Now disable aggregation - this should trigger an immediate flush
                await pm.logging_option(
                    stream_to_client=True, aggregate_window_sec=None
                )

                # Wait a bit to ensure logs are collected
                await asyncio.sleep(1)
                for _ in range(5):
                    await am.print.call("single log line")

                # Wait a bit to ensure flush completes
                await asyncio.sleep(1)

                # Flush all outputs
                stdout_file.flush()
                os.fsync(stdout_file.fileno())

            finally:
                # Restore Python's sys.stdout
                sys.stdout = original_sys_stdout

        # Restore original file descriptors
        os.dup2(original_stdout_fd, 1)

        # Read the captured output
        with open(stdout_path, "r") as f:
            stdout_content = f.read()

        # Clean up temp files
        os.unlink(stdout_path)

        # Verify that logs were flushed when aggregation was disabled
        # We should see the aggregated logs in the output
        # 10 = 5 log lines * 2 procs
        assert re.search(
            r"\[10 similar log lines\].*aggregated log line", stdout_content
        ), stdout_content

        # No aggregated single log lines
        assert not re.search(
            r"similar log lines.*single log line", stdout_content
        ), stdout_content

        # 10 = 5 log lines * 2 procs
        assert len(re.findall(r"single log line", stdout_content)) == 10, stdout_content

    finally:
        # Ensure file descriptors are restored even if something goes wrong
        try:
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
        except OSError:
            pass


@pytest.mark.timeout(60)
async def test_adjust_aggregation_window() -> None:
    """Test that the flush deadline is updated when the aggregation window is adjusted.

    This tests the corner case: "This can happen if the user has adjusted the aggregation window."
    """
    # Save original file descriptors
    original_stdout_fd = os.dup(1)  # stdout

    try:
        # Create temporary files to capture output
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as stdout_file:
            stdout_path = stdout_file.name

            # Redirect file descriptors to our temp files
            os.dup2(stdout_file.fileno(), 1)

            # Also redirect Python's sys.stdout
            original_sys_stdout = sys.stdout
            sys.stdout = stdout_file

            try:
                pm = await proc_mesh(gpus=2)
                am = await pm.spawn("printer", Printer)

                # Set a long aggregation window initially
                await pm.logging_option(stream_to_client=True, aggregate_window_sec=100)

                # Generate some logs that will be aggregated
                for _ in range(3):
                    await am.print.call("first batch of logs")
                await asyncio.sleep(1)

                # Now adjust to a shorter window - this should update the flush deadline
                await pm.logging_option(stream_to_client=True, aggregate_window_sec=2)

                # Generate more logs
                for _ in range(3):
                    await am.print.call("second batch of logs")

                # Wait just enough time for the shorter window to trigger a flush
                await asyncio.sleep(1)

                # Flush all outputs
                stdout_file.flush()
                os.fsync(stdout_file.fileno())

            finally:
                # Restore Python's sys.stdout/stderr
                sys.stdout = original_sys_stdout

        # Restore original file descriptors
        os.dup2(original_stdout_fd, 1)

        # Read the captured output
        with open(stdout_path, "r") as f:
            stdout_content = f.read()

        # Clean up temp files
        os.unlink(stdout_path)

        # Verify that logs were flushed when the aggregation window was adjusted
        # We should see both batches of logs in the output
        assert re.search(
            r"\[6 similar log lines\].*first batch of logs", stdout_content
        ), stdout_content

        assert re.search(
            r"similar log lines.*second batch of logs", stdout_content
        ), stdout_content

    finally:
        # Ensure file descriptors are restored even if something goes wrong
        try:
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
        except OSError:
            pass


class SendAlot(Actor):
    @endpoint
    async def send(self, port: Port[int]):
        for i in range(100):
            port.send(i)


@pytest.mark.timeout(60)
def test_port_as_argument() -> None:
    proc_mesh = local_proc_mesh(gpus=1)
    s = proc_mesh.spawn("send_alot", SendAlot).get()
    send, recv = Channel[int].open()

    s.send.broadcast(send)

    for i in range(100):
        assert i == recv.recv().get()


@pytest.mark.timeout(15)
async def test_same_actor_twice() -> None:
    pm = proc_mesh(gpus=1)
    await pm.spawn("dup", Counter, 0).initialized

    # The second spawn with the same name should fail with a specific error
    with pytest.raises(Exception) as exc_info:
        await pm.spawn("dup", Counter, 0).initialized

    # Assert that the error message contains the expected text about duplicate actor name
    error_msg = str(exc_info.value)
    assert (
        "gspawn failed: an actor with name 'dup' has already been spawned" in error_msg
    ), f"Expected error message about duplicate actor name, got: {error_msg}"


class LsActor(Actor):
    def __init__(self, workspace: str):
        self.workspace = workspace

    @endpoint
    async def ls(self) -> list[str]:
        return os.listdir(self.workspace)


async def test_sync_workspace() -> None:
    pm = await proc_mesh(gpus=1)

    # create two workspaces: one for local and one for remote
    with tempfile.TemporaryDirectory() as workspace_src, tempfile.TemporaryDirectory() as workspace_dst, unittest.mock.patch.dict(
        os.environ, {"WORKSPACE_DIR": workspace_dst}
    ):
        os.environ["WORKSPACE_DIR"] = workspace_dst
        config = defaults.config("slurm", workspace_src)
        await pm.sync_workspace(
            workspace=config.workspace, conda=False, auto_reload=True
        )

        # now file in remote workspace initially
        am = await pm.spawn("ls", LsActor, workspace_dst)
        for item in list(am.ls.call().get()):
            assert len(item[1]) == 0

        # write a file to local workspace
        file_path = os.path.join(workspace_src, "new_file")
        with open(file_path, "w") as f:
            f.write("hello world")
            f.flush()

        # force a sync and it should populate on the dst workspace
        await pm.sync_workspace(config.workspace, conda=False, auto_reload=True)
        for item in list(am.ls.call().get()):
            assert len(item[1]) == 1
            assert item[1][0] == "new_file"
            file_path = os.path.join(workspace_dst, item[1][0])
            with open(file_path, "r") as f:
                assert f.readline() == "hello world"


class TestActorMeshStop(unittest.IsolatedAsyncioTestCase):
    async def test_actor_mesh_stop(self) -> None:
        pm = proc_mesh(gpus=2)
        am_1 = await pm.spawn("printer", Printer)
        am_2 = await pm.spawn("printer2", Printer)
        await am_1.print.call("hello 1")
        await am_1.log.call("hello 2")
        await cast(ActorMesh, am_1).stop()

        with self.assertRaisesRegex(
            RuntimeError, expected_regex="`PythonActorMesh` has already been stopped"
        ):
            await am_1.print.call("hello 1")

        await am_2.print.call("hello 3")
        await am_2.log.call("hello 4")


class PortedActor(Actor):
    @endpoint(explicit_response_port=True)
    def add(self, port: "Port[int]", b: int) -> None:
        port.send(3 + b)


@pytest.mark.timeout(60)
def test_ported_actor():
    proc_mesh = local_proc_mesh(gpus=1).get()
    a = proc_mesh.spawn("port_actor", PortedActor).get()
    assert 5 == a.add.call_one(2).get()


async def _recv():
    return (7, 2, 3)


async def consume():
    r = await PythonTask.from_coroutine(_recv())
    assert r == (7, 2, 3)


@pytest.mark.timeout(60)
def test_python_task_tuple() -> None:
    PythonTask.from_coroutine(consume()).block_on()


def test_select_result() -> None:
    def s(t):
        time.sleep(t)
        return t

    a = PythonTask.spawn_blocking(lambda: s(4))
    b = PythonTask.spawn_blocking(lambda: s(0))
    r = PythonTask.select_one([a.task(), b.task()]).block_on()
    assert r == (0, 1)


def test_mesh_len():
    proc_mesh = local_proc_mesh(gpus=12).get()
    s = proc_mesh.spawn("sync_actor", SyncActor).get()
    assert 12 == len(s)
