# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib.resources
import os
import subprocess
import sys

import pytest
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcEvent
from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch.actor import Actor, ActorError, endpoint, local_proc_mesh, proc_mesh


class ExceptionActor(Actor):
    @endpoint
    async def raise_exception(self) -> None:
        raise Exception("This is a test exception")

    @endpoint
    async def print_value(self, value) -> None:
        """Endpoint that takes a value and prints it."""
        print(f"Value received: {value}")
        return value


class ExceptionActorSync(Actor):
    @endpoint  # pyre-ignore
    def raise_exception(self) -> None:
        raise Exception("This is a test exception")


class BrokenPickleClass:
    """A class that can be configured to raise exceptions during pickling/unpickling."""

    def __init__(
        self,
        raise_on_getstate=False,
        raise_on_setstate=False,
        exception_message="Pickle error",
    ):
        self.raise_on_getstate = raise_on_getstate
        self.raise_on_setstate = raise_on_setstate
        self.exception_message = exception_message
        self.value = "test_value"

    def __getstate__(self):
        """Called when pickling the object."""
        if self.raise_on_getstate:
            raise RuntimeError(f"__getstate__ error: {self.exception_message}")
        return {
            "raise_on_getstate": self.raise_on_getstate,
            "raise_on_setstate": self.raise_on_setstate,
            "exception_message": self.exception_message,
            "value": self.value,
        }

    def __setstate__(self, state):
        """Called when unpickling the object."""
        if state.get("raise_on_setstate", False):
            raise RuntimeError(
                f"__setstate__ error: {state.get('exception_message', 'Unpickle error')}"
            )
        self.__dict__.update(state)


@pytest.mark.parametrize(
    "mesh",
    [local_proc_mesh, proc_mesh],
    ids=["local_proc_mesh", "distributed_proc_mesh"],
)
@pytest.mark.parametrize(
    "actor_class",
    [ExceptionActor, ExceptionActorSync],
)
@pytest.mark.parametrize("num_procs", [1, 2])
async def test_actor_exception(mesh, actor_class, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = await mesh(gpus=num_procs)
    exception_actor = await proc.spawn("exception_actor", actor_class)

    with pytest.raises(ActorError, match="This is a test exception"):
        if num_procs == 1:
            await exception_actor.raise_exception.call_one()
        else:
            await exception_actor.raise_exception.call()


@pytest.mark.parametrize(
    "mesh",
    [local_proc_mesh, proc_mesh],
    ids=["local_proc_mesh", "distributed_proc_mesh"],
)
@pytest.mark.parametrize(
    "actor_class",
    [ExceptionActor, ExceptionActorSync],
)
@pytest.mark.parametrize("num_procs", [1, 2])
def test_actor_exception_sync(mesh, actor_class, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = mesh(gpus=num_procs).get()
    exception_actor = proc.spawn("exception_actor", actor_class).get()

    with pytest.raises(ActorError, match="This is a test exception"):
        if num_procs == 1:
            exception_actor.raise_exception.call_one().get()
        else:
            exception_actor.raise_exception.call().get()


'''
# oss_skip: importlib not pulling resource correctly in git CI, needs to be revisited
@pytest.mark.oss_skip
@pytest.mark.parametrize("num_procs", [1, 2])
@pytest.mark.parametrize("sync_endpoint", [False, True])
@pytest.mark.parametrize("sync_test_impl", [False, True])
@pytest.mark.parametrize("endpoint_name", ["cause_segfault", "cause_panic"])
def test_actor_supervision(num_procs, sync_endpoint, sync_test_impl, endpoint_name):
    """
    Test that an endpoint causing spontaenous process exit is handled by the supervisor.

    Today, these events are delivered to the client and cause the client process
    to exit with a non-zero code, so the only way we can test it is via a
    subprocess harness.
    """
    # Run the segfault test in a subprocess
    test_bin = importlib.resources.files("monarch.python.tests").joinpath("test_bin")
    cmd = [
        str(test_bin),
        "error-endpoint",
        f"--num-procs={num_procs}",
        f"--sync-endpoint={sync_endpoint}",
        f"--sync-test-impl={sync_test_impl}",
        f"--endpoint-name={endpoint_name}",
    ]
    try:
        print("running cmd", " ".join(cmd))
        process = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        print("timeout expired")
        if e.stdout is not None:
            print(e.stdout.decode())
        if e.stderr is not None:
            print(e.stderr.decode())
        raise

    # Assert that the subprocess exited with a non-zero code
    assert "Started function error_test" in process.stdout.decode()
    assert (
        process.returncode != 0
    ), f"Expected non-zero exit code, got {process.returncode}"
'''


# oss_skip: importlib not pulling resource correctly in git CI, needs to be revisited
@pytest.mark.oss_skip
def test_proc_mesh_bootstrap_error():
    """
    Test that attempts to spawn a ProcMesh with a failure during bootstrap.
    """
    # Run the segfault test in a subprocess
    test_bin = importlib.resources.files("monarch.python.tests").joinpath("test_bin")
    cmd = [
        str(test_bin),
        "error-bootstrap",
    ]
    try:
        print("running cmd", " ".join(cmd))
        process = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        print("timeout expired")
        if e.stdout is not None:
            print(e.stdout.decode())
        if e.stderr is not None:
            print(e.stderr.decode())
        raise

    # Assert that the subprocess exited with a non-zero code
    assert "Started function error_bootstrap" in process.stdout.decode()
    assert (
        process.returncode != 0
    ), f"Expected non-zero exit code, got {process.returncode}"


@pytest.mark.parametrize("raise_on_getstate", [True, False])
@pytest.mark.parametrize("raise_on_setstate", [True, False])
@pytest.mark.parametrize("num_procs", [1, 2])
async def test_broken_pickle_class(raise_on_getstate, raise_on_setstate, num_procs):
    """
    Test that exceptions during pickling/unpickling are properly handled.

    This test creates a BrokenPickleClass instance configured to raise exceptions
    during __getstate__ and/or __setstate__, then passes it to an ExceptionActor's
    print_value endpoint and verifies that an ActorError is raised.
    """
    if not raise_on_getstate and not raise_on_setstate:
        # Pass this test trivially
        return

    proc = await proc_mesh(gpus=num_procs)
    exception_actor = await proc.spawn("exception_actor", ExceptionActor)

    # Create a BrokenPickleClass instance configured to raise exceptions
    broken_obj = BrokenPickleClass(
        raise_on_getstate=raise_on_getstate,
        raise_on_setstate=raise_on_setstate,
        exception_message="Test pickle error",
    )

    # On the getstate path, we expect a RuntimeError to be raised locally.
    # On the setstate path, we expect an ActorError to be raised remotely.
    error_type = RuntimeError if raise_on_getstate else ActorError
    error_pattern = "__getstate__ error" if raise_on_getstate else "__setstate__ error"

    with pytest.raises(error_type, match=error_pattern):
        if num_procs == 1:
            await exception_actor.print_value.call_one(broken_obj)
        else:
            await exception_actor.print_value.call(broken_obj)


"""
# oss_skip: importlib not pulling resource correctly in git CI, needs to be revisited
@pytest.mark.oss_skip
async def test_exception_after_wait_unmonitored():
    # Run the test in a subprocess
    test_bin = importlib.resources.files("monarch.python.tests").joinpath("test_bin")
    cmd = [
        str(test_bin),
        "error-unmonitored",
    ]
    try:
        print("running cmd", " ".join(cmd))
        process = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        print("timeout expired")
        if e.stdout is not None:
            print(e.stdout.decode())
        if e.stderr is not None:
            print(e.stderr.decode())
        raise

    # Assert that the subprocess exited with a non-zero code
    assert "Started function _error_unmonitored" in process.stdout.decode()
    assert (
        process.returncode != 0
    ), f"Expected non-zero exit code, got {process.returncode}"
"""


# oss_skip: importlib not pulling resource correctly in git CI, needs to be revisited
@pytest.mark.oss_skip
def test_python_actor_process_cleanup():
    """
    Test that PythonActor processes are cleaned up when the parent process dies.

    This test spawns an 8 process procmesh and calls an endpoint that returns a normal exception,
    then verifies that all spawned processes have been cleaned up after the spawned binary dies.
    """
    import os
    import signal
    import time

    # Run the error-cleanup test in a subprocess
    test_bin = importlib.resources.files("monarch.python.tests").joinpath("test_bin")
    cmd = [
        str(test_bin),
        "error-cleanup",
    ]

    try:
        print("running cmd", " ".join(cmd))
        process = subprocess.run(cmd, capture_output=True, timeout=180, text=True)
    except subprocess.TimeoutExpired as e:
        print("timeout expired")
        if e.stdout is not None:
            print(e.stdout.decode())
        if e.stderr is not None:
            print(e.stderr.decode())
        raise

    # Read stdout line by line to get child PIDs
    assert "Started function _error_cleanup() for parent process" in process.stdout

    child_pids = set()
    for line in process.stdout.splitlines():
        if line.startswith("CHILD_PIDS: "):
            pids_str = line[len("CHILD_PIDS: ") :]  # noqa
            child_pids = {
                int(pid.strip()) for pid in pids_str.split(",") if pid.strip()
            }
            print(f"Extracted child PIDs: {child_pids}")
            break

    if not child_pids:
        raise AssertionError("No child PIDs found in output")

    assert child_pids, "No child PIDs were collected from subprocess output"

    # Wait for child processes to be cleaned up
    print("Waiting for child processes to be cleaned up...")
    cleanup_timeout = 120
    start_time = time.time()

    def is_process_running(pid):
        """Check if a process with the given PID is still running."""
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
            return True
        except OSError:
            return False

    still_running = set(child_pids)

    while time.time() - start_time < cleanup_timeout:
        if not still_running:
            print("All child processes have been cleaned up!")
            return

        still_running = {pid for pid in still_running if is_process_running(pid)}

        print(f"Still running child PIDs: {still_running}")
        time.sleep(2)

    # If we get here, some processes are still running
    # Try to clean up remaining processes
    for pid in still_running:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    raise AssertionError(
        f"Child processes not cleaned up after {cleanup_timeout}s: {still_running}"
    )


class ActorFailureError(BaseException):
    """Exception to simulate actor failure for supervision testing.

    Inherits from BaseException in order that supervision be
    triggered.

    """

    pass


class ErrorActor(Actor):
    @endpoint
    def fail_with_supervision_error(self) -> None:
        raise ActorFailureError("Simulated actor failure for supervision testing")

    @endpoint
    async def fail_with_supervision_error_async(self) -> None:
        raise ActorFailureError("Simulated actor failure for supervision testing")

    @endpoint
    async def check(self) -> str:
        return "this is a healthy check"

    @endpoint
    async def check_with_exception(self) -> None:
        raise RuntimeError("failed the check with app error")


@pytest.mark.parametrize(
    "mesh",
    [local_proc_mesh, proc_mesh],
    ids=["local_proc_mesh", "distributed_proc_mesh"],
)
async def test_proc_mesh_redundant_monitoring(mesh):
    proc = await mesh(hosts=1, gpus=1)
    await proc.monitor()

    with pytest.raises(
        Exception, match="user already registered a monitor for this proc mesh"
    ):
        await proc.monitor()


class Worker(Actor):
    @endpoint
    def work(self):
        raise ValueError("value error")


class Manager(Actor):
    @endpoint
    async def init(self):
        mesh = await proc_mesh(gpus=1)
        self.workers = await mesh.spawn("Worker", Worker)

    @endpoint
    async def route(self):
        return await self.workers.work.call_one()


@pytest.mark.parametrize(
    "mesh",
    [local_proc_mesh, proc_mesh],
    ids=["local_proc_mesh", "distributed_proc_mesh"],
)
async def test_errors_propagated(mesh):
    p_mesh = await mesh(gpus=1)
    mesh = await p_mesh.spawn("manager", Manager)

    await mesh.init.call_one()

    with pytest.raises(ActorError) as err_info:
        await mesh.route.call_one()
    assert "value error" in str(err_info.value)


@pytest.mark.parametrize(
    "mesh",
    [local_proc_mesh, proc_mesh],
    ids=["local_proc_mesh", "distributed_proc_mesh"],
)
async def test_proc_mesh_monitoring(mesh):
    proc = await mesh(hosts=1, gpus=1)
    monitor = await proc.monitor()

    e = await proc.spawn("error", ErrorActor)

    with pytest.raises(Exception):
        await e.fail_with_supervision_error.call_one()

    event = await anext(monitor)
    assert isinstance(event, ProcEvent.Crashed)
    assert event[0] == 0  # check rank
    assert "ActorFailureError" in event[1]  # check error message
    assert (
        "Simulated actor failure for supervision testing" in event[1]
    )  # check error message

    # should not be able to spawn actors anymore as proc mesh is unhealthy
    with pytest.raises(SupervisionError, match="proc mesh is stopped with reason"):
        await proc.spawn("ex", ExceptionActorSync)


@pytest.mark.parametrize(
    "mesh",
    [local_proc_mesh, proc_mesh],
    ids=["local_proc_mesh", "distributed_proc_mesh"],
)
async def test_actor_mesh_supervision_handling(mesh):
    proc = await mesh(hosts=1, gpus=1)

    e = await proc.spawn("error", ErrorActor)

    # first check() call should succeed
    await e.check.call()

    # throw an application error
    with pytest.raises(ActorError, match="failed the check with app error"):
        await e.check_with_exception.call()

    # actor mesh should still be healthy
    await e.check.call()

    # existing call should fail with supervision error
    with pytest.raises(SupervisionError, match="supervision error:"):
        await e.fail_with_supervision_error.call_one()

    # new call should fail with check of health state of actor mesh
    with pytest.raises(SupervisionError, match="actor mesh is not in a healthy state"):
        await e.check.call()

    # should not be able to spawn actors anymore as proc mesh is unhealthy
    with pytest.raises(SupervisionError, match="proc mesh is stopped with reason"):
        await proc.spawn("ex", ExceptionActorSync)


class HealthyActor(Actor):
    @endpoint
    async def check(self):
        return "this is a healthy check"

    @endpoint
    async def check_with_payload(self, payload: str):
        pass


class Intermediate(Actor):
    @endpoint
    async def init_local_mesh(self):
        mesh = await local_proc_mesh(gpus=1)
        self._error_actor = await mesh.spawn("error", ErrorActor)
        self._healthy_actor = await mesh.spawn("healthy", HealthyActor)

    @endpoint
    async def init_proc_mesh(self):
        mesh = await proc_mesh(gpus=1)
        self._error_actor = await mesh.spawn("error", ErrorActor)
        self._healthy_actor = await mesh.spawn("healthy", HealthyActor)

    @endpoint
    async def forward_success(self):
        return await self._error_actor.check.call()

    @endpoint
    async def forward_error(self):
        return await self._error_actor.fail_with_supervision_error.call_one()

    @endpoint
    async def forward_healthy_check(self):
        return await self._healthy_actor.check.call()


@pytest.mark.parametrize(
    "mesh", [local_proc_mesh, proc_mesh], ids=["local_proc_mesh", "proc_mesh"]
)
async def test_actor_mesh_supervision_handling_chained_error(mesh):
    proc = await mesh(hosts=1, gpus=1)

    intermediate_actor = await proc.spawn("intermediate", Intermediate)
    if mesh is proc_mesh:
        await intermediate_actor.init_proc_mesh.call()
    elif mesh is local_proc_mesh:
        await intermediate_actor.init_local_mesh.call()

    # first forward() call should succeed
    await intermediate_actor.forward_success.call()
    await intermediate_actor.forward_healthy_check.call()

    # in a chain of client -> Intermediate -> ErrorActor, a supervision error
    # happening in ErrorActor will be captured by Intermediate and re-raised
    # as an application error (ActorError).
    with pytest.raises(ActorError, match="supervision error:"):
        await intermediate_actor.forward_error.call()

    # calling success endpoint should fail with ActorError, but with supervision msg.
    with pytest.raises(ActorError, match="actor mesh is not in a healthy state"):
        await intermediate_actor.forward_success.call()

    # healthy actor should still be working
    await intermediate_actor.forward_healthy_check.call()


@pytest.mark.parametrize(
    "mesh", [local_proc_mesh, proc_mesh], ids=["local_proc_mesh", "proc_mesh"]
)
@pytest.mark.parametrize(
    "method_name",
    ["fail_with_supervision_error", "fail_with_supervision_error_async"],
)
async def test_base_exception_handling(mesh, method_name):
    """Test that BaseException subclasses trigger supervision errors.

    This test verifies that both synchronous and asynchronous methods
    that raise ActorFailureError (a BaseException subclass) trigger
    supervision errors properly.

    """
    proc = await mesh(hosts=1, gpus=1)
    error_actor = await proc.spawn("error", ErrorActor)

    # Get the method to call based on the parameter
    method = getattr(error_actor, method_name)

    # The call should raise a SupervisionError
    with pytest.raises(SupervisionError, match="supervision error:"):
        await method.call_one()

    # Subsequent calls should fail with a health state error
    with pytest.raises(SupervisionError, match="actor mesh is not in a healthy state"):
        await error_actor.check.call()


@pytest.mark.parametrize(
    "mesh", [local_proc_mesh, proc_mesh], ids=["local_proc_mesh", "proc_mesh"]
)
async def test_supervision_with_proc_mesh_stopped(mesh):
    proc = await mesh(hosts=1, gpus=1)
    actor_mesh = await proc.spawn("healthy", HealthyActor)

    await actor_mesh.check.call()

    await proc.stop()

    # new call should fail with check of health state of actor mesh
    with pytest.raises(SupervisionError, match="actor mesh is not in a healthy state"):
        await actor_mesh.check.call()

    # proc mesh cannot spawn new actors anymore
    with pytest.raises(RuntimeError, match="`ProcMesh` has already been stopped"):
        await proc.spawn("immediate", Intermediate)


# TODO - re-enable after resolving T232206970
@pytest.mark.oss_skip
async def test_supervision_with_sending_error():
    os.environ["HYPERACTOR_CODEC_MAX_FRAME_LENGTH"] = "9999999999"
    os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT_SECS"] = "1"

    proc = await proc_mesh(gpus=1)
    actor_mesh = await proc.spawn("healthy", HealthyActor)

    await actor_mesh.check.call()

    # send a small payload to trigger success
    await actor_mesh.check_with_payload.call(payload="a")

    # send a large payload to trigger send timeout error
    with pytest.raises(
        SupervisionError, match="supervision error:.*message not delivered:"
    ):
        await actor_mesh.check_with_payload.call(payload="a" * 5000000000)

    # new call should fail with check of health state of actor mesh
    with pytest.raises(SupervisionError, match="actor mesh is not in a healthy state"):
        await actor_mesh.check.call()
    with pytest.raises(SupervisionError, match="actor mesh is not in a healthy state"):
        await actor_mesh.check_with_payload.call(payload="a")
