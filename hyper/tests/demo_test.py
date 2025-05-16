# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from __future__ import annotations

import asyncio
import contextlib
import importlib.resources as resources
import itertools
import os
import shutil
from asyncio.subprocess import create_subprocess_exec, Process
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp, mkstemp, TemporaryFile
from types import TracebackType
from typing import IO, List

from later.unittest import TestCase


@lru_cache()
def hyper_bin() -> Path:
    with resources.path((__package__ or "") + ".hyper", "hyper") as path:
        return path


async def wait_for_open(addr: str, timeout: float) -> None:
    async def inner() -> None:
        while True:
            try:
                match addr.split("!"):
                    case ("unix", path):
                        reader, writer = await asyncio.open_unix_connection(path)
                        writer.close()
                    case (transport, _):
                        raise RuntimeError(
                            f"unsupported transport: {transport}",
                        )
                break
            except (ConnectionRefusedError, FileNotFoundError):
                await asyncio.sleep(0.1)

    await asyncio.wait_for(inner(), timeout)


class HyperProc:
    args: list[str]
    wraps: Process
    stdout: IO[bytes]
    stderr: IO[bytes]
    tasks: List[asyncio.Task]
    addr: str | None

    def __init__(
        self,
        args: list[str],
        wraps: Process,
        stdout: IO[bytes],
        stderr: IO[bytes],
        addr: str | None = None,
    ) -> None:
        self.args = args
        self.wraps = wraps
        self.stdout = stdout
        self.stderr = stderr
        self.tasks: List[asyncio.Task] = []
        self.addr = addr

    @property
    def returncode(self) -> int | None:
        return self.wraps.returncode

    def terminate(self) -> None:
        self.wraps.terminate()

    async def wait_for_output(self, timeout: float = 4) -> None:
        await asyncio.wait_for(self._wait_for_output(), timeout)

    async def wait_if_listening(self, timeout: float = 4) -> None:
        # not supposed to be listening to anything
        if self.addr is None:
            return
        await wait_for_open(self.addr, timeout)

    async def _wait_for_output(self) -> None:
        self.stdout.seek(0)
        self.stdout.read(1)

    def __str__(self) -> str:
        args = " ".join(self.args)
        return f"cmd: {args} exit: {self.returncode}\n{self.output()}"

    def output(self) -> str:
        self.stdout.seek(0)
        self.stderr.seek(0)
        stdout = self.stdout.read().decode()
        stderr = self.stderr.read().decode()
        return f"stdout:\n{stdout}\nstderr:\n{stderr}"

    async def wait(self) -> None:
        await asyncio.gather(self.wraps.wait(), *self.tasks)

    @staticmethod
    async def spawn(
        subcommand: str,
        *args: str,
        stdout: IO[bytes] | None = None,
        stderr: IO[bytes] | None = None,
        **kwargs: str,
    ) -> HyperProc:
        args_and_options = list(args) + list(
            itertools.chain(
                *[(f"--{name}", str(value)) for name, value in kwargs.items()]
            )
        )

        if stdout is None:
            stdout = TemporaryFile()
        if stderr is None:
            stderr = TemporaryFile()

        h = HyperProc(
            [hyper_bin().name, subcommand] + args_and_options,
            await create_subprocess_exec(
                hyper_bin(),
                subcommand,
                *args_and_options,
                stdout=stdout,
                stderr=stderr,
            ),
            stdout=stdout,
            stderr=stderr,
        )
        return h


class TaskGroup(contextlib.AbstractAsyncContextManager["TaskGroup", None]):
    tasks: List[asyncio.Task]
    procs: List[HyperProc]
    test_dir: Path

    def __init__(self) -> None:
        self.tasks: List[asyncio.Task] = []
        self.procs: List[HyperProc] = []
        self.test_dir = Path(mkdtemp(prefix="monarch_test"))

    async def spawn(self, subcommand: str, *args: str, **kwargs: str) -> HyperProc:
        nprocs = len(self.procs)
        prefix = f"{subcommand}.{nprocs}."
        stdout, _ = mkstemp(dir=self.test_dir, prefix=prefix, suffix=".out")
        stderr, _ = mkstemp(dir=self.test_dir, prefix=prefix, suffix=".err")

        proc = await HyperProc.spawn(
            subcommand,
            *args,
            stdout=os.fdopen(stdout, "r+b"),
            stderr=os.fdopen(stderr, "r+b"),
            **kwargs,
        )
        self.procs.append(proc)
        return proc

    async def __aenter__(self) -> TaskGroup:
        print(f"Test context can be found in {self.test_dir}")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self.test_dir.exists()
        if self.test_dir.is_symlink():
            self.test_dir.unlink(True)
        elif self.test_dir.is_dir():
            shutil.rmtree(self.test_dir)
        else:
            os.remove(self.test_dir)

        # make sure everything is dead
        for p in self.procs:
            try:
                p.terminate()
                await p.wait()
            except Exception:
                pass
        exceptions = await asyncio.gather(*self.tasks, return_exceptions=True)

        if exc is not None:
            print(exceptions)
            for p in self.procs:
                print(p)


class DemoIntegrationTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    async def asyncSetUp(self) -> None:
        # add here your setup code that needs to be run in a async context
        await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        # add here your teardown code that needs to be run in a async context
        await super().asyncTearDown()

    async def test_start_hyperactor(self) -> None:
        async with TaskGroup() as tg:
            path = Path(tg.test_dir) / "sock"
            addr = f"unix!{path}"
            system = await tg.spawn("serve", addr=addr)
            # make sure the binary starts and is listening on the socket
            await wait_for_open(addr, 4)
            await system.wait_for_output(10)
            system.terminate()
            await system.wait()
            self.assertEqual(system.returncode, -15)

    async def test_spawn_many(self) -> None:
        async with TaskGroup() as tg:
            nworkers = 10
            path = Path(tg.test_dir)
            system_addr = f"unix!{path}/system.sock"
            system = await tg.spawn("serve", addr=system_addr)
            world = await tg.spawn(
                "demo", "world", system_addr, "integration", "4", "4"
            )
            await system.wait_if_listening(4)
            await world.wait_for_output(4)
            workers = await asyncio.gather(
                *[
                    tg.spawn(
                        "demo",
                        "proc",
                        system_addr,
                        f"integration[{i}]",
                        addr=f"unix!{tg.test_dir}/integration[{i}].act.sock",
                    )
                    for i in range(nworkers)
                ]
            )

            await asyncio.gather(*[w.wait_for_output(10) for w in workers])

            for w in workers:
                await w.wait_if_listening(4)
                w.terminate()
                await w.wait()
            for w in workers:
                self.assertEqual(w.returncode, -15)
            system.terminate()
            await system.wait()
            self.assertEqual(system.returncode, -15)
