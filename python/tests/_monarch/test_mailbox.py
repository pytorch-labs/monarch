# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import pickle
from typing import Callable, cast, final, Generic, TypeVar

import monarch

from monarch._rust_bindings.hyperactor_extension.alloc import (
    AllocConstraints,
    AllocSpec,
)

from monarch._rust_bindings.monarch_hyperactor.actor import PanicFlag, PythonMessage

from monarch._rust_bindings.monarch_hyperactor.mailbox import (
    Mailbox,
    PortReceiver,
    PortRef,
)
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh
from monarch._rust_bindings.monarch_hyperactor.shape import Shape


S = TypeVar("S")
U = TypeVar("U")


@final
class Reducer(Generic[U]):
    def __init__(
        self,
        reduce_f: Callable[[U, U], U],
    ) -> None:
        self._reduce_f: Callable[[U, U], U] = reduce_f

    def __call__(self, left: PythonMessage, right: PythonMessage) -> PythonMessage:
        l: U = cast(U, pickle.loads(left.message))
        r: U = cast(U, pickle.loads(right.message))
        result: U = self._reduce_f(l, r)
        return PythonMessage(left.method, pickle.dumps(result), None, None)


@final
class Accumulator(Generic[S, U]):
    def __init__(
        self,
        initial_state: S,
        accumulate_f: Callable[[S, U], S],
        reduce_f: Callable[[U, U], U] | None,
    ) -> None:
        self._initial_state: S = initial_state
        self._accumulate_f: Callable[[S, U], S] = accumulate_f
        self._reducer: Reducer[U] | None = (
            Reducer(reduce_f) if reduce_f is not None else None
        )

    def __call__(self, state: PythonMessage, update: PythonMessage) -> PythonMessage:
        s: S = cast(S, pickle.loads(state.message))
        u: U = cast(U, pickle.loads(update.message))
        result: S = self._accumulate_f(s, u)
        return PythonMessage(state.method, pickle.dumps(result), None, None)

    @property
    def initial_state(self) -> PythonMessage:
        return PythonMessage(
            " @Accumulator.initial_state", pickle.dumps(self._initial_state), None, None
        )

    @property
    def reducer(self) -> Reducer[U] | None:
        return self._reducer


async def allocate() -> ProcMesh:
    spec = AllocSpec(AllocConstraints(), replica=1)
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate(spec)
    proc_mesh = await ProcMesh.allocate_nonblocking(alloc)
    return proc_mesh


async def test_accumulator() -> None:
    proc_mesh = await allocate()
    mailbox: Mailbox = proc_mesh.client

    def my_accumulate(state: str, update: int) -> str:
        return f"{state}+{update}"

    accumulator = Accumulator("init", my_accumulate, None)
    receiver: PortReceiver
    handle, receiver = mailbox.open_accum_port(accumulator)
    port_ref: PortRef = handle.bind()

    def post_message(value: int) -> None:
        port_ref.send(
            mailbox, PythonMessage("test_accumulator", pickle.dumps(value), None, None)
        )

    async def recv_message() -> str:
        messge = await asyncio.wait_for(receiver.recv(), timeout=5)
        value = pickle.loads(messge.message)
        return cast(str, value)

    post_message(1)
    # Receive the first message
    assert await recv_message() == "init+1"

    post_message(2)
    post_message(3)
    post_message(4)
    # Receive the 2nd to 4th messages
    assert await recv_message() == "init+1+2+3+4"


class MyActor:
    async def handle(
        self, mailbox: Mailbox, message: PythonMessage, panic_flag: PanicFlag
    ) -> None:
        return None

    async def handle_cast(
        self,
        mailbox: Mailbox,
        rank: int,
        shape: Shape,
        message: PythonMessage,
        panic_flag: PanicFlag,
    ) -> None:
        assert message.response_port is not None
        reply_port = message.response_port
        reply_port.send(mailbox, PythonMessage("echo", message.message, None, None))
        for i in range(100):
            reply_port.send(
                mailbox, PythonMessage("echo", pickle.dumps(f"msg{i}"), None, None)
            )


async def test_reducer() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)

    def my_accumulate(state: str, update: str) -> str:
        return state + update

    def my_reduce(state: str, update: str) -> str:
        return f"[reduced]({state}+{update})"

    accumulator = Accumulator("", my_accumulate, my_reduce)
    receiver: PortReceiver
    handle, receiver = proc_mesh.client.open_accum_port(accumulator)
    port_ref = handle.bind()

    actor_mesh.cast(PythonMessage("echo", pickle.dumps("start"), port_ref, None))

    messge = await asyncio.wait_for(receiver.recv(), timeout=5)
    value = cast(str, pickle.loads(messge.message))
    assert "[reduced](start+msg0)" in value
