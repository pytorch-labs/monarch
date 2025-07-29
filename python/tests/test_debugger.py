# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import asyncio
import re
import sys
from typing import cast, List
from unittest.mock import AsyncMock, patch

import monarch
import monarch.actor as actor

import pytest

import torch

from monarch._src.actor.actor_mesh import Actor, ActorError, current_rank
from monarch._src.actor.debugger import (
    Attach,
    Cast,
    Continue,
    DebugCommand,
    DebugSession,
    DebugSessionInfo,
    DebugSessions,
    Help,
    ListCommand,
    Quit,
)
from monarch._src.actor.endpoint import endpoint

from monarch._src.actor.proc_mesh import proc_mesh

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _bad_rank():
    raise ValueError("bad rank")


def _debugee_actor_internal(rank):
    if rank == 0:
        breakpoint()  # noqa
        rank += 1
        rank += 1
        return rank
    elif rank == 1:
        breakpoint()  # noqa
        rank += 2
        rank += 2
        return rank
    elif rank == 2:
        breakpoint()  # noqa
        rank += 3
        rank += 3
        _bad_rank()
    elif rank == 3:
        breakpoint()  # noqa
        rank += 4
        rank += 4
        return rank


class DebugeeActor(Actor):
    @endpoint
    async def to_debug(self):
        rank = current_rank().rank
        return _debugee_actor_internal(rank)


async def _wait_for_breakpoints(debug_client, n_breakpoints) -> List[DebugSessionInfo]:
    breakpoints: List[DebugSessionInfo] = []
    for i in range(10):
        breakpoints = await debug_client.list.call_one()
        if len(breakpoints) == n_breakpoints:
            break
        await asyncio.sleep(1)
        if i == 9:
            raise RuntimeError("timed out waiting for breakpoints")
    return breakpoints


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs, this test requires at least 2 GPUs",
)
async def test_debug() -> None:
    input_mock = AsyncMock()
    input_mock.side_effect = [
        "attach debugee 1",
        "n",
        "n",
        "n",
        "n",
        "detach",
        "attach debugee 1",
        "detach",
        "quit",
        "cast debugee ranks(0,3) n",
        "cast debugee ranks(0,3) n",
        # Attaching to 0 and 3 ensures that when we call "list"
        # the next time, their function/lineno info will be
        # up-to-date.
        "attach debugee 0",
        "detach",
        "attach debugee 3",
        "detach",
        "quit",
        "attach debugee 2",
        "c",
        "detach",
        "quit",
        "attach debugee 2",
        "bt",
        "c",
        "quit",
        "continue",
    ]

    outputs = []

    def _patch_output(msg):
        nonlocal outputs
        outputs.append(msg)

    with patch(
        "monarch._src.actor.debugger._debugger_input", side_effect=input_mock
    ), patch("monarch._src.actor.debugger._debugger_output", new=_patch_output):
        proc = await proc_mesh(hosts=2, gpus=2)
        debugee = await proc.spawn("debugee", DebugeeActor)
        debug_client = actor.debug_client()

        fut = debugee.to_debug.call()
        await debug_client.wait_pending_session.call_one()
        breakpoints = await _wait_for_breakpoints(debug_client, 4)

        initial_linenos = {}
        for i in range(len(breakpoints)):
            info = breakpoints[i]
            initial_linenos[info.rank] = info.lineno
            assert info.rank == i
            assert info.coords == {"hosts": info.rank // 2, "gpus": info.rank % 2}
            assert info.function == "test_debugger._debugee_actor_internal"
            assert info.lineno == cast(int, breakpoints[0].lineno) + 5 * info.rank

        await debug_client.enter.call_one()

        # Check that when detaching and re-attaching to a session, the last portion of the output is repeated
        expected_last_output = [
            r"--Return--",
            r"\n",
            r"> (/.*/)+test_debugger.py\(\d+\)to_debug\(\)->5\n-> return _debugee_actor_internal\(rank\)",
            r"\n",
            r"\(Pdb\) ",
        ]
        output_len = len(expected_last_output)
        assert outputs[-2 * output_len : -output_len] == outputs[-output_len:]
        for real_output, expected_output in zip(
            outputs[-output_len:], expected_last_output
        ):
            assert re.match(expected_output, real_output) is not None

        breakpoints = await debug_client.list.call_one()
        for i in range(len(breakpoints)):
            if i == 1:
                assert breakpoints[i].function == "test_debugger.to_debug"
            else:
                assert (
                    breakpoints[i].function == "test_debugger._debugee_actor_internal"
                )
                assert breakpoints[i].lineno == initial_linenos[i]

        await debug_client.enter.call_one()

        breakpoints = await debug_client.list.call_one()
        for i in range(len(breakpoints)):
            if i == 1:
                assert breakpoints[i].function == "test_debugger.to_debug"
            elif i in (0, 3):
                assert (
                    breakpoints[i].function == "test_debugger._debugee_actor_internal"
                )
                assert breakpoints[i].lineno == initial_linenos[i] + 2
            else:
                assert (
                    breakpoints[i].function == "test_debugger._debugee_actor_internal"
                )
                assert breakpoints[i].lineno == initial_linenos[i]

        await debug_client.enter.call_one()

        breakpoints = await debug_client.list.call_one()
        assert len(breakpoints) == 4
        # Expect post-mortem debugging for rank 2
        assert breakpoints[2].function == "test_debugger._bad_rank"

        await debug_client.enter.call_one()

        expected_last_output = [
            r"\s*(/.*/)+test_debugger.py\(\d+\)_debugee_actor_internal\(\)\n-> _bad_rank\(\)",
            r"\n",
            r'> (/.*/)+test_debugger.py\(\d+\)_bad_rank\(\)\n-> raise ValueError\("bad rank"\)',
            r"\n",
            r"\(Pdb\) ",
        ]

        for output, expected_output in zip(
            outputs[-len(expected_last_output) :], expected_last_output
        ):
            assert re.match(expected_output, output) is not None

        breakpoints = await debug_client.list.call_one()
        assert len(breakpoints) == 3
        for i, rank in enumerate((0, 1, 3)):
            assert breakpoints[i].rank == rank

        await debug_client.enter.call_one()
        breakpoints = await debug_client.list.call_one()
        assert len(breakpoints) == 0

        with pytest.raises(
            monarch._src.actor.actor_mesh.ActorError, match="ValueError: bad rank"
        ):
            await fut


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs, this test requires at least 2 GPUs",
)
async def test_debug_multi_actor() -> None:
    input_mock = AsyncMock()
    input_mock.side_effect = [
        "attach debugee_2 2",
        "n",
        "detach",
        "attach debugee_1 1",
        "n",
        "detach",
        "quit",
        "cast debugee_1 ranks(:) c",
        "cast debugee_2 ranks(:) c",
        "attach debugee_2 2",
        "c",
        "quit",
        "continue",
    ]

    with patch("monarch._src.actor.debugger._debugger_input", side_effect=input_mock):
        proc = await proc_mesh(hosts=2, gpus=2)
        debugee_1 = await proc.spawn("debugee_1", DebugeeActor)
        debugee_2 = await proc.spawn("debugee_2", DebugeeActor)
        debug_client = actor.debug_client()

        fut_1 = debugee_1.to_debug.call()
        fut_2 = debugee_2.to_debug.call()
        await debug_client.wait_pending_session.call_one()

        breakpoints = await _wait_for_breakpoints(debug_client, 8)

        initial_linenos = {}
        for i in range(len(breakpoints)):
            info = breakpoints[i]
            initial_linenos[info.rank] = info.lineno
            assert info.rank == i % 4
            assert info.actor_name == "debugee_1" if i < 4 else "debugee_2"
            assert info.coords == {"hosts": info.rank // 2, "gpus": info.rank % 2}
            assert info.function == "test_debugger._debugee_actor_internal"
            assert info.lineno == cast(int, breakpoints[0].lineno) + 5 * info.rank

        await debug_client.enter.call_one()

        breakpoints = await _wait_for_breakpoints(debug_client, 8)
        for i in range(len(breakpoints)):
            if i == 1:
                assert breakpoints[i].actor_name == "debugee_1"
                assert breakpoints[i].rank == 1
                assert breakpoints[i].lineno == initial_linenos[breakpoints[i].rank] + 1
            elif i == 6:
                assert breakpoints[i].actor_name == "debugee_2"
                assert breakpoints[i].rank == 2
                assert breakpoints[i].lineno == initial_linenos[breakpoints[i].rank] + 1
            else:
                assert (
                    breakpoints[i].actor_name == "debugee_1" if i < 4 else "debugee_2"
                )
                assert breakpoints[i].rank == i % 4
                assert breakpoints[i].lineno == initial_linenos[breakpoints[i].rank]

        await debug_client.enter.call_one()

        breakpoints = await _wait_for_breakpoints(debug_client, 1)
        with pytest.raises(ActorError, match="ValueError: bad rank"):
            await fut_2
        assert breakpoints[0].actor_name == "debugee_1"
        assert breakpoints[0].rank == 2
        assert breakpoints[0].function == "test_debugger._bad_rank"

        await debug_client.enter.call_one()

        breakpoints = await _wait_for_breakpoints(debug_client, 0)
        with pytest.raises(ActorError, match="ValueError: bad rank"):
            await fut_1


async def test_debug_sessions_insert_get_remove() -> None:
    mock_sessions = []
    for actor_name in ("actor_a", "actor_b"):
        for rank in range(2):
            mock_session = DebugSession(rank, {}, "", actor_name)
            mock_sessions.append(mock_session)

    debug_sessions = DebugSessions()

    with pytest.raises(ValueError, match="No debug sessions for actor actor_a"):
        debug_sessions.get("actor_a", 0)
    debug_sessions.insert(mock_sessions[0])
    assert debug_sessions.get("actor_a", 0) is mock_sessions[0]
    assert ("actor_a", 0) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 0 already exists for actor actor_a"
    ):
        debug_sessions.insert(mock_sessions[0])

    with pytest.raises(
        ValueError, match="No debug session for rank 1 for actor actor_a"
    ):
        debug_sessions.get("actor_a", 1)
    debug_sessions.insert(mock_sessions[1])
    assert debug_sessions.get("actor_a", 1) is mock_sessions[1]
    assert ("actor_a", 1) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 1 already exists for actor actor_a"
    ):
        debug_sessions.insert(mock_sessions[1])

    with pytest.raises(ValueError, match="No debug sessions for actor actor_b"):
        debug_sessions.get("actor_b", 0)
    debug_sessions.insert(mock_sessions[2])
    assert debug_sessions.get("actor_b", 0) is mock_sessions[2]
    assert ("actor_b", 0) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 0 already exists for actor actor_b"
    ):
        debug_sessions.insert(mock_sessions[2])

    with pytest.raises(
        ValueError, match="No debug session for rank 1 for actor actor_b"
    ):
        debug_sessions.get("actor_b", 1)
    debug_sessions.insert(mock_sessions[3])
    assert debug_sessions.get("actor_b", 1) is mock_sessions[3]
    assert ("actor_b", 1) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 1 already exists for actor actor_b"
    ):
        debug_sessions.insert(mock_sessions[3])

    assert len(debug_sessions) == 4

    assert debug_sessions.remove("actor_a", 0) is mock_sessions[0]
    assert len(debug_sessions) == 3
    assert ("actor_a", 0) not in debug_sessions
    with pytest.raises(
        ValueError, match="No debug session for rank 0 for actor actor_a"
    ):
        debug_sessions.remove("actor_a", 0)

    assert debug_sessions.remove("actor_a", 1) is mock_sessions[1]
    assert len(debug_sessions) == 2
    assert ("actor_a", 1) not in debug_sessions
    with pytest.raises(ValueError, match="No debug sessions for actor actor_a"):
        debug_sessions.remove("actor_a", 1)

    assert debug_sessions.remove("actor_b", 0) is mock_sessions[2]
    assert len(debug_sessions) == 1
    assert ("actor_b", 0) not in debug_sessions
    with pytest.raises(
        ValueError, match="No debug session for rank 0 for actor actor_b"
    ):
        debug_sessions.remove("actor_b", 0)

    assert debug_sessions.remove("actor_b", 1) is mock_sessions[3]
    assert len(debug_sessions) == 0
    assert ("actor_b", 1) not in debug_sessions
    with pytest.raises(ValueError, match="No debug sessions for actor actor_b"):
        debug_sessions.remove("actor_b", 1)


async def test_debug_sessions_iter() -> None:
    debug_sessions = DebugSessions()
    mock_sessions = []

    for actor_name in ("actor_a", "actor_b"):
        for host in range(3):
            for gpu in range(8):
                rank = host * 8 + gpu
                mock_session = DebugSession(
                    rank, {"hosts": host, "gpus": gpu}, "", actor_name
                )
                mock_sessions.append(mock_session)
                debug_sessions.insert(mock_session)

    # Single rank
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = list(debug_sessions.iter((actor_name, 2)))
        assert len(sessions) == 1
        assert sessions[0] is mock_sessions[i * 24 + 2]

    # List of ranks
    ranks = [1, 3, 5]
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, ranks)), key=lambda s: s.get_info()
        )
        assert len(sessions) == 3
        for j in range(3):
            assert sessions[j] is mock_sessions[i * 24 + ranks[j]]

    # Range of ranks
    ranks = range(2, 24, 3)
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, ranks)), key=lambda s: s.get_info()
        )
        ranks = list(ranks)
        assert len(sessions) == len(ranks)
        for j in range(len(ranks)):
            assert sessions[j] is mock_sessions[i * 24 + ranks[j]]

    # All ranks
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, None)), key=lambda s: s.get_info()
        )
        assert len(sessions) == 24
        for j in range(24):
            assert sessions[j] is mock_sessions[i * 24 + j]

    # All ranks, all actors
    sessions = sorted(debug_sessions.iter(None), key=lambda s: s.get_info())
    assert len(sessions) == 48
    for i in range(48):
        assert sessions[i] is mock_sessions[i]

    # Dimension filtering with a single value
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"hosts": 1})), key=lambda s: s.get_info()
        )
        assert len(sessions) == 8
        for j in range(8):
            assert sessions[j] is mock_sessions[i * 24 + 8 + j]

    # Dimension filtering with a list
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"hosts": [0, 2]})),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 16
        j = 0
        for host in (0, 2):
            for gpu in range(8):
                assert sessions[j] is mock_sessions[i * 24 + host * 8 + gpu]
                j += 1

    # Dimension filtering with a range
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"gpus": range(5, 8)})),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 9
        j = 0
        for host in range(3):
            for gpu in range(5, 8):
                assert sessions[j] is mock_sessions[i * 24 + host * 8 + gpu]
                j += 1

    # Multiple dimension filters
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter(
                (actor_name, {"hosts": [1, 3], "gpus": range(0, sys.maxsize, 3)})
            ),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 3
        j = 0
        for gpu in range(0, 8, 3):
            assert sessions[j] is mock_sessions[i * 24 + 8 + gpu]
            j += 1

    # Non-existent dimension
    for actor_name in ("actor_a", "actor_b"):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"hosts": 0, "gpus": 0, "foo": 0})),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 0


@pytest.mark.parametrize(
    ["user_input", "expected_output"],
    [
        ("attach debugee 1", Attach("debugee", 1)),
        ("a my_awesome_actor 100", Attach("my_awesome_actor", 100)),
        ("list", ListCommand()),
        ("l", ListCommand()),
        ("help", Help()),
        ("h", Help()),
        ("quit", Quit()),
        ("q", Quit()),
        ("continue", Continue()),
        ("c", Continue()),
        (
            "cast debugee ranks(123) b 25",
            Cast(actor_name="debugee", ranks=123, command="b 25"),
        ),
        (
            "cast my_awesome_actor ranks(12,34,56) b 25",
            Cast(actor_name="my_awesome_actor", ranks=[12, 34, 56], command="b 25"),
        ),
        (
            "cast debugee ranks(:) b 25",
            Cast(actor_name="debugee", ranks=range(0, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(:123) b 25",
            Cast(actor_name="debugee", ranks=range(0, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(123:) b 25",
            Cast(actor_name="debugee", ranks=range(123, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(123:456) b 25",
            Cast(actor_name="debugee", ranks=range(123, 456), command="b 25"),
        ),
        (
            "cast debugee ranks(::) b 25",
            Cast(actor_name="debugee", ranks=range(0, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(::123) b 25",
            Cast(
                actor_name="debugee", ranks=range(0, sys.maxsize, 123), command="b 25"
            ),
        ),
        (
            "cast debugee ranks(123::) b 25",
            Cast(actor_name="debugee", ranks=range(123, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(:123:) b 25",
            Cast(actor_name="debugee", ranks=range(0, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(:456:123) b 25",
            Cast(actor_name="debugee", ranks=range(0, 456, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(456::123) b 25",
            Cast(
                actor_name="debugee", ranks=range(456, sys.maxsize, 123), command="b 25"
            ),
        ),
        (
            "cast debugee ranks(123:456:) b 25",
            Cast(actor_name="debugee", ranks=range(123, 456), command="b 25"),
        ),
        (
            "cast debugee ranks(456:789:123) b 25",
            Cast(actor_name="debugee", ranks=range(456, 789, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(dim1=123) up 2",
            Cast(actor_name="debugee", ranks={"dim1": 123}, command="up 2"),
        ),
        (
            "cast debugee ranks(dim1=123, dim2=(12,34,56), dim3=15::2) up 2",
            Cast(
                actor_name="debugee",
                ranks={
                    "dim1": 123,
                    "dim2": [12, 34, 56],
                    "dim3": range(15, sys.maxsize, 2),
                },
                command="up 2",
            ),
        ),
    ],
)
async def test_debug_command_parser_valid_inputs(user_input, expected_output):
    assert DebugCommand.parse(user_input) == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "a",
        "attach",
        "a actor",
        "attach actor",
        "attacha actor 1" "attch actor 1",
        "attach actor 1abc",
        "attach actor 1 a",
        "cast ranks(123) b 25",
        "cast   ranks(123) b 25",
        "castactor ranks(123) b 25",
        "cast actor rnks(123) b 25",
        "cast actor ranks() b 25",
        "cast actor ranks(1ab) b 25",
        "cast actor ranks(1,a,3) b 25",
        "cast actor ranks(a:2:4) b 25",
        "cast actor ranks(1,2,3",
        "cast actor ranks(1,2,3)) b 25",
        "cast actor ranks(1,) b 25",
        "cast actor ranks(1,2,) b 25",
        "cast actor ranks(,1,2) b 25",
        "cast actor ranks(1,,2) b 25",
        "cast actor ranks(:::) b 25",
        "cast actor ranks(:123::) b 25",
        "cast actor ranks(1:2:3,4) b 25",
        "cast actor ranks(dim1=) b 25",
        "cast actor ranks(dim1=123, dim2=) b 25",
        "cast actor ranks(dim1=123, dim2=(12,34,56) b 25",
        "cast actor ranks(dim1=123, dim2=(,12,34,56) b 25",
        "cast actor ranks(dim1=123, dim2=(12,,34,56) b 25",
        "cast actor ranks(dim1=123, dim2=(12,34,56), dim3=15::2 b 25",
        "cast actor ranks(dim1=123,) b 25",
    ],
)
async def test_debug_command_parser_invalid_inputs(invalid_input):
    assert DebugCommand.parse(invalid_input) is None
