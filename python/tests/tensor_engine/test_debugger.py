# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import asyncio
import re
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import monarch
import monarch.actor as actor

import pytest

import torch

from monarch._src.actor.actor_mesh import Actor, current_rank
from monarch._src.actor.debugger import (
    Attach,
    Cast,
    Continue,
    DebugClient,
    DebugCommand,
    DebugSession,
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


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs, this test requires at least 2 GPUs",
)
async def test_debug() -> None:
    input_mock = AsyncMock()
    input_mock.side_effect = [
        "attach 1",
        "n",
        "n",
        "n",
        "n",
        "detach",
        "attach 1",
        "detach",
        "quit",
        "cast ranks(0,3) n",
        "cast ranks(0,3) n",
        # Attaching to 0 and 3 ensures that when we call "list"
        # the next time, their function/lineno info will be
        # up-to-date.
        "attach 0",
        "detach",
        "attach 3",
        "detach",
        "quit",
        "attach 2",
        "c",
        "detach",
        "quit",
        "attach 2",
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
        breakpoints = []
        for i in range(10):
            breakpoints = await debug_client.list.call_one()
            if len(breakpoints) == 4:
                break
            await asyncio.sleep(1)
            if i == 9:
                raise RuntimeError("timed out waiting for breakpoints")

        initial_linenos = {}
        for i in range(len(breakpoints)):
            rank, coords, _, _, function, lineno = breakpoints[i]
            initial_linenos[rank] = lineno
            assert rank == i
            assert coords == {"hosts": rank // 2, "gpus": rank % 2}
            assert function == "test_debugger._debugee_actor_internal"
            assert lineno == breakpoints[0][5] + 5 * rank

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
                assert breakpoints[i][4] == "test_debugger.to_debug"
            else:
                assert breakpoints[i][4] == "test_debugger._debugee_actor_internal"
                assert breakpoints[i][5] == initial_linenos[i]

        await debug_client.enter.call_one()

        breakpoints = await debug_client.list.call_one()
        for i in range(len(breakpoints)):
            if i == 1:
                assert breakpoints[i][4] == "test_debugger.to_debug"
            elif i in (0, 3):
                assert breakpoints[i][4] == "test_debugger._debugee_actor_internal"
                assert breakpoints[i][5] == initial_linenos[i] + 2
            else:
                assert breakpoints[i][4] == "test_debugger._debugee_actor_internal"
                assert breakpoints[i][5] == initial_linenos[i]

        await debug_client.enter.call_one()

        breakpoints = await debug_client.list.call_one()
        assert len(breakpoints) == 4
        # Expect post-mortem debugging for rank 2
        assert breakpoints[2][4] == "test_debugger._bad_rank"

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
            assert breakpoints[i][0] == rank

        await debug_client.enter.call_one()
        breakpoints = await debug_client.list.call_one()
        assert len(breakpoints) == 0

        with pytest.raises(
            monarch._src.actor.actor_mesh.ActorError, match="ValueError: bad rank"
        ):
            await fut


async def test_cast_input_and_wait() -> None:
    debug_client = DebugClient()

    mock_sessions = {}
    for host in range(3):
        for gpu in range(8):
            rank = host * 8 + gpu
            mock_session = MagicMock(spec=DebugSession)
            mock_session.attach = AsyncMock()
            mock_session.rank = rank
            mock_session.coords = {"hosts": host, "gpus": gpu}
            mock_sessions[rank] = mock_session

    debug_client.sessions = mock_sessions

    # Cast to a single rank
    await debug_client._cast_input_and_wait("n", 2)
    mock_sessions[2].attach.assert_called_once_with("n", suppress_output=True)
    for rank, session in mock_sessions.items():
        if rank != 2:
            session.attach.assert_not_called()

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast to a list of ranks
    ranks = [1, 3, 5]
    await debug_client._cast_input_and_wait("n", ranks)
    for rank in ranks:
        mock_sessions[rank].attach.assert_called_once_with("n", suppress_output=True)
    for rank, session in mock_sessions.items():
        if rank not in ranks:
            session.attach.assert_not_called()

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast to a range of ranks
    ranks = range(2, 24, 3)
    await debug_client._cast_input_and_wait("n", ranks)
    for rank in ranks:
        mock_sessions[rank].attach.assert_called_once_with("n", suppress_output=True)
    for rank, session in mock_sessions.items():
        if rank not in ranks:
            session.attach.assert_not_called()

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast to all ranks
    await debug_client._cast_input_and_wait("n", None)
    for session in mock_sessions.values():
        session.attach.assert_called_once_with("n", suppress_output=True)

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast using dimension filtering with a single value
    await debug_client._cast_input_and_wait("n", {"hosts": 1})
    for session in mock_sessions.values():
        if session.coords["hosts"] == 1:
            session.attach.assert_called_once_with("n", suppress_output=True)
        else:
            session.attach.assert_not_called()

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast using dimension filtering with a list
    await debug_client._cast_input_and_wait("n", {"hosts": [0, 2]})
    for _rank, session in mock_sessions.items():
        if session.coords["hosts"] in [0, 2]:
            session.attach.assert_called_once_with("n", suppress_output=True)
        else:
            session.attach.assert_not_called()

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast using dimension filtering with a range
    await debug_client._cast_input_and_wait("n", {"gpus": range(5, 8)})
    for session in mock_sessions.values():
        if session.coords["gpus"] in range(5, 8):
            session.attach.assert_called_once_with("n", suppress_output=True)
        else:
            session.attach.assert_not_called()

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast using multiple dimension filters
    await debug_client._cast_input_and_wait(
        "n", {"hosts": [1, 3], "gpus": range(0, sys.maxsize, 3)}
    )
    for session in mock_sessions.values():
        if session.coords["hosts"] in [1, 3] and session.coords["gpus"] in range(
            0, sys.maxsize, 3
        ):
            session.attach.assert_called_once_with("n", suppress_output=True)
        else:
            session.attach.assert_not_called()

    for session in mock_sessions.values():
        session.attach.reset_mock()

    # Cast with non-existent dimension
    await debug_client._cast_input_and_wait("n", {"hosts": 0, "gpus": 0, "foo": 0})
    for session in mock_sessions.values():
        session.attach.assert_not_called()


@pytest.mark.parametrize(
    ["user_input", "expected_output"],
    [
        ("attach 1", Attach(1)),
        ("a 100", Attach(100)),
        ("list", ListCommand()),
        ("l", ListCommand()),
        ("help", Help()),
        ("h", Help()),
        ("quit", Quit()),
        ("q", Quit()),
        ("continue", Continue()),
        ("c", Continue()),
        ("cast ranks(123) b 25", Cast(ranks=123, command="b 25")),
        ("cast ranks(12,34,56) b 25", Cast(ranks=[12, 34, 56], command="b 25")),
        ("cast ranks(:) b 25", Cast(ranks=range(0, sys.maxsize), command="b 25")),
        ("cast ranks(:123) b 25", Cast(ranks=range(0, 123), command="b 25")),
        ("cast ranks(123:) b 25", Cast(ranks=range(123, sys.maxsize), command="b 25")),
        ("cast ranks(123:456) b 25", Cast(ranks=range(123, 456), command="b 25")),
        ("cast ranks(::) b 25", Cast(ranks=range(0, sys.maxsize), command="b 25")),
        (
            "cast ranks(::123) b 25",
            Cast(ranks=range(0, sys.maxsize, 123), command="b 25"),
        ),
        ("cast ranks(123::) b 25", Cast(ranks=range(123, sys.maxsize), command="b 25")),
        ("cast ranks(:123:) b 25", Cast(ranks=range(0, 123), command="b 25")),
        ("cast ranks(:456:123) b 25", Cast(ranks=range(0, 456, 123), command="b 25")),
        (
            "cast ranks(456::123) b 25",
            Cast(ranks=range(456, sys.maxsize, 123), command="b 25"),
        ),
        ("cast ranks(123:456:) b 25", Cast(ranks=range(123, 456), command="b 25")),
        (
            "cast ranks(456:789:123) b 25",
            Cast(ranks=range(456, 789, 123), command="b 25"),
        ),
        ("cast ranks(dim1=123) up 2", Cast(ranks={"dim1": 123}, command="up 2")),
        (
            "cast ranks(dim1=123, dim2=(12,34,56), dim3=15::2) up 2",
            Cast(
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
        "attch 1",
        "attach",
        "cast rnks(123) b 25",
        "cast ranks() b 25",
        "cast ranks(1ab) b 25",
        "cast ranks(1,a,3) b 25",
        "cast ranks(a:2:4) b 25",
        "cast ranks(1,2,3",
        "cast ranks(1,2,3)) b 25",
        "cast ranks(1,) b 25",
        "cast ranks(1,2,) b 25",
        "cast ranks(,1,2) b 25",
        "cast ranks(1,,2) b 25",
        "cast ranks(:::) b 25",
        "cast ranks(:123::) b 25",
        "cast ranks(1:2:3,4) b 25",
        "cast ranks(dim1=) b 25",
        "cast ranks(dim1=123, dim2=) b 25",
        "cast ranks(dim1=123, dim2=(12,34,56) b 25",
        "cast ranks(dim1=123, dim2=(,12,34,56) b 25",
        "cast ranks(dim1=123, dim2=(12,,34,56) b 25",
        "cast ranks(dim1=123, dim2=(12,34,56), dim3=15::2 b 25",
        "cast ranks(dim1=123,) b 25",
    ],
)
async def test_debug_command_parser_invalid_inputs(invalid_input):
    assert DebugCommand.parse(invalid_input) is None
