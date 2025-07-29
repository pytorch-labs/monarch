# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import asyncio
import functools
import inspect
import logging
import os
import sys
from dataclasses import dataclass
from typing import cast, Dict, Generator, List, Tuple, Union

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._src.actor.actor_mesh import (
    _ActorMeshRefImpl,
    Actor,
    ActorMeshRef,
    DebugContext,
    MonarchContext,
)
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.pdb_wrapper import DebuggerWrite, PdbWrapper
from monarch._src.actor.sync_state import fake_sync_state


logger = logging.getLogger(__name__)

_DEBUG_MANAGER_ACTOR_NAME = "debug_manager"


async def _debugger_input(prompt=""):
    return await asyncio.to_thread(input, prompt)


def _debugger_output(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


@dataclass
class DebugSessionInfo:
    rank: int
    coords: Dict[str, int]
    hostname: str
    actor_id: ActorId
    function: str | None
    lineno: int | None


class DebugSession:
    """Represents a single session with a remote debugger."""

    def __init__(
        self, rank: int, coords: Dict[str, int], hostname: str, actor_id: ActorId
    ):
        self.rank = rank
        self.coords = coords
        self.hostname = hostname
        self.actor_id = actor_id
        self._active = False
        self._message_queue = asyncio.Queue()
        self._task = None
        self._pending_send_to_actor = asyncio.Queue()
        self._outputs_since_last_input = []
        self._function_lineno = None
        self._need_read = False

    async def _event_loop(self, line=None, suppress_output=False):
        if not suppress_output:
            # If the user had previously attached to this debug session,
            # then it would have printed various messages from the
            # message queue. When the user re-attaches, we want to
            # print out all of the output that was printed since the
            # last command sent to this session.
            for output in self._outputs_since_last_input:
                _debugger_output(output.payload.decode())

        while True:
            # When the user inputs "detach", it uses up a "read" message
            # without actually responding to the actor being debugged. We
            # can't manually reinsert the "read" message into the message queue,
            # so instead the self._need_read flag indicates there's an additional
            # "read" that we need to respond to.
            if self._need_read:
                self._need_read = False
                message = "read"
            else:
                message = await self._message_queue.get()
            if message == "detach":
                # Return to the main outer debug loop.
                break
            elif message == "read":
                break_after = False
                if line is not None:
                    break_after = True
                else:
                    line = await _debugger_input()
                if line.strip("\n") == "detach":
                    self._need_read = True
                    break
                else:
                    self._outputs_since_last_input = []
                    await self._pending_send_to_actor.put((line + "\n").encode())
                    line = None
                    if break_after:
                        break
            elif message[0] == "write":
                output = message[1]
                # If the user sees this output but then detaches from the session,
                # its useful to store all outputs since the last input so that
                # they can be printed again when the user re-attaches.
                self._outputs_since_last_input.append(output)
                if not suppress_output:
                    _debugger_output(output.payload.decode())

        if not suppress_output:
            print(
                f"Detaching from debug session for rank {self.rank} ({self.hostname})"
            )

    def get_info(self):
        function = lineno = None
        if self._function_lineno is not None:
            function, lineno = self._function_lineno
        return DebugSessionInfo(
            self.rank, self.coords, self.hostname, self.actor_id, function, lineno
        )

    async def attach(self, line=None, suppress_output=False):
        self._active = True
        if not suppress_output:
            print(f"Attached to debug session for rank {self.rank} ({self.hostname})")
        self._task = asyncio.create_task(self._event_loop(line, suppress_output))
        await self._task
        if not suppress_output:
            print(f"Detached from debug session for rank {self.rank} ({self.hostname})")
        self._active = False

    async def detach(self):
        if self._active:
            await self._message_queue.put("detach")

    async def debugger_read(self, size: int) -> DebuggerWrite:
        await self._message_queue.put("read")
        input_data = await self._pending_send_to_actor.get()
        if len(input_data) > size:
            input_data = input_data[:size]
        return DebuggerWrite(input_data, None, None)

    async def debugger_write(self, write: DebuggerWrite) -> None:
        if write.function is not None and write.lineno is not None:
            self._function_lineno = (write.function, write.lineno)
        await self._message_queue.put(("write", write))


RanksType = Union[int, List[int], range, Dict[str, Union[range, List[int], int]]]


_debug_input_parser = None


# Wrap the parser in a function so that jobs don't have to import lark
# unless they want to use the debugger.
def _get_debug_input_parser():
    global _debug_input_parser
    if _debug_input_parser is None:
        from lark import Lark

        _debug_input_parser = Lark(
            """
            rank_list: INT "," INT ("," INT)*
            start: INT?
            stop: INT?
            step: INT?
            rank_range: start ":" stop (":" step)?
            dim: CNAME "=" (rank_range | "(" rank_list ")" | INT)
            dims: dim ("," dim)*
            ranks: "ranks(" (dims | rank_range | rank_list | INT) ")"
            pdb_command: /\\w+.*/
            cast: "cast" ranks pdb_command
            help: "h" | "help"
            attach: ("a" | "attach") INT
            cont: "c" | "continue"
            quit: "q" | "quit"
            list: "l" | "list"
            command: attach | list | cast | help | cont | quit

            %import common.INT
            %import common.CNAME
            %import common.WS
            %ignore WS
            """,
            start="command",
        )
    return _debug_input_parser


_debug_input_transformer = None


# Wrap the transformer in a function so that jobs don't have to import lark
# unless they want to use the debugger.
def _get_debug_input_transformer():
    global _debug_input_transformer
    if _debug_input_transformer is None:
        from lark import Transformer
        from lark.lexer import Token

        class _IntoDebugCommandTransformer(Transformer):
            def rank_list(self, items: List[Token]) -> List[int]:
                return [int(item.value) for item in items]

            def start(self, items: List[Token]) -> int:
                if len(items) == 0:
                    return 0
                return int(items[0].value)

            def stop(self, items: List[Token]) -> int:
                if len(items) == 0:
                    return sys.maxsize
                return int(items[0].value)

            def step(self, items: List[Token]) -> int:
                if len(items) == 0:
                    return 1
                return int(items[0].value)

            def rank_range(self, items: List[int]) -> range:
                return range(*items)

            def dim(
                self, items: Tuple[Token, Union[range, List[int], Token]]
            ) -> Tuple[str, Union[range, List[int], int]]:
                if isinstance(items[1], range):
                    return (items[0].value, cast(range, items[1]))
                elif isinstance(items[1], list):
                    return (items[0].value, cast(List[int], items[1]))
                else:
                    return (items[0].value, int(cast(Token, items[1]).value))

            def dims(
                self, items: List[Tuple[str, Union[range, List[int], int]]]
            ) -> Dict[str, Union[range, List[int], int]]:
                return {dim[0]: dim[1] for dim in items}

            def ranks(self, items: List[Union[RanksType, Token]]) -> RanksType:
                if isinstance(items[0], Token):
                    return int(cast(Token, items[0]).value)
                return cast(RanksType, items[0])

            def pdb_command(self, items: List[Token]) -> str:
                return items[0].value

            def help(self, _items: List[Token]) -> "Help":
                return Help()

            def attach(self, items: List[Token]) -> "Attach":
                return Attach(int(items[0].value))

            def cont(self, _items: List[Token]) -> "Continue":
                return Continue()

            def quit(self, _items: List[Token]) -> "Quit":
                return Quit()

            def cast(self, items: Tuple[RanksType, str]) -> "Cast":
                return Cast(items[0], items[1])

            def list(self, items: List[Token]) -> "ListCommand":
                return ListCommand()

            def command(self, items: List["DebugCommand"]) -> "DebugCommand":
                return items[0]

        _debug_input_transformer = _IntoDebugCommandTransformer()
    return _debug_input_transformer


class DebugCommand:
    @staticmethod
    def parse(line: str) -> Union["DebugCommand", None]:
        try:
            tree = _get_debug_input_parser().parse(line)
            return _get_debug_input_transformer().transform(tree)
        except Exception as e:
            print(f"Error parsing input: {e}")
            return None


@dataclass
class Attach(DebugCommand):
    rank: int


@dataclass
class ListCommand(DebugCommand):
    pass


@dataclass
class Quit(DebugCommand):
    pass


@dataclass
class Help(DebugCommand):
    pass


@dataclass
class Continue(DebugCommand):
    pass


@dataclass
class Cast(DebugCommand):
    ranks: RanksType
    command: str


class DebugClient(Actor):
    """
    Single actor for both remote debuggers and users to talk to.

    Handles multiple sessions simultanesouly
    """

    def __init__(self) -> None:
        self.sessions = {}  # rank -> DebugSession

    @endpoint
    async def wait_pending_session(self):
        while len(self.sessions) == 0:
            await asyncio.sleep(1)

    @endpoint
    async def list(self) -> List[Tuple[int, Dict[str, int], str, ActorId, str, int]]:
        session_info = []
        for _, session in self.sessions.items():
            info = session.get_info()
            session_info.append(
                (
                    info.rank,
                    info.coords,
                    info.hostname,
                    info.actor_id,
                    info.function,
                    info.lineno,
                )
            )
        table_info = sorted(session_info, key=lambda r: r[0])

        from tabulate import tabulate

        print(
            tabulate(
                table_info,
                headers=[
                    "Rank",
                    "Coords",
                    "Hostname",
                    "Actor ID",
                    "Function",
                    "Line No.",
                ],
                tablefmt="grid",
            )
        )
        return table_info

    @endpoint
    async def enter(self) -> None:
        await asyncio.sleep(0.5)
        logger.info("Remote breakpoint hit. Entering monarch debugger...")
        print("\n\n************************ MONARCH DEBUGGER ************************")
        print("Enter 'help' for a list of commands.")
        print("Enter 'list' to show all active breakpoints.\n")

        while True:
            try:
                user_input = await _debugger_input("monarch_dbg> ")
                command = DebugCommand.parse(user_input)
                if isinstance(command, Help):
                    print("monarch_dbg commands:")
                    print("\tattach <rank> - attach to a debug session")
                    print("\tlist - list all debug sessions")
                    print("\tquit - exit the debugger, leaving all sessions in place")
                    print(
                        "\tcast ranks(...) <command> - send a command to a set of ranks.\n"
                        "\t\tThe value inside ranks(...) can be a single rank (ranks(1)),\n"
                        "\t\ta list of ranks (ranks(1,4,6)), a range of ranks (ranks(start?:stop?:step?)),\n"
                        "\t\tor a dict of dimensions (ranks(dim1=1:5:2,dim2=3, dim4=(3,6)))."
                    )
                    print(
                        "\tcontinue - tell all ranks to continue execution, then exit the debugger"
                    )
                    print("\thelp - print this help message")
                elif isinstance(command, Attach):
                    if command.rank not in self.sessions:
                        print(f"No debug session for rank {command.rank}")
                    else:
                        await self.sessions[command.rank].attach()
                elif isinstance(command, ListCommand):
                    # pyre-ignore
                    await self.list._method(self)
                elif isinstance(command, Continue):
                    # Clear all breakpoints and make sure all ranks have
                    # exited their debug sessions. If we sent "quit", it
                    # would raise BdbQuit, crashing the process, which
                    # probably isn't what we want.
                    await self._cast_input_and_wait("clear")
                    while len(self.sessions) > 0:
                        await self._cast_input_and_wait("c")
                    return
                elif isinstance(command, Quit):
                    return
                elif isinstance(command, Cast):
                    await self._cast_input_and_wait(command.command, command.ranks)
            except Exception as e:
                print(f"Error processing command: {e}")

    async def _cast_input_and_wait(
        self,
        command: str,
        ranks: RanksType | None = None,
    ) -> None:
        if ranks is None:
            ranks = self.sessions.keys()
        elif isinstance(ranks, dict):
            ranks = self._iter_ranks_dict(ranks)
        elif isinstance(ranks, range):
            ranks = self._iter_ranks_range(ranks)
        elif isinstance(ranks, int):
            ranks = [ranks]
        tasks = []
        for rank in ranks:
            if rank in self.sessions:
                tasks.append(
                    self.sessions[rank].attach(
                        command,
                        suppress_output=True,
                    )
                )
            else:
                print(f"No debug session for rank {rank}")
        await asyncio.gather(*tasks)

    def _iter_ranks_dict(
        self, dims: Dict[str, Union[range, List[int], int]]
    ) -> Generator[int, None, None]:
        for rank, session in self.sessions.items():
            include_rank = True
            for dim, ranks in dims.items():
                if dim not in session.coords:
                    include_rank = False
                    break
                elif (
                    isinstance(ranks, range) or isinstance(ranks, list)
                ) and session.coords[dim] not in ranks:
                    include_rank = False
                    break
                elif isinstance(ranks, int) and session.coords[dim] != ranks:
                    include_rank = False
                    break
            if include_rank:
                yield rank

    def _iter_ranks_range(self, rng: range) -> Generator[int, None, None]:
        for rank in self.sessions.keys():
            if rank in rng:
                yield rank

    ##########################################################################
    # Debugger APIs
    #
    # These endpoints are called by the remote debuggers to establish sessions
    # and communicate with them.
    @endpoint
    async def debugger_session_start(
        self, rank: int, coords: Dict[str, int], hostname: str, actor_id: ActorId
    ) -> None:
        # Create a session if it doesn't exist
        if rank not in self.sessions:
            self.sessions[rank] = DebugSession(rank, coords, hostname, actor_id)

    @endpoint
    async def debugger_session_end(self, rank: int) -> None:
        """Detach from the current debug session."""
        session = self.sessions.pop(rank)
        await session.detach()

    @endpoint
    async def debugger_read(self, rank: int, size: int) -> DebuggerWrite | str:
        """Read from the debug session for the given rank."""
        session = self.sessions[rank]

        return await session.debugger_read(size)

    @endpoint
    async def debugger_write(self, rank: int, write: DebuggerWrite) -> None:
        """Write to the debug session for the given rank."""
        session = self.sessions[rank]
        await session.debugger_write(write)


class DebugManager(Actor):
    @staticmethod
    @functools.cache
    def ref() -> "DebugManager":
        ctx = MonarchContext.get()
        return cast(
            DebugManager,
            ActorMeshRef(
                DebugManager,
                _ActorMeshRefImpl.from_actor_id(
                    ctx.mailbox,
                    ActorId.from_string(
                        f"{ctx.proc_id}.{_DEBUG_MANAGER_ACTOR_NAME}[0]"
                    ),
                ),
                ctx.mailbox,
            ),
        )

    def __init__(self, debug_client: DebugClient) -> None:
        self._debug_client = debug_client

    # pyre-ignore
    @endpoint
    def get_debug_client(self) -> DebugClient:
        return self._debug_client


def remote_breakpointhook():
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    assert frame is not None
    file = frame.f_code.co_filename
    line = frame.f_lineno
    module = frame.f_globals.get("__name__", "__main__")
    if module == "__main__" and not os.path.exists(file):
        raise NotImplementedError(
            f"Remote debugging not supported for breakpoint at {file}:{line} because "
            f"it is defined inside __main__, and the file does not exist on the host. "
            "In this case, cloudpickle serialization does not interact nicely with pdb. "
            "To debug your code, move it out of __main__ and into a module that "
            "exists on both your client and worker processes."
        )

    with fake_sync_state():
        manager = DebugManager.ref().get_debug_client.call_one().get()
    ctx = MonarchContext.get()
    pdb_wrapper = PdbWrapper(
        ctx.point.rank,
        ctx.point.shape.coordinates(ctx.point.rank),
        ctx.mailbox.actor_id,
        manager,
    )
    DebugContext.set(DebugContext(pdb_wrapper))
    pdb_wrapper.set_trace(frame)
