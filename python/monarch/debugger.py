# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch.actor_mesh import Actor, endpoint

from monarch.pdb_wrapper import DebuggerWrite

from monarch.proc_mesh import local_proc_mesh
from tabulate import tabulate


logger = logging.getLogger(__name__)


CANCEL_TOKEN = object()


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


class DebugCommand:
    @staticmethod
    def parse(line: str) -> Union["DebugCommand", None]:
        parts = line.strip("\n").split(" ")
        if len(parts) == 0:
            return None
        command = parts[0]
        match command:
            case "attach":
                return Attach._parse(parts)
            case "list":
                return ListCommand()
            case "quit":
                return Quit()
            case "cast":
                return Cast._parse(parts)
            case "help":
                return Help()
            case "continue":
                return Continue()
            case _:
                print(
                    f"Unknown command {command}. Expected: attach | list | quit | cast | continue | help"
                )
                return None


@dataclass
class Attach(DebugCommand):
    rank: int

    @classmethod
    def _parse(cls, parts: List[str]) -> "Attach":
        if len(parts) != 2:
            raise ValueError("Invalid attach command. Expected: attach <rank>")
        try:
            rank = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid rank {parts[1]}. Expected: int")
        return cls(rank)


class ListCommand(DebugCommand):
    pass


class Quit(DebugCommand):
    pass


class Help(DebugCommand):
    pass


class Continue(DebugCommand):
    pass


@dataclass
class Cast(DebugCommand):
    ranks: List[int] | None
    command: str

    @classmethod
    def _parse(cls, parts: List[str]) -> "Cast":
        if len(parts) < 3:
            raise ValueError(
                "Invalid cast command. Expected: cast {<r0,r1,...> | *} <command>"
            )
        str_ranks = parts[1]
        command = " ".join(parts[2:])
        if str_ranks == "*":
            return cls(None, command)
        else:
            str_ranks = str_ranks.split(",")
            if len(str_ranks) == 0:
                raise ValueError(
                    "Invalid rank list for cast. Expected at least one rank."
                )
            ranks = []
            for rank in str_ranks:
                try:
                    ranks.append(int(rank))
                except ValueError:
                    raise ValueError(f"Invalid rank {rank}. Expected: int")
            return cls(ranks, command)


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
        table_data = []
        for _, session in self.sessions.items():
            info = session.get_info()
            table_data.append(
                (
                    info.rank,
                    info.coords,
                    info.hostname,
                    info.actor_id,
                    info.function,
                    info.lineno,
                )
            )
        table_data = sorted(table_data, key=lambda r: r[0])

        headers = ["Rank", "Coords", "Hostname", "Actor ID", "Function", "Line No."]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        return table_data

    @endpoint
    async def enter(self) -> None:
        # pyre-ignore
        await getattr(self, "list")._method(self)  # noqa

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
                        "\tcast {<r0,r1,...> | *} <command> - send a command to a comma-separated list of ranks, or all ranks"
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
                    await getattr(self, "list")._method(self)  # noqa
                elif isinstance(command, Continue):
                    # Make sure all ranks have exited their debug sessions.
                    # If we sent "quit", it would raise BdbQuit, crashing
                    # the process, which probably isn't what we want.
                    while len(self.sessions) > 0:
                        tasks = []
                        for rank in self.sessions:
                            tasks.append(
                                self.sessions[rank].attach("c", suppress_output=True)
                            )
                        await asyncio.gather(*tasks)
                    return
                elif isinstance(command, Quit):
                    return
                elif isinstance(command, Cast):
                    if command.ranks is None:
                        ranks = self.sessions.keys()
                    else:
                        ranks = command.ranks
                    tasks = []
                    for rank in ranks:
                        if rank in self.sessions:
                            tasks.append(
                                self.sessions[rank].attach(
                                    command.command,
                                    suppress_output=True,
                                )
                            )
                        else:
                            print(f"No debug session for rank {rank}")
                    await asyncio.gather(*tasks)
            except Exception as e:
                print(f"Error processing command: {e}")

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


async def init_debugging(actor_mesh: Actor) -> DebugClient:
    debugger_proc_mesh = await local_proc_mesh(gpus=1, hosts=1)
    debug_client_mesh = await debugger_proc_mesh.spawn("debug_client", DebugClient)
    await actor_mesh._set_debug_client.call(debug_client_mesh)
    return debug_client_mesh
