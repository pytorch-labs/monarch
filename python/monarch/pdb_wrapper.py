# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bdb
import inspect
import io
import pdb  # noqa
import socket
import sys
from dataclasses import dataclass

from typing import Dict, TYPE_CHECKING

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId

if TYPE_CHECKING:
    from monarch.debugger import DebugClient


@dataclass
class DebuggerWrite:
    payload: bytes
    function: str | None
    lineno: int | None


class PdbWrapper(pdb.Pdb):
    def __init__(
        self,
        rank: int,
        coords: Dict[str, int],
        actor_id: ActorId,
        client_ref: "DebugClient",
        header: str | None = None,
    ):
        self.rank = rank
        self.coords = coords
        self.header = header
        self.actor_id = actor_id
        self.client_ref = client_ref
        # pyre-ignore
        super().__init__(stdout=WriteWrapper(self), stdin=ReadWrapper.create(self))
        self._first = True

    def setup(self, *args, **kwargs):
        r = super().setup(*args, **kwargs)
        if self._first:
            self._first = False
            # when we enter the debugger, we want to present the user's stack frame
            # not the nested one inside session.run. This means that the local
            # variables are what gets printed, etc. To do this
            # we first execute up 2 to get to that frame.
            self.do_up(2)
        return r

    def set_continue(self) -> None:
        r = super().set_continue()
        if not self.breaks:
            # no more breakpoints so this debugger will not
            # be used again, and we detach from the controller io.
            self.client_ref.debugger_session_end.call_one(self.rank).get()
            # break cycle with itself before we exit
            self.stdin = sys.stdin
            self.stdout = sys.stdout
        return r

    def set_trace(self):
        self.client_ref.debugger_session_start.call_one(
            self.rank, self.coords, socket.getfqdn(socket.gethostname()), self.actor_id
        ).get()
        if self.header:
            self.message(self.header)
        super().set_trace()


class ReadWrapper(io.RawIOBase):
    def __init__(self, session: "PdbWrapper"):
        self.session = session

    def readinto(self, b):
        response = self.session.client_ref.debugger_read.call_one(
            self.session.rank, len(b)
        ).get()
        if response == "detach":
            # this gets injected by the worker event loop to
            # get the worker thread to exit on an Exit command.
            raise bdb.BdbQuit
        assert isinstance(response, DebuggerWrite) and len(response.payload) <= len(b)
        b[: len(response.payload)] = response.payload
        return len(response.payload)

    def readable(self) -> bool:
        return True

    @classmethod
    def create(cls, session: "PdbWrapper"):
        return io.TextIOWrapper(io.BufferedReader(cls(session)))


class WriteWrapper:
    def __init__(self, session: "PdbWrapper"):
        self.session = session

    def writable(self) -> bool:
        return True

    def write(self, s: str):
        function = None
        lineno = None
        if self.session.curframe is not None:
            # pyre-ignore
            function = f"{inspect.getmodulename(self.session.curframe.f_code.co_filename)}.{self.session.curframe.f_code.co_name}"
            # pyre-ignore
            lineno = self.session.curframe.f_lineno
        self.session.client_ref.debugger_write.call_one(
            self.session.rank,
            DebuggerWrite(
                s.encode(),
                function,
                lineno,
            ),
        ).get()

    def flush(self):
        pass


def remote_breakpointhook(
    rank: int, coords: Dict[str, int], actor_id: ActorId, client_ref: "DebugClient"
):
    ds = PdbWrapper(rank, coords, actor_id, client_ref)
    ds.set_trace()
