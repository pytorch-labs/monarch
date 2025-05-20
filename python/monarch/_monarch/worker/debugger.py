# pyre-unsafe

import bdb
import io
import logging
import pdb  # noqa
import sys
from typing import cast, Optional

from monarch._monarch import debugger

logger = logging.getLogger(__name__)


def _set_trace(*, header=None):
    ds = PdbWrapper(header)
    ds.set_trace()


class PdbWrapper(pdb.Pdb):
    def __init__(self, header: Optional[str]):
        self._actor = debugger.PdbActor()
        self.header = header
        super().__init__(
            # pyre-ignore
            stdout=WriteWrapper(self._actor),
            stdin=ReadWrapper.create(self._actor),
        )
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
            self._actor.send(debugger.DebuggerAction.Detach())
            self._actor.drain_and_stop()
            # break cycle with itself before we exit
            self.stdin = sys.stdin
            self.stdout = sys.stdout
        return r

    def set_trace(self):
        self._actor.send(debugger.DebuggerAction.Paused())
        message = self._actor.receive()
        # we give the controller the option to ignore this request to debug
        # by issuing a "detach" message immediately.
        if isinstance(message, debugger.DebuggerAction.Detach):
            return
        elif isinstance(message, debugger.DebuggerAction.Attach):
            pass
        else:
            raise RuntimeError(f"unexpected debugger message {message}")
        if self.header:
            self.message(self.header)
        super().set_trace()

    def set_quit(self):
        self._actor.send(debugger.DebuggerAction.Detach())
        self._actor.drain_and_stop()
        super().set_quit()


class ReadWrapper(io.RawIOBase):
    def __init__(self, actor: debugger.PdbActor):
        self._actor = actor

    def readinto(self, b):
        self._actor.send(debugger.DebuggerAction.Read(len(b)))
        response = self._actor.receive()
        if isinstance(response, debugger.DebuggerAction.Detach):
            raise bdb.BdbQuit
        assert isinstance(response, debugger.DebuggerAction.Write)
        response = cast(debugger.DebuggerAction.Write, response)
        payload = debugger.get_bytes_from_write_action(response)
        assert len(payload) <= len(b)
        b[: len(payload)] = payload
        return len(payload)

    def readable(self) -> bool:
        return True

    @classmethod
    def create(cls, actor: debugger.PdbActor):
        return io.TextIOWrapper(io.BufferedReader(cls(actor)))


class WriteWrapper:
    def __init__(self, actor: debugger.PdbActor):
        self._actor = actor

    def writable(self) -> bool:
        return True

    def write(self, s: str):
        self._actor.send(debugger.DebuggerAction.Write(s.encode()))

    def flush(self):
        pass
