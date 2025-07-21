# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is the main function for the worker / pipe processes. It expects the args to
the process to be passed in on the command line and accessible in `sys.argv`.

To see the supported arguments checkout `monarch_tensor_worker::bootstrap`.
"""

# pyre-unsafe

import bdb

import importlib.resources
import io

import logging
import os

import pdb  # noqa  # noqa
import socket
import sys
from pathlib import Path
from typing import cast, Optional

from monarch._rust_bindings.monarch_extension import debugger
from monarch._rust_bindings.monarch_messages.debugger import DebuggerAction

logger = logging.getLogger(__name__)


def _handle_unhandled_exception(*args):
    logger.error("Uncaught exception", exc_info=args)


_glog_level_to_abbr = {
    "DEBUG": "V",  # V is for VERBOSE in glog
    "INFO": "I",
    "WARNING": "W",
    "ERROR": "E",
    "CRITICAL": "C",
}


def fix_exception_lines(tb_lines):
    formatted_lines = []
    for line in tb_lines:
        # Replace the standard file and line format with the custom format
        if line.startswith("  File"):
            # Extract the filename and line number
            parts = line.split(",")
            file_info = parts[0].strip()[6:-1]  # Remove '  File "' and '"'
            line_info = parts[1].strip()[5:]  # Remove 'line '
            new_line = f"  File {file_info}:{line_info}"
            if len(parts) > 2:
                new_line += ", " + ",".join(parts[2:]).strip()
            formatted_lines.append(new_line)
        else:
            formatted_lines.append(line.strip())
    return formatted_lines


class _Formatter(logging.Formatter):
    def __init__(self, suffix):
        self.suffix = suffix

    def format(self, record):
        message = record.getMessage()
        asctime = self.formatTime(record, "%m%d %H:%M:%S")

        lines = message.strip().split("\n")
        if record.exc_info:
            exc_info = fix_exception_lines(
                self.formatException(record.exc_info).split("\n")
            )
            lines.extend(exc_info)
        if record.stack_info:
            stack_info = self.formatStack(record.stack_info)
            lines.extend(stack_info.strip().split("\n"))

        shortlevel = _glog_level_to_abbr.get(record.levelname, record.levelname[0])

        prefix = (
            f"{shortlevel}{asctime}.{int(record.msecs*1000):06d} "
            f"{record.filename}:"
            f"{record.lineno}]{self.suffix}"
        )
        return "\n".join(f"{prefix} {line}" for line in lines)


def initialize_logging(process_name=None):
    log_folder = os.environ.get("TORCH_MONARCH_LOG_FOLDER")
    log_level = os.environ.get("TORCH_MONARCH_LOG_LEVEL", "INFO")
    suffix = "" if process_name is None else f" {process_name}:"
    handler = None
    if log_folder is not None:
        log_folder_path = Path(log_folder)
        log_folder_path.mkdir(parents=True, exist_ok=True)
        safe_process_name = (
            process_name.replace("/", "_") if process_name else "logfile.log"
        )
        log_file_name = f"{safe_process_name}.log"
        log_file_path = log_folder_path / log_file_name
        handler = logging.FileHandler(log_file_path)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(_Formatter(suffix))
    handler.setLevel(log_level)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)
    sys.excepthook = _handle_unhandled_exception


def gethostname():
    """Get the hostname of the machine."""
    hostname = socket.gethostname()
    hostname = hostname.replace(".facebook.com", "")
    return hostname


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
            self._actor.send(DebuggerAction.Detach())
            self._actor.drain_and_stop()
            # break cycle with itself before we exit
            self.stdin = sys.stdin
            self.stdout = sys.stdout
        return r

    def set_trace(self):
        self._actor.send(DebuggerAction.Paused())
        message = self._actor.receive()
        # we give the controller the option to ignore this request to debug
        # by issuing a "detach" message immediately.
        if isinstance(message, DebuggerAction.Detach):
            return
        elif isinstance(message, DebuggerAction.Attach):
            pass
        else:
            raise RuntimeError(f"unexpected debugger message {message}")
        if self.header:
            self.message(self.header)
        super().set_trace()

    def set_quit(self):
        self._actor.send(DebuggerAction.Detach())
        self._actor.drain_and_stop()
        super().set_quit()


class ReadWrapper(io.RawIOBase):
    def __init__(self, actor: debugger.PdbActor):
        self._actor = actor

    def readinto(self, b):
        self._actor.send(DebuggerAction.Read(len(b)))
        response = self._actor.receive()
        if isinstance(response, DebuggerAction.Detach):
            raise bdb.BdbQuit
        assert isinstance(response, DebuggerAction.Write)
        response = cast(DebuggerAction.Write, response)
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
        self._actor.send(DebuggerAction.Write(s.encode()))

    def flush(self):
        pass


if __name__ == "__main__":
    # torch is import to make sure all the dynamic types are registered
    import torch  # noqa

    if torch.cuda.is_available():
        # Force CUDA initialization early on. CUDA init is lazy, and Python CUDA
        # APIs are guarded to init CUDA if necessary. But our worker calls
        # raw libtorch APIs which are not similarly guarded. So just initialize here
        # to avoid issues with potentially using uninitialized CUDA state.
        torch.cuda.init()

    from monarch._rust_bindings.monarch_extension import (  # @manual=//monarch/monarch_extension:monarch_extension
        tensor_worker,
    )

    initialize_logging()

    def check_set_device(device):
        import os

        if str(device) not in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","):
            raise ValueError(
                f"Only devices {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')} are available to monarch worker, "
                f"but torch.cuda.set_device({device}) was called"
            )

    torch.cuda.set_device = check_set_device

    with (
        importlib.resources.as_file(
            importlib.resources.files("monarch") / "py-spy"
        ) as pyspy,
    ):
        if pyspy.exists():
            os.environ["PYSPY_BIN"] = str(pyspy)
        # fallback to using local py-spy

    pdb.set_trace = _set_trace
    # pyre-ignore[16]
    tensor_worker.worker_main()
