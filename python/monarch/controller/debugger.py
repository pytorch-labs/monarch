# pyre-strict
import sys
from typing import Optional

_is_ipython: Optional[bool] = None


def is_ipython() -> bool:
    global _is_ipython
    if _is_ipython is not None:
        return _is_ipython
    try:
        from IPython import get_ipython

        _is_ipython = get_ipython() is not None
    except ImportError:
        _is_ipython = False
    return _is_ipython


def write(msg: str) -> None:
    sys.stdout.write(msg)
    sys.stdout.flush()


def read(requested_size: int) -> bytes:
    if not is_ipython():
        b = bytearray(requested_size)
        bytes_read = sys.stdin.buffer.raw.readinto(b)
        return bytes(b[:bytes_read])

    # ipython doesn't have stdin directly connected
    # so we need to use input() instead.
    user_input = input() + "\n"
    input_bytes = user_input.encode("utf-8")
    num_bytes_to_write = len(input_bytes)
    if requested_size < num_bytes_to_write:
        raise RuntimeError(
            f"Debugger input line too long, max length is {requested_size}"
        )
    return input_bytes[:num_bytes_to_write]
