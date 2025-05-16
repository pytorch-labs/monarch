# pyre-unsafe
from contextlib import contextmanager
from typing import Any, List


class Lines:
    """
    Simple way to emit code where we track a per-line context object.
    """

    def __init__(self, context=None):
        self._lines: List[str] = []
        self._context: List[Any] = []
        self._current_context = context

    def get_context(self, lineno) -> Any:
        return self._context[lineno - 1]

    @contextmanager
    def context(self, obj: Any):
        old, self._current_context = self._current_context, obj
        try:
            yield
        finally:
            self._current_context = old

    def emit(self, lines: str) -> None:
        self._lines.extend(lines.split("\n"))
        while len(self._context) < len(self._lines):
            self._context.append(self._current_context)

    def emit_lines(self, lines: "Lines") -> None:
        """
        Append another lines object on this one,
        preserving its per-line context.
        """
        self._lines.extend(lines._lines)
        self._context.extend(lines._context)

    def text(self) -> str:
        return "\n".join(self._lines)
