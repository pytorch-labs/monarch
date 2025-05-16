# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import math
import queue
import threading
from typing import Callable, Optional, Tuple

from monarch_supervisor import TTL


class Monitor:
    """A monitor is a thread that watches for reported events to expire."""

    def __init__(self) -> None:
        self.thread = threading.Thread(target=self._main, daemon=True, name="monitor")
        self.events: queue.Queue[Tuple[Callable[[], None], Callable[[], float]]] = (
            queue.Queue()
        )
        self.events.put((lambda: None, TTL(None)))

    def start(self) -> None:
        """Start the monitor thread."""
        self.thread.start()

    def _main(self) -> None:
        debug, ttl = self.events.get()
        while True:
            try:
                timeout = ttl()
                next_debug, next_ttl = self.events.get(
                    timeout=None if timeout == math.inf else timeout
                )
            except queue.Empty:
                debug()
                next_debug, next_ttl = self.events.get(timeout=None)

            debug, ttl = next_debug, next_ttl

    def __call__(
        self,
        debug_fn: Callable[[], None] = lambda: None,
        timeout: Optional[float] = None,
    ) -> None:
        """Start a new event with the provided timeout.
        If a timeout is specified, and a new event is not reported by before it expires,
        the provided debug_fn is called."""
        self.events.put((debug_fn, TTL(timeout)))
