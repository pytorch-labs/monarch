# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Callable, List, Tuple, TYPE_CHECKING
from weakref import ref, WeakKeyDictionary

from . import messages
from .borrows import Borrow
from .context_manager import activate_first_context_manager
from .fake import fake_call
from .reference import Referenceable

if TYPE_CHECKING:
    from monarch.common.client import Client  # @manual

    from .tensor import Tensor


class Stream:
    def __init__(self, name: str, _default=False):
        self.name = name
        self.default: bool = _default
        self.clients: WeakKeyDictionary["Client", "StreamRef"] = WeakKeyDictionary()

    def __repr__(self):
        return f"<Stream({repr(self.name)}) at {hex(id(self))}>"

    def __str__(self):
        return f"stream {repr(self.name)}"

    def activate(self):
        return _active_stream(self)

    def _to_ref(self, client: "Client"):
        if client not in self.clients:
            self.clients[client] = StreamRef(client, self.name, self.default)
        return self.clients[client]

    def borrow(self, t: "Tensor", mutable: bool = False) -> Tuple["Tensor", "Borrow"]:
        """
            borrowed_tensor, borrow = self.borrow(t)

        Borrows tensor 't' for use on this stream.
        The memory of t will stay alive until borrow.drop() is called, which will free t and
        and any of its alises on stream `self` and will cause t.stream to wait on self at that point so
        that the memory of t can be reused.

        If `mutable` then self can write to the storage of `t`, but t.stream cannot read or write `t` until,
        the borrow is returned (becomes free and a wait_for has been issued).

        If not `mutable` both `self` and `t.stream` can read from t's storage but neither can write to it.
        """
        client = t.mesh.client
        aliases = t._aliases
        r = type(t)(fake_call(t._fake.clone), t.mesh, self)
        client.new_node((r,), (t,))
        borrow = r._aliases.borrow_from(client.new_ref(), t.mesh, aliases, mutable)
        client.new_borrow(borrow)
        assert r.ref is not None
        t.mesh._send(
            messages.BorrowCreate(
                r, borrow._id, t, t.stream._to_ref(client), self._to_ref(client)
            )
        )
        r._on_first_use = lambda t: borrow._use()

        return r, borrow


class StreamRef(Referenceable):
    def __init__(self, client: "Client", name: str, default: bool):
        self.ref = client.new_ref()
        self.client = ref(client)
        self.name = name
        self.default = default
        client.send(
            client.all_ranks,
            messages.CreateStream(self, self.default),
        )

    def __repr__(self):
        return f"<StreamRef {repr(self.name)} {self.ref}>"

    def delete_ref(self, ref):
        client = self.client()
        if client is not None and not client._shutdown:
            client.handle_deletes(client.all_ranks, [ref])


_active = Stream("main", _default=True)
_on_change: List[Callable] = []


def get_active_stream():
    return _active


@activate_first_context_manager
def _active_stream(stream: Stream):
    global _active
    for on_change in _on_change:
        on_change(_active, stream)

    _active, old = stream, _active
    try:
        yield
    finally:
        for on_change in _on_change:
            on_change(_active, old)
        _active = old
