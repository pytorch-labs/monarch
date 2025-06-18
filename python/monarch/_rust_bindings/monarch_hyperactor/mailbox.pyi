# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

from monarch._rust_bindings.monarch_hyperactor.actor import PythonMessage

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId

from monarch._rust_bindings.monarch_hyperactor.shape import Shape

@final
class PortId:
    def __init__(self, actor_id: ActorId, index: int) -> None:
        """
        Create a new port id given an actor id and an index.
        """
        ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def actor_id(self) -> ActorId:
        """
        The ID of the actor that owns the port.
        """
        ...

    @property
    def index(self) -> int:
        """
        The actor-relative index of the port.
        """
        ...

    @staticmethod
    def from_string(port_id_str: str) -> PortId:
        """
        Parse a port id from the provided string.
        """
        ...

@final
class PortHandle:
    """
    A handle to a port over which PythonMessages can be sent.
    """

    def send(self, message: PythonMessage) -> None:
        """Send a message to the port's receiver."""

    def bind(self) -> PortRef:
        """Bind this port. The returned port ref can be used to reach the port externally."""
        ...

@final
class PortRef:
    """
    A reference to a remote port over which PythonMessages can be sent.
    """

    def send(self, mailbox: Mailbox, message: PythonMessage) -> None:
        """Send a single message to the port's receiver."""
        ...
    def __repr__(self) -> str: ...

@final
class PortReceiver:
    """
    A receiver to which PythonMessages are sent.
    """
    async def recv(self) -> PythonMessage:
        """Receive a PythonMessage from the port's sender."""
        ...
    def blocking_recv(self) -> PythonMessage:
        """Receive a single PythonMessage from the port's sender."""
        ...

@final
class OncePortHandle:
    """
    A variant of PortHandle that can only send a single message.
    """

    def send(self, message: PythonMessage) -> None:
        """Send a single message to the port's receiver."""
        ...

    def bind(self) -> OncePortRef:
        """Bind this port. The returned port ID can be used to reach the port externally."""
        ...

@final
class OncePortRef:
    """
    A reference to a remote once port over which a single PythonMessages can be sent.
    """

    def send(self, mailbox: Mailbox, message: PythonMessage) -> None:
        """Send a single message to the port's receiver."""
        ...
    def __repr__(self) -> str: ...

@final
class OncePortReceiver:
    """
    A variant of PortReceiver that can only receive a single message.
    """
    async def recv(self) -> PythonMessage:
        """Receive a single PythonMessage from the port's sender."""
        ...
    def blocking_recv(self) -> PythonMessage:
        """Receive a single PythonMessage from the port's sender."""
        ...

@final
class Mailbox:
    """
    A mailbox from that can receive messages.
    """

    def open_port(self) -> tuple[PortHandle, PortReceiver]:
        """Open a port to receive `PythonMessage` messages."""
        ...

    def open_once_port(self) -> tuple[OncePortHandle, OncePortReceiver]:
        """Open a port to receive a single `PythonMessage` message."""
        ...

    def post(self, dest: ActorId, message: PythonMessage) -> None:
        """
        Post a message to the provided destination. If the destination is an actor id,
        the message is sent to the default handler for `PythonMessage` on the actor.
        Otherwise, it is sent to the port directly.
        """
        ...

    def post_cast(
        self, dest: ActorId, rank: int, shape: Shape, message: PythonMessage
    ) -> None:
        """
        Post a message to the provided actor. It will be handled using the handle_cast
        endpoint as if the destination was `rank` of `shape`.
        """
        ...

    @property
    def actor_id(self) -> ActorId: ...
