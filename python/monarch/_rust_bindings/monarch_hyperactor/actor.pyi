# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc

from typing import final, List, Protocol

from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox, PortId
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId, Proc, Serialized
from monarch._rust_bindings.monarch_hyperactor.shape import Shape

@final
class PickledMessage:
    """
    A message that can be sent to PickledMessage{,Client}Actor. It is a wrapper around
    a serialized message and the sender's actor id.

    Arguments:
    - `sender_actor_id`: The actor id of the sender.
    - `message`: The pickled message.
    """

    def __init__(self, *, sender_actor_id: ActorId, message: bytes) -> None: ...
    @property
    def sender_actor_id(self) -> ActorId:
        """The actor id of the sender."""
        ...

    @property
    def message(self) -> bytes:
        """The pickled message."""
        ...

    def serialize(self) -> Serialized:
        """Serialize the message into a Serialized object."""
        ...

@final
class PickledMessageClientActor:
    """
    A python based detached actor that can be used to send messages to other
    actors and recieve PickledMessage objects from them.

    Arguments:
    - `proc`: The proc the actor is a part of.
    - `actor_name`: Name of the actor.
    """

    def __init__(self, proc: Proc, actor_name: str) -> None: ...
    def send(self, actor_id: ActorId, message: Serialized) -> None:
        """
        Send a message to the actor with the given actor id.

        Arguments:
        - `actor_id`: The actor id of the actor to send the message to.
        - `message`: The message to send.
        """
        ...

    def get_next_message(
        self, *, timeout_msec: int | None = None
    ) -> PickledMessage | None:
        """
        Get the next message sent to the actor. If the timeout is reached
        before a message is received, None is returned.

        Arguments:
        - `timeout_msec`: Number of milliseconds to wait for a message.
                None means wait forever.
        """
        ...

    def stop_worlds(self, world_names: List[str]) -> None:
        """Stop the system."""
        ...

    def drain_and_stop(self) -> list[PickledMessage]:
        """Stop the actor and drain all messages."""
        ...

    def world_status(self) -> dict[str, str]:
        """Get the world status from the system."""
        ...

    @property
    def actor_id(self) -> ActorId:
        """The actor id of the actor."""
        ...

@final
class PythonMessage:
    """
    A message that carries a python method and a pickled message that contains
    the arguments to the method.
    """

    def __init__(
        self,
        method: str,
        message: bytes,
        response_port: PortId | None = None,
        rank_in_response: bool = False,
    ) -> None: ...
    @property
    def method(self) -> str:
        """The method of the message."""
        ...

    @property
    def message(self) -> bytes:
        """The pickled arguments."""
        ...

    @property
    def response_port(self) -> PortId | None:
        """The response port."""
        ...

    @property
    def rank_in_response(self) -> bool:
        """Whether or not to include the rank of the handling actor in the response."""
        ...

@final
class PythonActorHandle:
    """
    A python wrapper around hyperactor ActorHandle. It represents a handle to an
    actor.

    Arguments:
    - `inner`: The inner actor handle.
    """

    def send(self, message: PythonMessage) -> None:
        """
        Send a message to the actor.

        Arguments:
        - `message`: The message to send.
        """
        ...

    def bind(self) -> ActorId:
        """
        Bind this actor. The returned actor id can be used to reach the actor externally.
        """
        ...

@final
class PanicFlag:
    """
    A mechanism to notify the hyperactor runtime that a panic has occurred in an
    asynchronous Python task. See [Panics in async endpoints] for more details.
    """

    def signal_panic(self, ex: BaseException) -> None:
        """
        Signal that a panic has occurred in an asynchronous Python task.
        """
        ...

class Actor(Protocol):
    async def handle(
        self, mailbox: Mailbox, message: PythonMessage, panic_flag: PanicFlag
    ) -> None: ...
    async def handle_cast(
        self,
        mailbox: Mailbox,
        rank: int,
        shape: Shape,
        message: PythonMessage,
        panic_flag: PanicFlag,
    ) -> None: ...
