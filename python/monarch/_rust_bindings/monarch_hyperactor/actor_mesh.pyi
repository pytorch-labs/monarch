# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import AsyncIterator, final

from monarch._rust_bindings.monarch_hyperactor.actor import PythonMessage
from monarch._rust_bindings.monarch_hyperactor.mailbox import (
    Mailbox,
    OncePortReceiver,
    PortReceiver,
)
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.selection import Selection
from monarch._rust_bindings.monarch_hyperactor.shape import Shape

@final
class PythonActorMesh:
    def cast(self, selection: Selection, message: PythonMessage) -> None:
        """
        Cast a message to the selected actors in the mesh.
        """

    def get_supervision_event(self) -> ActorSupervisionEvent | None:
        """
        Returns supervision event if there is any.
        """
        ...

    def get(self, rank: int) -> ActorId | None:
        """
        Get the actor id for the actor at the given rank.
        """
        ...

    # TODO(albertli): remove this when pushing all supervision logic to Rust
    def monitor(self) -> ActorMeshMonitor:
        """
        Returns a supervision monitor for this mesh.
        """
        ...

    @property
    def client(self) -> Mailbox:
        """
        A client that can be used to communicate with individual
        actors in the mesh, and also to create ports that can be
        broadcast across the mesh)
        """
        ...

    @property
    def shape(self) -> Shape:
        """
        The Shape object that describes how the rank of an actor
        retrieved with get corresponds to coordinates in the
        mesh.
        """
        ...

@final
class ActorMeshMonitor:
    def __aiter__(self) -> AsyncIterator["ActorSupervisionEvent"]:
        """
        Returns an async iterator for this monitor.
        """
        ...

    async def __anext__(self) -> "ActorSupervisionEvent":
        """
        Returns the next proc event in the proc mesh.
        """
        ...

@final
class MonitoredPortReceiver:
    """
    A monitored receiver to which PythonMessages are sent.
    """

    def __init__(self, receiver: PortReceiver, monitor: ActorMeshMonitor) -> None:
        """
        Create a new monitored receiver from a PortReceiver.
        """
        ...

    async def recv(self) -> PythonMessage:
        """Receive a PythonMessage from the port's sender."""
        ...
    def blocking_recv(self) -> PythonMessage:
        """Receive a single PythonMessage from the port's sender."""
        ...

@final
class MonitoredOncePortReceiver:
    """
    A variant of monitored PortReceiver that can only receive a single message.
    """

    def __init__(self, receiver: OncePortReceiver, monitor: ActorMeshMonitor) -> None:
        """
        Create a new monitored receiver from a PortReceiver.
        """
        ...

    async def recv(self) -> PythonMessage:
        """Receive a single PythonMessage from the port's sender."""
        ...
    def blocking_recv(self) -> PythonMessage:
        """Receive a single PythonMessage from the port's sender."""
        ...

@final
class ActorSupervisionEvent:
    @property
    def actor_id(self) -> ActorId:
        """
        The actor id of the actor.
        """
        ...

    @property
    def actor_status(self) -> str:
        """
        Detailed actor status.
        """
        ...
