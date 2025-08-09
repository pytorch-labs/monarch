# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import AsyncIterator, final, NoReturn

from monarch._rust_bindings.monarch_hyperactor.actor import PythonMessage
from monarch._rust_bindings.monarch_hyperactor.mailbox import (
    Mailbox,
    OncePortReceiver,
    PortReceiver,
)
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask
from monarch._rust_bindings.monarch_hyperactor.selection import Selection
from monarch._rust_bindings.monarch_hyperactor.shape import Shape
from typing_extensions import Self

@final
class PythonActorMeshRef:
    """
    A reference to a remote actor mesh over which PythonMessages can be sent.
    """

    def cast(
        self, mailbox: Mailbox, selection: Selection, message: PythonMessage
    ) -> None:
        """Cast a message to the selected actors in the mesh."""
        ...

    def slice(self, **kwargs: int | slice[int | None, int | None, int | None]) -> Self:
        """
        See PythonActorMeshRef.slice for documentation.
        """
        ...

    def new_with_shape(self, shape: Shape) -> PythonActorMeshRef:
        """
        Return a new mesh ref with the given sliced shape. If the provided shape
        is not a valid slice of the current shape, an exception will be raised.
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
class PythonActorMesh:
    def bind(self) -> PythonActorMeshRef:
        """
        Bind this actor mesh. The returned mesh ref can be used to reach the
        mesh remotely.
        """
        ...

    def cast(
        self, mailbox: Mailbox, selection: Selection, message: PythonMessage
    ) -> None:
        """
        Cast a message to the selected actors in the mesh.
        """
        ...

    def slice(
        self, **kwargs: int | slice[int | None, int | None, int | None]
    ) -> PythonActorMeshRef:
        """
        Slice the mesh into a new mesh ref with the given selection. The reason
        it returns a mesh ref, rather than the mesh object itself, is because
        sliced mesh is a view of the original mesh, and does not own the mesh's
        resources.

        Arguments:
        - `kwargs`: argument name is the label, and argument value is how to
          slice the mesh along the dimension of that label.
        """
        ...

    def new_with_shape(self, shape: Shape) -> PythonActorMeshRef:
        """
        Return a new mesh ref with the given sliced shape. If the provided shape
        is not a valid slice of the current shape, an exception will be raised.
        """
        ...

    def get_supervision_event(self) -> ActorSupervisionEvent | None:
        """
        Returns supervision event if there is any.
        """
        ...

    def supervision_event(self) -> PythonTask[Exception]:
        """
        Completes with an exception when there is a supervision error.
        """
        ...

    def get(self, rank: int) -> ActorId | None:
        """
        Get the actor id for the actor at the given rank.
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

    async def stop(self) -> None:
        """
        Stop all actors that are part of this mesh.
        Using this mesh after stop() is called will raise an Exception.
        """
        ...

    @property
    def stopped(self) -> bool:
        """
        If the mesh has been stopped.
        """
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
