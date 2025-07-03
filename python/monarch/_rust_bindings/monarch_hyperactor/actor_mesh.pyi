# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

from monarch._rust_bindings.monarch_hyperactor.actor import PythonMessage
from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.selection import Selection
from monarch._rust_bindings.monarch_hyperactor.shape import Shape

@final
class PythonActorMesh:
    def cast(self, selection: Selection, message: PythonMessage) -> None:
        """
        Cast a message to the selected actors in the mesh.
        """

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
