# pyre-strict

from typing import final, Type

from monarch._rust_bindings.hyperactor_extension import Alloc
from monarch._rust_bindings.monarch_hyperactor.actor_mesh import PythonActorMesh
from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox
from monarch._rust_bindings.monarch_hyperactor.proc import Actor

@final
class ProcMesh:
    @classmethod
    async def allocate_nonblocking(self, alloc: Alloc) -> ProcMesh:
        """
        Allocate a process mesh according to the provided alloc.
        Returns when the mesh is fully allocated.

        Arguments:
        - `alloc`: The alloc to allocate according to.
        """
        ...

    @classmethod
    def allocate_blocking(self, alloc: Alloc) -> ProcMesh:
        """
        Allocate a process mesh according to the provided alloc.
        Blocks until the mesh is fully allocated.

        Arguments:
        - `alloc`: The alloc to allocate according to.
        """
        ...

    async def spawn_nonblocking(self, name: str, actor: Type[Actor]) -> PythonActorMesh:
        """
        Spawn a new actor on this mesh.

        Arguments:
        - `name`: Name of the actor.
        - `actor`: The type of the actor that will be spawned.
        """
        ...

    async def spawn_blocking(self, name: str, actor: Type[Actor]) -> PythonActorMesh:
        """
        Spawn a new actor on this mesh. Blocks until the actor is fully spawned.

        Arguments:
        - `name`: Name of the actor.
        - `actor`: The type of the actor that will be spawned.
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
    def __repr__(self) -> str: ...
