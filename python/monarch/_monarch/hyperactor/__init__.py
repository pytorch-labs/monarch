# pyre-strict
import abc

from monarch._rust_bindings.hyperactor import (  # @manual=//monarch/monarch_extension:monarch_extension
    ActorId,
    Alloc,
    AllocConstraints,
    AllocSpec,
    init_proc,
    LocalAllocatorBase,
    Mailbox,
    OncePortHandle,
    OncePortReceiver,
    PickledMessage,
    PickledMessageClientActor,
    PortHandle,
    PortId,
    PortReceiver,
    Proc,
    ProcessAllocatorBase,
    ProcMesh,
    PythonActorHandle as ActorHandle,
    PythonActorMesh,
    PythonMessage,
    Serialized,
)

from monarch._rust_bindings.shape import (  # @manual=//monarch/monarch_extension:monarch_extension
    Shape,
)


class Actor(abc.ABC):
    @abc.abstractmethod
    async def handle(self, mailbox: Mailbox, message: PythonMessage) -> None: ...

    async def handle_cast(
        self,
        mailbox: Mailbox,
        rank: int,
        coordinates: list[tuple[str, int]],
        message: PythonMessage,
    ) -> None:
        await self.handle(mailbox, message)


__all__ = [
    "init_proc",
    "Actor",
    "ActorId",
    "ActorHandle",
    "Alloc",
    "AllocSpec",
    "PortId",
    "Proc",
    "Serialized",
    "PickledMessage",
    "PickledMessageClientActor",
    "PythonMessage",
    "Mailbox",
    "PortHandle",
    "PortReceiver",
    "OncePortHandle",
    "OncePortReceiver",
    "Alloc",
    "AllocSpec",
    "AllocConstraints",
    "ProcMesh",
    "PythonActorMesh",
    "ProcessAllocatorBase",
    "Shape",
    "LocalAllocatorBase",
]
