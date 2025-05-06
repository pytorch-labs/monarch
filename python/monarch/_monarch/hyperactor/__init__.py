# pyre-strict
import abc

from .._lib import hyperactor  # @manual=//monarch/monarch_extension:monarch_extension

init_proc = hyperactor.init_proc
init_asyncio_loop = hyperactor.init_asyncio_loop

ActorId = hyperactor.ActorId
ActorHandle = hyperactor.PythonActorHandle
PortId = hyperactor.PortId
Proc = hyperactor.Proc

Serialized = hyperactor.Serialized
PickledMessage = hyperactor.PickledMessage
PickledMessageClientActor = hyperactor.PickledMessageClientActor
PythonMessage = hyperactor.PythonMessage
PythonActorHandle = hyperactor.PythonActorHandle

Mailbox = hyperactor.Mailbox
PortHandle = hyperactor.PortHandle
PortReceiver = hyperactor.PortReceiver
OncePortHandle = hyperactor.OncePortHandle
OncePortReceiver = hyperactor.OncePortReceiver

AllocConstraints = hyperactor.AllocConstraints
AllocSpec = hyperactor.AllocSpec
Alloc = hyperactor.Alloc
ProcessAllocator = hyperactor.ProcessAllocator
LocalAllocator = hyperactor.LocalAllocator

ProcMesh = hyperactor.ProcMesh
PythonActorMesh = hyperactor.PythonActorMesh
Shape = hyperactor.Shape


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
    "init_asyncio_loop",
    "Actor",
    "ActorId",
    "ActorHandle",
    "PortId",
    "Proc",
    "Serialized",
    "PickledMessage",
    "PickledMessageClientActor",
    "PythonMessage",
    "PythonActorHandle",
    "Mailbox",
    "PortHandle",
    "PortReceiver",
    "OncePortHandle",
    "OncePortReceiver",
    "Alloc",
    "AllocSpec",
    "AllocConstraints",
    "ProcessAllocator",
    "LocalAllocator",
    "ProcMesh",
    "PythonActorMesh",
]
