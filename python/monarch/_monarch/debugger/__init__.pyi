from typing import final, Optional, Union

from monarch._monarch.hyperactor import Serialized

@final
class DebuggerAction:
    """Enum representing actions for the debugger communication between worker and client."""

    class Paused:
        """
        Sent from worker to client to indicate that the worker has entered
        a pdb debugging session.
        """

        pass

    class Attach:
        """
        Sent from client to worker to indicate that the client has started
        the debugging session.
        """

        pass

    class Detach:
        """Sent to client or to worker to end the debugging session."""

        pass

    class Write:
        """Sent to client or to worker to write bytes to receiver's stdout."""

        def __init__(self, bytes: bytes) -> None: ...

    class Read:
        """Sent from worker to client to read bytes from client's stdin."""

        def __init__(self, requested_size: int) -> None: ...
        @property
        def requested_size(self) -> int:
            """Get the number of bytes to read from stdin."""
            ...

DebuggerActionType = Union[
    DebuggerAction.Paused,
    DebuggerAction.Attach,
    DebuggerAction.Detach,
    DebuggerAction.Read,
    DebuggerAction.Write,
]

@final
class DebuggerMessage:
    """A message for debugger communication between worker and client."""

    def __init__(self, action: DebuggerActionType) -> None:
        """
        Create a new DebuggerMessage.

        Arguments:
            action: The debugger action to include in the message.
        """
        ...

    @property
    def action(self) -> DebuggerActionType:
        """Get the debugger action contained in this message."""
        ...

    def serialize(self) -> Serialized:
        """
        Serialize this message for transmission.

        Returns:
            A serialized representation of this message.
        """
        ...

@final
class PdbActor:
    """An actor for interacting with PDB debugging sessions."""

    def __init__(self) -> None:
        """Create a new PdbActor."""
        ...

    def send(self, action: DebuggerActionType) -> None:
        """
        Send a debugger action to the worker.

        Arguments:
            action: The debugger action to send.
        """
        ...

    def receive(self) -> Optional[DebuggerActionType]:
        """
        Receive a debugger action from the worker.

        Returns:
            A DebuggerAction if one is available, or None if no action is available.
        """
        ...

    def drain_and_stop(self) -> None:
        """
        Drain any remaining messages and stop the actor.
        """
        ...

def get_bytes_from_write_action(action: DebuggerAction.Write) -> bytes:
    """
    Extract the bytes from the provided write action.
    """
    ...
