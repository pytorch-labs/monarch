# pyre-strict

from __future__ import annotations

from traceback import FrameSummary
from typing import (
    cast,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
)

from monarch._monarch import worker
from monarch.common.function import ResolvableFromCloudpickle, ResolvableFunction
from monarch.common.invocation import DeviceException, RemoteException
from monarch.common.reference import Referenceable
from monarch.common.stream import StreamRef
from monarch.common.tree import flattener
from pyre_extensions import none_throws

from .shape import NDSlice
from .tensor_factory import TensorFactory

if TYPE_CHECKING:
    from .device_mesh import DeviceMesh, RemoteProcessGroup
    from .pipe import Pipe
    from .recording import Recording
    from .tensor import Tensor


Dims = Tuple[str, ...]


def _to_rust_function(
    x: ResolvableFunction,
) -> worker.ResolvableFunction:
    if isinstance(x, ResolvableFromCloudpickle):
        return worker.Cloudpickle(bytes=x.data)
    return worker.FunctionPath(path=str(x))


def _result_to_references(result: object) -> List[worker.Ref | None]:
    """
    Flatten the result pytree.
    Only keep the referenceables and leave the rest as None.
    The workers will generate the full result list so we know
    what referenceables to be assigned to.
    """
    leaves = flattener(result, lambda x: True)(result)
    return [
        worker.Ref(id=none_throws(leaf.ref))
        if isinstance(leaf, Referenceable)
        else None
        for leaf in leaves
    ]


# We cant do inheritance with NamedTuple so we can use this protocol for
# type casting for now until we can move to rust messages entirely.
# Preferring this over a massive if else to keep everything co-located and
# easier to identify drift.
class SupportsToRustMessage(Protocol):
    def to_rust_message(self) -> worker.WorkerMessage: ...


class CreateDeviceMesh(NamedTuple):
    result: DeviceMesh
    names: Dims
    ranks: NDSlice

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.CreateDeviceMesh(
            result=worker.Ref(id=self.result.ref),
            names=self.names,
            ranks=NDSlice(
                offset=self.ranks.offset,
                sizes=self.ranks.sizes,
                strides=self.ranks.strides,
            ),
        )


class CreateStream(NamedTuple):
    result: StreamRef
    default: bool

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.CreateStream(
            id=worker.StreamRef(id=self.result.ref),
            stream_creation=(
                worker.StreamCreationMode.UseDefaultStream
                if self.default
                else worker.StreamCreationMode.CreateNewStream
            ),
        )


class CreateRemoteProcessGroup(NamedTuple):
    result: Referenceable
    device_mesh: DeviceMesh
    dims: Dims

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.CreateRemoteProcessGroup(
            result=worker.Ref(id=none_throws(self.result.ref)),
            device_mesh=worker.Ref(id=self.device_mesh.ref),
            dims=self.dims,
        )


class CallFunction(NamedTuple):
    ident: int
    result: object  # pytree with tensors in it
    mutates: Tuple[Tensor, ...]
    function: ResolvableFunction
    args: Tuple[object, ...]
    kwargs: Dict[str, object]
    stream: StreamRef
    device_mesh: DeviceMesh
    remote_process_groups: List[RemoteProcessGroup]

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.CallFunction(
            seq=self.ident,
            results=_result_to_references(self.result),
            mutates=[worker.Ref(id=none_throws(r.ref)) for r in self.mutates],
            function=_to_rust_function(self.function),
            args=self.args,
            kwargs=self.kwargs,
            stream=worker.StreamRef(id=self.stream.ref),
            remote_process_groups=[
                worker.Ref(id=none_throws(remote_process_group.ref))
                for remote_process_group in self.remote_process_groups
            ],
        )


class Exit(NamedTuple):
    destroy_pg: bool
    error: Optional[RemoteException | DeviceException | Exception]

    def to_rust_message(self) -> worker.WorkerMessage:
        actor_id = None
        error_message = None
        if isinstance(self.error, (RemoteException, DeviceException)):
            actor_id = self.error.source_actor_id
            error_message = self.error.message
        elif self.error is not None:
            error_message = str(self.error)

        error_reason = None if error_message is None else (actor_id, error_message)
        return worker.Exit(error_reason=error_reason)


class CommandGroup(NamedTuple):
    commands: List[NamedTuple]

    def to_rust_message(self) -> worker.WorkerMessage:
        rust_commands = []
        for c in self.commands:
            if hasattr(c, "to_rust_message"):
                c = cast(SupportsToRustMessage, c)
                rust_commands.append(c.to_rust_message())
            else:
                raise NotImplementedError(f"Unsupported command {c}")
        return worker.CommandGroup(commands=rust_commands)


class RecordingFormal(NamedTuple):
    result: "Tensor"
    argument_index: int
    stream: "StreamRef"

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.RecordingFormal(
            result=worker.Ref(id=none_throws(self.result.ref)),
            argument_index=self.argument_index,
            stream=worker.StreamRef(id=self.stream.ref),
        )


class RecordingResult(NamedTuple):
    input: "Tensor"
    output_index: int
    stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.RecordingResult(
            result=worker.Ref(id=none_throws(self.input.ref)),
            output_index=self.output_index,
            stream=worker.StreamRef(id=self.stream.ref),
        )


class DefineRecording(NamedTuple):
    result: Recording
    nresults: int
    nformals: int
    commands: List[NamedTuple]

    def to_rust_message(self) -> worker.WorkerMessage:
        rust_commands = []
        for c in self.commands:
            if hasattr(c, "to_rust_message"):
                c = cast(SupportsToRustMessage, c)
                rust_commands.append(c.to_rust_message())
            else:
                raise NotImplementedError(f"Unsupported command {c}")
        return worker.DefineRecording(
            result=worker.Ref(id=none_throws(self.result.ref)),
            nresults=self.nresults,
            nformals=self.nformals,
            commands=rust_commands,
        )


class CallRecording(NamedTuple):
    ident: int
    recording: Recording
    results: List[Tensor]
    actuals: List[Tensor]

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.CallRecording(
            seq=self.ident,
            recording=worker.Ref(id=none_throws(self.recording.ref)),
            results=[worker.Ref(id=none_throws(r.ref)) for r in self.results],
            actuals=[worker.Ref(id=none_throws(r.ref)) for r in self.actuals],
        )


class DeleteRefs(NamedTuple):
    refs: List[int]

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.DeleteRefs(refs=[worker.Ref(id=r) for r in self.refs])


# This is worker <> controller/backend comms only will be supported differently
class Restarted(NamedTuple):
    result: int


class SendValue(NamedTuple):
    ident: int
    destination: Pipe | None  # if present the pipe along which to send the result,
    # otherwise send FetchResult to controller
    mutates: Tuple[Tensor, ...]
    function: ResolvableFunction | None  # None is equivalent to lambda x: x
    args: Tuple[object, ...]
    kwargs: Dict[str, object]
    stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.SendValue(
            seq=self.ident,
            destination=(
                worker.Ref(id=self.destination.ref) if self.destination else None
            ),
            mutates=[worker.Ref(id=none_throws(r.ref)) for r in self.mutates],
            function=_to_rust_function(self.function) if self.function else None,
            args=self.args,
            kwargs=self.kwargs,
            stream=worker.StreamRef(id=self.stream.ref),
        )


# Worker -> Controller comm only handled differently
class FetchResult(NamedTuple):
    ident: int
    value: object


# Worker -> Controller comm only handled differently
class RemoteFunctionFailed(NamedTuple):
    failing_ident: int
    stack_offset: int
    exception: Exception
    worker_frames: List[FrameSummary]


# Worker -> Controller comm only handled differently
class InternalException(NamedTuple):
    exception: Exception
    frames: List[FrameSummary]


# Worker -> Controller comm only handled differently
class RemoteGeneratorFailed(NamedTuple):
    exception: Exception
    frames: List[FrameSummary]


# Worker -> Controller comm only handled differently
class Status(NamedTuple):
    first_uncompleted_ident: int


# When the controller is waiting on a status update,
# it will request one even if it is before the
# periodic one.
class RequestStatus(NamedTuple):
    ident: int
    controller: bool

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.RequestStatus(seq=self.ident, controller=self.controller)


class BorrowCreate(NamedTuple):
    result: Tensor
    borrow: int
    tensor: Tensor
    from_stream: StreamRef
    to_stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.BorrowCreate(
            result=worker.Ref(id=none_throws(self.result.ref)),
            borrow=self.borrow,
            tensor=worker.Ref(id=none_throws(self.tensor.ref)),
            from_stream=worker.StreamRef(id=self.from_stream.ref),
            to_stream=worker.StreamRef(id=self.to_stream.ref),
        )


class BorrowDrop(NamedTuple):
    borrow: int  # id of borrowed tensor

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.BorrowDrop(
            borrow=self.borrow,
        )


class BorrowFirstUse(NamedTuple):
    borrow: int  # id of borrowed tensor

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.BorrowFirstUse(
            borrow=self.borrow,
        )


class BorrowLastUse(NamedTuple):
    borrow: int  # id of borrowed tensor

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.BorrowLastUse(
            borrow=self.borrow,
        )


class SendTensor(NamedTuple):
    result: Tensor
    from_ranks: NDSlice
    to_ranks: NDSlice
    tensor: Tensor
    factory: TensorFactory
    from_stream: StreamRef
    to_stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.SendTensor(
            result=worker.Ref(id=none_throws(self.result.ref)),
            from_ranks=NDSlice(
                offset=self.from_ranks.offset,
                sizes=self.from_ranks.sizes,
                strides=self.from_ranks.strides,
            ),
            to_ranks=NDSlice(
                offset=self.to_ranks.offset,
                sizes=self.to_ranks.sizes,
                strides=self.to_ranks.strides,
            ),
            tensor=worker.Ref(id=none_throws(self.tensor.ref)),
            factory=worker.TensorFactory(
                size=self.factory.size,
                dtype=self.factory.dtype,
                device=self.factory.device,
                layout=self.factory.layout,
            ),
            from_stream=worker.StreamRef(id=self.from_stream.ref),
            to_stream=worker.StreamRef(id=self.to_stream.ref),
        )


class SplitComm(NamedTuple):
    dims: Dims
    device_mesh: DeviceMesh
    stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.SplitComm(
            dims=self.dims,
            device_mesh=worker.Ref(id=self.device_mesh.ref),
            stream=worker.StreamRef(id=self.stream.ref),
        )


class SplitCommForProcessGroup(NamedTuple):
    remote_process_group: DeviceMesh
    stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.SplitCommForProcessGroup(
            remote_process_group=worker.Ref(id=self.remote_process_group.ref),
            stream=worker.StreamRef(id=self.stream.ref),
        )


class Reduce(NamedTuple):
    result: Tensor
    local_tensor: Tensor
    factory: TensorFactory
    source_mesh: DeviceMesh
    stream: StreamRef
    dims: Dims
    reduction: str
    scatter: bool
    inplace: bool
    out: Tensor | None

    def to_rust_message(self) -> worker.WorkerMessage:
        match self.reduction:
            case "sum":
                reduction = worker.ReductionType.Sum
            case "prod":
                reduction = worker.ReductionType.Prod
            case "stack":
                reduction = worker.ReductionType.Stack
            case "avg":
                reduction = worker.ReductionType.Avg
            case "min":
                reduction = worker.ReductionType.Min
            case "max":
                reduction = worker.ReductionType.Max
            case _:
                raise ValueError(f"Unsupported reduction {self.reduction}")

        return worker.Reduce(
            result=worker.Ref(id=none_throws(self.result.ref)),
            tensor=worker.Ref(id=none_throws(self.local_tensor.ref)),
            factory=worker.TensorFactory(
                size=self.factory.size,
                dtype=self.factory.dtype,
                device=self.factory.device,
                layout=self.factory.layout,
            ),
            mesh=worker.Ref(id=self.source_mesh.ref),
            stream=worker.StreamRef(id=self.stream.ref),
            dims=self.dims,
            reduction=reduction,
            scatter=self.scatter,
            in_place=self.inplace,
            out=worker.Ref(id=none_throws(self.out.ref))
            if self.out is not None
            else None,
        )


class CreatePipe(NamedTuple):
    result: Pipe
    key: str
    function: ResolvableFunction
    max_messages: int
    device_mesh: DeviceMesh
    args: Tuple[object, ...]
    kwargs: Dict[str, object]

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.CreatePipe(
            result=worker.Ref(id=self.result.ref),
            key=self.key,
            function=_to_rust_function(self.function),
            max_messages=self.max_messages,
            mesh=worker.Ref(id=self.device_mesh.ref),
            args=self.args,
            kwargs=self.kwargs,
        )


class PipeRecv(NamedTuple):
    ident: int
    result: object  # pytree with tensors in it
    pipe: Pipe
    stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.PipeRecv(
            seq=self.ident,
            results=_result_to_references(self.result),
            pipe=worker.Ref(id=self.pipe.ref),
            stream=worker.StreamRef(id=self.stream.ref),
        )


class BackendNetworkInit(NamedTuple):
    hostname: str | None = None
    port: int | None = None

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.BackendNetworkInit()


class BackendNetworkPointToPointInit(NamedTuple):
    from_stream: StreamRef
    to_stream: StreamRef

    def to_rust_message(self) -> worker.WorkerMessage:
        return worker.BackendNetworkPointToPointInit(
            from_stream=worker.StreamRef(id=self.from_stream.ref),
            to_stream=worker.StreamRef(id=self.to_stream.ref),
        )


# TODO: This is not supported on the rust side and might be only needed for remote funcs
class DebuggerRead(NamedTuple):
    requested: int


# TODO: This is not supported on the rust side and might be only needed for remote funcs
class DebuggerWrite(NamedTuple):
    payload: bytes


# TODO: This is not supported on the rust side and might be only needed for remote funcs
class DebuggerMessage(NamedTuple):
    stream_id: int
    action: Literal["paused", "attach", "detach"] | DebuggerRead | DebuggerWrite


# TODO: Might need to be supported differently through typed worker exceptions
class DependentOnError(Exception):
    def __init__(self, ident: int) -> None:
        self.ident = ident
