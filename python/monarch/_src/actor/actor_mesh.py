# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import collections
import contextvars
import functools
import importlib
import inspect
import itertools
import logging
import os
import random
import traceback

from abc import ABC, abstractmethod

from dataclasses import dataclass
from operator import mul
from traceback import extract_tb, StackSummary
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    cast,
    Concatenate,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    ParamSpec,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
)

from monarch._rust_bindings.monarch_hyperactor.actor import (
    PanicFlag,
    PythonMessage,
    PythonMessageKind,
)
from monarch._rust_bindings.monarch_hyperactor.actor_mesh import (
    ActorMeshMonitor,
    MonitoredOncePortReceiver,
    MonitoredPortReceiver,
    PythonActorMesh,
    PythonActorMeshRef,
)
from monarch._rust_bindings.monarch_hyperactor.mailbox import (
    Mailbox,
    OncePortReceiver,
    OncePortRef,
    PortRef,
)

if TYPE_CHECKING:
    from monarch._rust_bindings.monarch_hyperactor.mailbox import PortReceiverBase

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.selection import Selection as HySelection
from monarch._rust_bindings.monarch_hyperactor.shape import Point as HyPoint, Shape
from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch._rust_bindings.monarch_hyperactor.telemetry import enter_span, exit_span
from monarch._src.actor.allocator import LocalAllocator, ProcessAllocator
from monarch._src.actor.future import Future
from monarch._src.actor.pdb_wrapper import PdbWrapper

from monarch._src.actor.pickle import flatten, unflatten

from monarch._src.actor.shape import MeshTrait, NDSlice
from monarch._src.actor.sync_state import fake_sync_state
from typing_extensions import Self

if TYPE_CHECKING:
    from monarch._src.actor.proc_mesh import ProcMesh

logger: logging.Logger = logging.getLogger(__name__)

Allocator = ProcessAllocator | LocalAllocator

try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = bool(fbmake.get("par_style"))
except ImportError:
    IN_PAR = False

T1 = TypeVar("T1")
T2 = TypeVar("T2")


class Point(HyPoint, collections.abc.Mapping):
    pass


@dataclass
class MonarchContext:
    mailbox: Mailbox
    proc_id: str
    point: Point

    @staticmethod
    def get() -> "MonarchContext":
        return _context.get()


_context: contextvars.ContextVar[MonarchContext] = contextvars.ContextVar(
    "monarch.actor_mesh._context"
)


@dataclass
class DebugContext:
    pdb_wrapper: Optional[PdbWrapper] = None

    @staticmethod
    def get() -> "DebugContext":
        return _debug_context.get()

    @staticmethod
    def set(debug_context: "DebugContext") -> None:
        _debug_context.set(debug_context)


_debug_context: contextvars.ContextVar[DebugContext] = contextvars.ContextVar(
    "monarch.actor_mesh._debug_context"
)

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
A = TypeVar("A")

# keep this load balancing deterministic, but
# equally distributed.
_load_balancing_seed = random.Random(4)


Selection = Literal["all", "choose"] | int  # TODO: replace with real selection objects


def to_hy_sel(selection: Selection, shape: Shape) -> HySelection:
    if selection == "choose":
        dim = len(shape.labels)
        assert dim > 0
        query = ",".join(["?"] * dim)
        return HySelection.from_string(f"{query}")
    elif selection == "all":
        return HySelection.from_string("*")
    else:
        raise ValueError(f"invalid selection: {selection}")


# A temporary gate used by the PythonActorMesh/PythonActorMeshRef migration.
# We can use this gate to quickly roll back to using _ActorMeshRefImpl, if we
# encounter any issues with the migration.
#
# This should be removed once we confirm PythonActorMesh/PythonActorMeshRef is
# working correctly in production.
def _use_standin_mesh() -> bool:
    return bool(os.getenv("USE_STANDIN_ACTOR_MESH", default=False))


class ActorMeshProtocol(Protocol):
    """
    Protocol defining the common interface for actor mesh, mesh ref and _ActorMeshRefImpl.

    Note: We do not want to use ABC because _ActorMeshRefImpl already inherits
    from MeshTrait and we want to avoid multiple inheritance, especially when
    _ActorMeshRefImpl will be deleted soon.
    """

    @property
    def shape(self) -> Shape: ...

    @property
    def monitor(self) -> Optional[ActorMeshMonitor]: ...

    @property
    def proc_mesh(self) -> Optional["ProcMesh"]: ...

    @property
    def inner_name(Self) -> str: ...

    def cast(
        self,
        message: PythonMessage,
        selection: Selection,
        mailbox: Optional[Mailbox],
    ) -> None: ...

    def slice(self, **kwargs: Any) -> Self: ...

    def bind(self) -> Self: ...


class _PythonActorMeshAdapter(ActorMeshProtocol):
    """
    Adapter for PythonActorMesh to implement the normalized ActorMeshProtocol
    interface. This adapter also provides a convenient way to add states to
    the mesh on the python side, without changing the rust side implementation.

    Since PythonActorMesh cannot be pickled, this adapter also provides a
    custom pickling logic which bind the mesh to PythonActorMeshRef during
    pickling.
    """

    def __init__(self, inner: PythonActorMesh, proc_mesh: "ProcMesh") -> None:
        self._inner = inner
        self._proc_mesh = proc_mesh

    @property
    def shape(self) -> Shape:
        return self._inner.shape

    @property
    def monitor(self) -> Optional[ActorMeshMonitor]:
        return self._inner.monitor()

    @property
    def proc_mesh(self) -> Optional["ProcMesh"]:
        return self._proc_mesh

    @property
    def inner_name(self) -> str:
        return self._inner.__class__.__name__

    def cast(
        self,
        message: PythonMessage,
        selection: Selection,
        mailbox: Optional[Mailbox],
    ) -> None:
        self._inner.cast(to_hy_sel(selection, self.shape), message)

    def slice(self, **kwargs: Any) -> "ActorMeshProtocol":
        sliced: PythonActorMeshRef = self._inner.slice(**kwargs)
        return _PythonActorMeshRefAdapter(sliced, self.proc_mesh, self.monitor)

    def bind(self) -> "ActorMeshProtocol":
        # PythonActorMesh.bind returns PythonActorMeshRef
        mesh_ref: PythonActorMeshRef = self._inner.bind()
        return _PythonActorMeshRefAdapter(mesh_ref, self.proc_mesh, self.monitor)

    def __reduce_ex__(self, protocol: ...) -> Tuple[Any, Tuple[Any, ...]]:
        """
        Automatically pickle as a PythonActorMeshRef by binding the mesh.
        Unpicklable states such as proc_mesh and monitor are dropped as well.
        """
        mesh_ref = self._inner.bind()
        return _PythonActorMeshRefAdapter, (mesh_ref, None, None)


class _PythonActorMeshRefAdapter(ActorMeshProtocol):
    """
    Adapter for PythonActorMeshRef to implement the normalized ActorMeshProtocol interface. It is
    also used to store unpickable states such as proc_mesh and monitor. It is useful to have these
    unpickable states when pickling is not needed. For example, slicing a mesh will result in a
    mesh ref, and this mesh ref could be used by the same caller. This caller would expect the
    mesh ref to have the same supervision behavior as the original mesh. In this case, having the
    monitor field will be helpful.
    """

    def __init__(
        self,
        inner: PythonActorMeshRef,
        proc_mesh: "Optional[ProcMesh]",
        monitor: Optional[ActorMeshMonitor],
    ) -> None:
        self._inner = inner
        self._proc_mesh = proc_mesh
        self._monitor = monitor

    @property
    def shape(self) -> Shape:
        return self._inner.shape

    @property
    def monitor(self) -> Optional[ActorMeshMonitor]:
        return self._monitor

    @property
    def proc_mesh(self) -> Optional["ProcMesh"]:
        return self._proc_mesh

    @property
    def inner_name(self) -> str:
        return self._inner.__class__.__name__

    def cast(
        self,
        message: PythonMessage,
        selection: Selection,
        mailbox: Optional[Mailbox] = None,
    ) -> None:
        if mailbox is None:
            raise ValueError("mailbox is required for PythonActorMeshRef.cast()")
        self._inner.cast(mailbox, to_hy_sel(selection, self.shape), message)

    def slice(self, **kwargs: Any) -> "ActorMeshProtocol":
        sliced: PythonActorMeshRef = self._inner.slice(**kwargs)
        return _PythonActorMeshRefAdapter(sliced, self._proc_mesh, self._monitor)

    def bind(self) -> "ActorMeshProtocol":
        raise NotImplementedError("PythonActorMeshRef.bind() is not supported")

    def __reduce_ex__(self, protocol: ...) -> Tuple[Any, Tuple[Any, ...]]:
        """
        Dropping all unpickable states.
        """
        return _PythonActorMeshRefAdapter, (self._inner, None, None)


class _ActorIdAdapter(ActorMeshProtocol):
    def __init__(self, inner: ActorId) -> None:
        self._inner: ActorId = inner

    @property
    def shape(self) -> Shape:
        return singleton_shape

    @property
    def monitor(self) -> Optional[ActorMeshMonitor]:
        return None

    @property
    def proc_mesh(self) -> Optional["ProcMesh"]:
        return None

    @property
    def inner_name(self) -> str:
        return self._inner.__class__.__name__

    def cast(
        self,
        message: PythonMessage,
        selection: Selection,
        mailbox: Optional[Mailbox],
    ) -> None:
        if mailbox is None:
            raise ValueError("mailbox is required for ActorId")
        mailbox.post(self._inner, message)

    def slice(self, **kwargs: Any) -> Self:
        raise NotImplementedError("ActorId does not support slicing")

    def bind(self) -> Self:
        raise NotImplementedError("ActorId does not support binding")


# standin class for whatever is the serializable python object we use
# to name an actor mesh. Hacked up today because ActorMesh
# isn't plumbed to non-clients
class _ActorMeshRefImpl(MeshTrait, ActorMeshProtocol):
    def __init__(
        self,
        mailbox: Mailbox,
        hy_actor_mesh: Optional[PythonActorMesh],
        proc_mesh: "Optional[ProcMesh]",
        shape: Shape,
        actor_ids: List[ActorId],
    ) -> None:
        if not _use_standin_mesh():
            raise ValueError(
                "ActorMeshRefImpl should only be used when USE_STANDIN_ACTOR_MESH is set"
            )
        self._mailbox = mailbox
        self._actor_mesh = hy_actor_mesh
        # actor meshes do not have a way to look this up at the moment,
        # so we fake it here
        self._proc_mesh = proc_mesh
        self._shape = shape
        self._please_replace_me_actor_ids = actor_ids

    @staticmethod
    def from_hyperactor_mesh(
        mailbox: Mailbox, hy_actor_mesh: PythonActorMesh, proc_mesh: "ProcMesh"
    ) -> "_ActorMeshRefImpl":
        shape: Shape = hy_actor_mesh.shape
        return _ActorMeshRefImpl(
            mailbox,
            hy_actor_mesh,
            proc_mesh,
            hy_actor_mesh.shape,
            [cast(ActorId, hy_actor_mesh.get(i)) for i in range(len(shape))],
        )

    @property
    def monitor(self) -> Optional[ActorMeshMonitor]:
        return self._actor_mesh.monitor() if self._actor_mesh is not None else None

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def _ndslice(self) -> NDSlice:
        return self._shape.ndslice

    @property
    def _labels(self) -> Iterable[str]:
        return self._shape.labels

    def _new_with_shape(self, shape: Shape) -> "_ActorMeshRefImpl":
        return _ActorMeshRefImpl(
            self._mailbox, None, None, shape, self._please_replace_me_actor_ids
        )

    def __getstate__(
        self,
    ) -> Tuple[Shape, List[ActorId], Mailbox]:
        return self._shape, self._please_replace_me_actor_ids, self._mailbox

    def __setstate__(
        self,
        state: Tuple[Shape, List[ActorId], Mailbox],
    ) -> None:
        self._actor_mesh = None
        self._shape, self._please_replace_me_actor_ids, self._mailbox = state

    def _check_state(self) -> None:
        # This is temporary until we have real cast integration here. We need to actively check
        # supervision error here is because all communication is done through direct mailbox sending
        # and not through comm actor casting.
        # TODO: remove this when casting integration is done.
        if self._actor_mesh is not None:
            event = self._actor_mesh.get_supervision_event()
            if event is not None:
                raise SupervisionError(f"actor mesh is not in a healthy state: {event}")

    def cast(
        self,
        message: PythonMessage,
        selection: Selection,
        mailbox: Optional[Mailbox],
    ) -> None:
        self._check_state()

        # TODO: use the actual actor mesh when available. We cannot currently use it
        # directly because we risk bifurcating the message delivery paths from the same
        # client, since slicing the mesh will produce a reference, which calls actors
        # directly. The reason these paths are bifurcated is that actor meshes will
        # use multicasting, while direct actor comms do not. Separately we need to decide
        # whether actor meshes are ordered with actor references.
        #
        # The fix is to provide a first-class reference into Python, and always call "cast"
        # on it, including for load balanced requests.
        if selection == "choose":
            idx = _load_balancing_seed.randrange(len(self._shape))
            actor_rank = self._shape.ndslice[idx]
            self._mailbox.post(self._please_replace_me_actor_ids[actor_rank], message)
        elif selection == "all":
            # replace me with actual remote actor mesh
            call_shape = Shape(
                self._shape.labels, NDSlice.new_row_major(self._shape.ndslice.sizes)
            )
            for i, rank in enumerate(self._shape.ranks()):
                self._mailbox.post_cast(
                    self._please_replace_me_actor_ids[rank],
                    i,
                    call_shape,
                    message,
                )
        elif isinstance(selection, int):
            try:
                self._mailbox.post(
                    self._please_replace_me_actor_ids[selection], message
                )
            except IndexError:
                raise IndexError(
                    f"Tried to send to an out-of-range rank {selection}: "
                    f"mesh has {len(self._please_replace_me_actor_ids)} elements."
                )
        else:
            raise ValueError(f"invalid selection: {selection}")

    def bind(self) -> Self:
        return self

    def __len__(self) -> int:
        return len(self._shape)


class Extent(NamedTuple):
    labels: Sequence[str]
    sizes: Sequence[int]

    @property
    def nelements(self) -> int:
        return functools.reduce(mul, self.sizes, 1)

    def __str__(self) -> str:
        return str(dict(zip(self.labels, self.sizes)))


class Endpoint(ABC, Generic[P, R]):
    @abstractmethod
    def _send(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        port: "Optional[Port]" = None,
        selection: Selection = "all",
    ) -> Extent:
        """
        Implements sending a message to the endpoint. The return value of the endpoint will
        be sent to port if provided. If port is not provided, the return will be dropped,
        and any exception will cause the actor to fail.

        The return value is the (multi-dimension) size of the actors that were sent a message.
        For ActorEndpoints this will be the actor_meshes size. For free-function endpoints,
        this will be the size of the currently active proc_mesh.
        """
        pass

    @abstractmethod
    def _port(self, once: bool = False) -> "PortTuple[R]":
        pass

    # the following are all 'adverbs' or different ways to handle the
    # return values of this endpoint. Adverbs should only ever take *args, **kwargs
    # of the original call. If we want to add syntax sugar for something that needs additional
    # arguments, it should be implemented as function indepdendent of endpoint like `send`
    # and `Accumulator`
    def choose(self, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        """
        Load balanced sends a message to one chosen actor and awaits a result.

        Load balanced RPC-style entrypoint for request/response messaging.
        """
        p, r = port(self, once=True)
        # pyre-ignore
        self._send(args, kwargs, port=p, selection="choose")
        return r.recv()

    def call_one(self, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        p, r = port(self, once=True)
        # pyre-ignore
        extent = self._send(args, kwargs, port=p, selection="choose")
        if extent.nelements != 1:
            raise ValueError(
                f"Can only use 'call_one' on a single Actor but this actor has shape {extent}"
            )
        return r.recv()

    def call(self, *args: P.args, **kwargs: P.kwargs) -> "Future[ValueMesh[R]]":
        p, r = ranked_port(self)
        # pyre-ignore
        extent = self._send(args, kwargs, port=p)

        async def process() -> ValueMesh[R]:
            results: Dict[int, R] = {}
            for _ in range(extent.nelements):
                rank, value = await r.recv()
                results[rank] = value
            call_shape = Shape(
                extent.labels,
                NDSlice.new_row_major(extent.sizes),
            )
            sorted_values = [results[rank] for rank in sorted(results)]
            return ValueMesh(call_shape, sorted_values)

        return Future(impl=process, requires_loop=False)

    async def stream(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[R, R]:
        """
        Broadcasts to all actors and yields their responses as a stream / generator.

        This enables processing results from multiple actors incrementally as
        they become available. Returns an async generator of response values.
        """
        p, r = port(self)
        # pyre-ignore
        extent = self._send(args, kwargs, port=p)
        for _ in range(extent.nelements):
            yield await r.recv()

    def broadcast(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """
        Fire-and-forget broadcast to all actors without waiting for actors to
        acknowledge receipt.

        In other words, the return of this method does not guarrantee the
        delivery of the message.
        """
        # pyre-ignore
        send(self, args, kwargs)


class ActorEndpoint(Endpoint[P, R]):
    def __init__(
        self,
        actor_mesh: ActorMeshProtocol,
        name: str,
        impl: Callable[Concatenate[Any, P], Awaitable[R]],
        mailbox: Mailbox,
    ) -> None:
        self._actor_mesh = actor_mesh
        self._name = name
        self._signature: inspect.Signature = inspect.signature(impl)
        self._mailbox = mailbox

    def _send(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        port: "Optional[Port]" = None,
        selection: Selection = "all",
    ) -> Extent:
        """
        Fire-and-forget broadcast invocation of the endpoint across all actors in the mesh.

        This sends the message to all actors but does not wait for any result.
        """
        self._signature.bind(None, *args, **kwargs)
        objects, bytes = flatten((args, kwargs), _is_ref_or_mailbox)
        refs = [obj for obj in objects if hasattr(obj, "__monarch_ref__")]
        if not refs:
            message = PythonMessage(
                PythonMessageKind.CallMethod(
                    self._name, None if port is None else port._port_ref
                ),
                bytes,
            )
            self._actor_mesh.cast(message, selection, self._mailbox)
        else:
            importlib.import_module("monarch." + "mesh_controller").actor_send(
                self, bytes, refs, port, selection
            )
        shape = self._actor_mesh.shape
        return Extent(shape.labels, shape.ndslice.sizes)

    def _port(self, once: bool = False) -> "PortTuple[R]":
        monitor = self._actor_mesh.monitor
        return PortTuple.create(self._mailbox, monitor, once)


class Accumulator(Generic[P, R, A]):
    def __init__(
        self, endpoint: Endpoint[P, R], identity: A, combine: Callable[[A, R], A]
    ) -> None:
        self._endpoint: Endpoint[P, R] = endpoint
        self._identity: A = identity
        self._combine: Callable[[A, R], A] = combine

    def accumulate(self, *args: P.args, **kwargs: P.kwargs) -> "Future[A]":
        gen: AsyncGenerator[R, R] = self._endpoint.stream(*args, **kwargs)

        async def impl() -> A:
            value = self._identity
            async for x in gen:
                value = self._combine(value, x)
            return value

        return Future(impl=impl)


class ValueMesh(MeshTrait, Generic[R]):
    """
    Container of return values, indexed by rank.
    """

    def __init__(self, shape: Shape, values: List[R]) -> None:
        self._shape = shape
        self._values = values

    def _new_with_shape(self, shape: Shape) -> "ValueMesh[R]":
        return ValueMesh(shape, self._values)

    def item(self, **kwargs) -> R:
        coordinates = [kwargs.pop(label) for label in self._labels]
        if kwargs:
            raise KeyError(f"item has extra dimensions: {list(kwargs.keys())}")

        return self._values[self._ndslice.nditem(coordinates)]

    def items(self) -> Iterable[Tuple[Point, R]]:
        for rank in self._shape.ranks():
            yield Point(rank, self._shape), self._values[rank]

    def __iter__(self) -> Iterator[Tuple[Point, R]]:
        return iter(self.items())

    def __len__(self) -> int:
        return len(self._shape)

    def __repr__(self) -> str:
        return f"ValueMesh({self._shape})"

    @property
    def _ndslice(self) -> NDSlice:
        return self._shape.ndslice

    @property
    def _labels(self) -> Iterable[str]:
        return self._shape.labels


def send(
    endpoint: Endpoint[P, R],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    port: "Optional[Port]" = None,
    selection: Selection = "all",
) -> None:
    """
    Fire-and-forget broadcast invocation of the endpoint across all actors in the mesh.

    This sends the message to all actors but does not wait for any result.
    """
    endpoint._send(args, kwargs, port, selection)


class EndpointProperty(Generic[P, R]):
    @overload
    def __init__(self, method: Callable[Concatenate[Any, P], Awaitable[R]]) -> None: ...

    @overload
    def __init__(self, method: Callable[Concatenate[Any, P], R]) -> None: ...

    def __init__(self, method: Any) -> None:
        self._method = method

    def __get__(self, instance, owner) -> Endpoint[P, R]:
        # this is a total lie, but we have to actually
        # recognize this was defined as an endpoint,
        # and also lookup the method
        return cast(Endpoint[P, R], self)


@overload
def endpoint(
    method: Callable[Concatenate[Any, P], Awaitable[R]],
) -> EndpointProperty[P, R]: ...


@overload
def endpoint(
    method: Callable[Concatenate[Any, P], R],
) -> EndpointProperty[P, R]: ...


def endpoint(method):
    return EndpointProperty(method)


class Port(Generic[R]):
    def __init__(
        self,
        port_ref: PortRef | OncePortRef | None,
        mailbox: Mailbox,
        rank: Optional[int],
    ) -> None:
        self._port_ref = port_ref
        self._mailbox = mailbox
        self._rank = rank

    def send(self, obj: R) -> None:
        if self._port_ref is None:
            return
        self._port_ref.send(
            self._mailbox,
            PythonMessage(PythonMessageKind.Result(self._rank), _pickle(obj)),
        )

    def exception(self, obj: Exception) -> None:
        # we deliver each error exactly once, so if there is no port to respond to,
        # the error is sent to the current actor as an exception.
        if self._port_ref is None:
            raise obj from None
        self._port_ref.send(
            self._mailbox,
            PythonMessage(PythonMessageKind.Exception(self._rank), _pickle(obj)),
        )


R = TypeVar("R")

T = TypeVar("T")

if TYPE_CHECKING:
    # Python <= 3.10 cannot inherit from Generic[R] and NamedTuple at the same time.
    # we only need it for type checking though, so copypasta it until 3.11.
    class PortTuple(NamedTuple, Generic[R]):
        sender: "Port[R]"
        receiver: "PortReceiver[R]"

        @staticmethod
        def create(
            mailbox: Mailbox, monitor: Optional[ActorMeshMonitor], once: bool = False
        ) -> "PortTuple[Any]":
            handle, receiver = mailbox.open_once_port() if once else mailbox.open_port()
            port_ref = handle.bind()
            if monitor is not None:
                receiver = (
                    MonitoredOncePortReceiver(receiver, monitor)
                    if isinstance(receiver, OncePortReceiver)
                    else MonitoredPortReceiver(receiver, monitor)
                )

            return PortTuple(
                Port(port_ref, mailbox, rank=None),
                PortReceiver(mailbox, receiver),
            )
else:

    class PortTuple(NamedTuple):
        sender: "Port[Any]"
        receiver: "PortReceiver[Any]"

        @staticmethod
        def create(
            mailbox: Mailbox, monitor: Optional[ActorMeshMonitor], once: bool = False
        ) -> "PortTuple[Any]":
            handle, receiver = mailbox.open_once_port() if once else mailbox.open_port()
            port_ref = handle.bind()
            if monitor is not None:
                receiver = (
                    MonitoredOncePortReceiver(receiver, monitor)
                    if isinstance(receiver, OncePortReceiver)
                    else MonitoredPortReceiver(receiver, monitor)
                )

            return PortTuple(
                Port(port_ref, mailbox, rank=None),
                PortReceiver(mailbox, receiver),
            )


# advance lower-level API for sending messages. This is intentially
# not part of the Endpoint API because they way it accepts arguments
# and handles concerns is different.
def port(endpoint: Endpoint[P, R], once: bool = False) -> "PortTuple[R]":
    return endpoint._port(once)


def ranked_port(
    endpoint: Endpoint[P, R], once: bool = False
) -> Tuple["Port[R]", "RankedPortReceiver[R]"]:
    p, receiver = port(endpoint, once)
    return p, RankedPortReceiver[R](receiver._mailbox, receiver._receiver)


class PortReceiver(Generic[R]):
    def __init__(
        self,
        mailbox: Mailbox,
        receiver: "PortReceiverBase",
    ) -> None:
        self._mailbox: Mailbox = mailbox
        self._receiver = receiver

    async def _recv(self) -> R:
        return self._process(await self._receiver.recv_task())

    def _process(self, msg: PythonMessage) -> R:
        # TODO: Try to do something more structured than a cast here
        payload = cast(R, unflatten(msg.message, itertools.repeat(self._mailbox)))
        match msg.kind:
            case PythonMessageKind.Result():
                return payload
            case PythonMessageKind.Exception():
                raise cast(Exception, payload)
            case _:
                raise ValueError(f"Unexpected message kind: {msg.kind}")

    def recv(self) -> "Future[R]":
        return Future(impl=lambda: self._recv(), requires_loop=False)


class RankedPortReceiver(PortReceiver[Tuple[int, R]]):
    def _process(self, msg: PythonMessage) -> Tuple[int, R]:
        rank = getattr(msg.kind, "rank", None)
        if rank is None:
            raise ValueError(
                f"RankedPort receiver got a message without a rank {msg}",
            )
        return rank, super()._process(msg)


singleton_shape = Shape([], NDSlice(offset=0, sizes=[], strides=[]))


# Currently the synchronous function of actors are run on a python thread that has an active event loop.
# Technically it is unsafe for them to block at all because they will block the loop of other
# calls, so all calls to .get() should be failing.
# But in the meantime, to implement get() by reusing async functions,
#  we need to signal to the consumer of the PythonTask object that the thread really isn't in an async context.
# We do this by blanking out the running event loop during the call to the synchronous actor function.


class _Actor:
    """
    This is the message handling implementation of a Python actor.

    The layering goes:
        Rust `PythonActor` -> `_Actor` -> user-provided `Actor` instance

    Messages are received from the Rust backend, and forwarded to the `handle`
    methods on this class.

    This class wraps the actual `Actor` instance provided by the user, and
    routes messages to it, managing argument serialization/deserialization and
    error handling.
    """

    def __init__(self) -> None:
        self.instance: object | None = None
        # TODO: (@pzhang) remove this with T229200522
        self._saved_error: ActorError | None = None

    async def handle(
        self,
        mailbox: Mailbox,
        rank: int,
        shape: Shape,
        message: PythonMessage,
        panic_flag: PanicFlag,
        local_state: Iterable[Any],
    ) -> None:
        match message.kind:
            case PythonMessageKind.CallMethod(response_port=response_port):
                pass
            case _:
                response_port = None
        # response_port can be None. If so, then sending to port will drop the response,
        # and raise any exceptions to the caller.
        port = Port(response_port, mailbox, rank)
        try:
            ctx: MonarchContext = MonarchContext(
                mailbox, mailbox.actor_id.proc_id, Point(rank, shape)
            )
            _context.set(ctx)

            DebugContext.set(DebugContext())

            args, kwargs = unflatten(message.message, local_state)

            match message.kind:
                case PythonMessageKind.CallMethod(name=name):
                    method = name
                    if method == "__init__":
                        Class, *args = args
                        try:
                            self.instance = Class(*args, **kwargs)
                        except Exception as e:
                            self._saved_error = ActorError(
                                e, f"Remote actor {Class}.__init__ call failed."
                            )
                            raise e
                        port.send(None)
                        return None
                case _:
                    raise ValueError(f"Unexpected message kind: {message.kind}")

            if self.instance is None:
                # This could happen because of the following reasons. Both
                # indicates a possible bug in the framework:
                # 1. the execution of the previous message for "__init__" failed,
                #    but that error is not surfaced to the caller.
                #      - TODO(T229200522): there is a known bug. fix it.
                # 2. this message is delivered to this actor before the previous
                #    message of "__init__" is delivered. Out-of-order delivery
                #    should never happen. It indicates either a bug in the
                #    message delivery mechanism, or the framework accidentally
                #    mixed the usage of cast and direct send.
                error_message = f"Actor object is missing when executing method {method} on actor {mailbox.actor_id}."
                if self._saved_error is not None:
                    error_message += (
                        f" This is likely due to an earlier error: {self._saved_error}"
                    )
                raise AssertionError(error_message)
            the_method = getattr(self.instance, method)._method

            if inspect.iscoroutinefunction(the_method):

                async def instrumented():
                    enter_span(
                        the_method.__module__,
                        method,
                        str(ctx.mailbox.actor_id),
                    )
                    try:
                        result = await the_method(self.instance, *args, **kwargs)
                        self._maybe_exit_debugger()
                    except Exception as e:
                        logging.critical(
                            "Unhandled exception in actor endpoint",
                            exc_info=e,
                        )
                        raise e
                    exit_span()
                    return result

                result = await instrumented()
            else:
                enter_span(the_method.__module__, method, str(ctx.mailbox.actor_id))
                with fake_sync_state():
                    result = the_method(self.instance, *args, **kwargs)
                self._maybe_exit_debugger()
                exit_span()

            port.send(result)
        except Exception as e:
            self._post_mortem_debug(e.__traceback__)
            traceback.print_exc()
            port.exception(ActorError(e))
        except BaseException as e:
            self._post_mortem_debug(e.__traceback__)
            # A BaseException can be thrown in the case of a Rust panic.
            # In this case, we need a way to signal the panic to the Rust side.
            # See [Panics in async endpoints]
            try:
                panic_flag.signal_panic(e)
            except Exception:
                # The channel might be closed if the Rust side has already detected the error
                pass
            raise

    def _maybe_exit_debugger(self, do_continue=True) -> None:
        if (pdb_wrapper := DebugContext.get().pdb_wrapper) is not None:
            if do_continue:
                pdb_wrapper.clear_all_breaks()
                pdb_wrapper.do_continue("")
            pdb_wrapper.end_debug_session()
        DebugContext.set(DebugContext())

    def _post_mortem_debug(self, exc_tb) -> None:
        from monarch._src.actor.debugger import DebugManager

        if (pdb_wrapper := DebugContext.get().pdb_wrapper) is not None:
            with fake_sync_state():
                ctx = MonarchContext.get()
                pdb_wrapper = PdbWrapper(
                    ctx.point.rank,
                    ctx.point.shape.coordinates(ctx.point.rank),
                    ctx.mailbox.actor_id,
                    DebugManager.ref().get_debug_client.call_one().get(),
                )
                DebugContext.set(DebugContext(pdb_wrapper))
                pdb_wrapper.post_mortem(exc_tb)
                self._maybe_exit_debugger(do_continue=False)


def _is_mailbox(x: object) -> bool:
    if hasattr(x, "__monarch_ref__"):
        raise NotImplementedError(
            "Sending monarch tensor references directly to a port."
        )
    return isinstance(x, Mailbox)


def _is_ref_or_mailbox(x: object) -> bool:
    return hasattr(x, "__monarch_ref__") or isinstance(x, Mailbox)


def _pickle(obj: object) -> bytes:
    _, msg = flatten(obj, _is_mailbox)
    return msg


class Actor(MeshTrait):
    @functools.cached_property
    def logger(cls) -> logging.Logger:
        lgr = logging.getLogger(cls.__class__.__name__)
        lgr.setLevel(logging.DEBUG)
        return lgr

    @property
    def _ndslice(self) -> NDSlice:
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )

    @property
    def _labels(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )

    def _new_with_shape(self, shape: Shape) -> Self:
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )


class ActorMesh(MeshTrait, Generic[T]):
    def __init__(
        self,
        Class: Type[T],
        inner: ActorMeshProtocol,
        mailbox: Mailbox,
    ) -> None:
        self.__name__: str = Class.__name__
        self._class: Type[T] = Class
        self._inner: ActorMeshProtocol = inner
        self._mailbox: Mailbox = mailbox
        for attr_name in dir(self._class):
            attr_value = getattr(self._class, attr_name, None)
            if isinstance(attr_value, EndpointProperty):
                setattr(
                    self,
                    attr_name,
                    ActorEndpoint(
                        self._inner,
                        attr_name,
                        attr_value._method,
                        self._mailbox,
                    ),
                )

    def __getattr__(self, name: str) -> Any:
        # This method is called when an attribute is not found
        # For linting purposes, we need to tell the type checker that any attribute
        # could be an endpoint that's dynamically added at runtime
        # At runtime, we still want to raise AttributeError for truly missing attributes

        # Check if this is a method on the underlying class
        if hasattr(self._class, name):
            attr = getattr(self._class, name)
            if isinstance(attr, EndpointProperty):
                # Dynamically create the endpoint
                endpoint = ActorEndpoint(
                    self._inner,
                    name,
                    attr._method,
                    self._mailbox,
                )
                # Cache it for future use
                setattr(self, name, endpoint)
                return endpoint

        # If we get here, it's truly not found
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @classmethod
    def create(
        cls,
        Class: Type[T],
        actor_mesh: PythonActorMesh,
        mailbox: Mailbox,
        proc_mesh: "ProcMesh",
        # args and kwargs are passed to the __init__ method of the user defined
        # python actor object.
        *args: Any,
        **kwargs: Any,
    ) -> "ActorMesh[T]":
        if _use_standin_mesh():
            wrapper = _ActorMeshRefImpl.from_hyperactor_mesh(
                mailbox, actor_mesh, proc_mesh
            )
        else:
            wrapper = _PythonActorMeshAdapter(actor_mesh, proc_mesh)
        mesh = cls(Class, wrapper, mailbox)

        # send __init__ message to the mesh to initialize the user defined
        # python actor object.
        async def null_func(*_args: Iterable[Any], **_kwargs: Dict[str, Any]) -> None:
            return None

        ep = ActorEndpoint(
            mesh._inner,
            "__init__",
            null_func,
            mesh._mailbox,
        )
        send(ep, (mesh._class, *args), kwargs)

        return mesh

    @classmethod
    def from_actor_id(
        cls,
        Class: Type[T],
        actor_id: ActorId,
        mailbox: Mailbox,
    ) -> "ActorMesh[T]":
        return cls(Class, _ActorIdAdapter(actor_id), mailbox)

    def bind(self) -> "ActorMesh[T]":
        if not isinstance(self._inner, _PythonActorMeshAdapter) or not isinstance(
            self._inner, _ActorMeshRefImpl
        ):
            raise AttributeError(
                "{msg} is only available on PythonActorMesh or _ActorMeshRefImpl, but got {self._inner.inner_name}"
            )
        mesh_ref = self._inner.bind()
        return ActorMesh(self._class, mesh_ref, self._mailbox)

    def __reduce_ex__(self, protocol: ...) -> "Tuple[Type[ActorMesh], Tuple[Any, ...]]":
        return ActorMesh, (
            self._class,
            self._inner,
            self._mailbox,
        )

    @property
    def proc_mesh(self) -> "Optional[ProcMesh]":
        return self._inner.proc_mesh

    @property
    def _ndslice(self) -> NDSlice:
        return self._inner.shape.ndslice

    @property
    def _labels(self) -> Iterable[str]:
        return self._inner.shape.labels

    def _new_with_shape(self, shape: Shape) -> "ActorMesh":
        raise NotImplementedError(
            "should not be called because def slice is overridden"
        )

    def slice(self, **kwargs) -> "ActorMesh[T]":
        sliced = self._inner.slice(**kwargs)
        return ActorMesh(self._class, sliced, self._mailbox)

    def __repr__(self) -> str:
        return f"ActorMesh(class={self._class}, shape={self._inner.shape}), inner={self._inner.inner_name})"


class ActorError(Exception):
    """
    Deterministic problem with the user's code.
    For example, an OOM resulting in trying to allocate too much GPU memory, or violating
    some invariant enforced by the various APIs.
    """

    def __init__(
        self,
        exception: Exception,
        message: str = "A remote actor call has failed.",
    ) -> None:
        self.exception = exception
        self.actor_mesh_ref_frames: StackSummary = extract_tb(exception.__traceback__)
        self.message = message

    def __str__(self) -> str:
        exe = str(self.exception)
        actor_mesh_ref_tb = "".join(traceback.format_list(self.actor_mesh_ref_frames))
        return (
            f"{self.message}\n"
            f"Traceback of where the remote call failed (most recent call last):\n{actor_mesh_ref_tb}{type(self.exception).__name__}: {exe}"
        )


def current_actor_name() -> str:
    return str(MonarchContext.get().mailbox.actor_id)


def current_rank() -> Point:
    ctx = MonarchContext.get()
    return ctx.point


def current_size() -> Dict[str, int]:
    ctx = MonarchContext.get()
    return dict(zip(ctx.point.shape.labels, ctx.point.shape.ndslice.sizes))
