# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import collections
import contextvars
import functools
import inspect
import itertools
import logging
import random
import traceback

from dataclasses import dataclass
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
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
)

from monarch._rust_bindings.monarch_hyperactor.actor import (
    MethodSpecifier,
    PanicFlag,
    PythonMessage,
    PythonMessageKind,
)
from monarch._rust_bindings.monarch_hyperactor.actor_mesh import PythonActorMesh
from monarch._rust_bindings.monarch_hyperactor.mailbox import (
    Mailbox,
    OncePortReceiver,
    OncePortRef,
    PortReceiver as HyPortReceiver,
    PortRef,
)

if TYPE_CHECKING:
    from monarch._rust_bindings.monarch_hyperactor.actor import PortProtocol
    from monarch._rust_bindings.monarch_hyperactor.mailbox import PortReceiverBase

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.shape import Point as HyPoint, Shape
from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch._rust_bindings.monarch_hyperactor.telemetry import enter_span, exit_span
from monarch._src.actor.allocator import LocalAllocator, ProcessAllocator
from monarch._src.actor.endpoint import (
    Endpoint,
    EndpointProperty,
    Extent,
    NotAnEndpoint,
    Propagator,
    Selection,
)
from monarch._src.actor.future import Future
from monarch._src.actor.pdb_wrapper import PdbWrapper

from monarch._src.actor.pickle import flatten, unflatten

from monarch._src.actor.shape import MeshTrait, NDSlice
from monarch._src.actor.sync_state import fake_sync_state

from monarch._src.actor.tensor_engine_shim import actor_rref, actor_send

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


# standin class for whatever is the serializable python object we use
# to name an actor mesh. Hacked up today because ActorMesh
# isn't plumbed to non-clients
class _ActorMeshRefImpl:
    def __init__(
        self,
        mailbox: Mailbox,
        hy_actor_mesh: Optional[PythonActorMesh],
        proc_mesh: "Optional[ProcMesh]",
        shape: Shape,
        actor_ids: List[ActorId],
    ) -> None:
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

    @staticmethod
    def from_actor_id(mailbox: Mailbox, actor_id: ActorId) -> "_ActorMeshRefImpl":
        return _ActorMeshRefImpl(mailbox, None, None, singleton_shape, [actor_id])

    @staticmethod
    def from_actor_ref_with_shape(
        ref: "_ActorMeshRefImpl", shape: Shape
    ) -> "_ActorMeshRefImpl":
        return _ActorMeshRefImpl(
            ref._mailbox, None, None, shape, ref._please_replace_me_actor_ids
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
            if self._actor_mesh.stopped:
                raise SupervisionError(
                    "actor mesh is not in a healthy state: `ActorMesh` has been stopped"
                )

            event = self._actor_mesh.get_supervision_event()
            if event is not None:
                raise SupervisionError(f"actor mesh is not in a healthy state: {event}")

    def send(self, rank: int, message: PythonMessage) -> None:
        self._check_state()
        actor = self._please_replace_me_actor_ids[rank]
        self._mailbox.post(actor, message)

    def cast(
        self,
        message: PythonMessage,
        selection: Selection,
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

    def __len__(self) -> int:
        return len(self._shape)

    @property
    def _name_pid(self):
        actor_id0 = self._please_replace_me_actor_ids[0]
        return actor_id0.actor_name, actor_id0.pid

    async def stop(self):
        await self._actor_mesh.stop()


class ActorEndpoint(Endpoint[P, R]):
    def __init__(
        self,
        actor_mesh_ref: _ActorMeshRefImpl,
        name: MethodSpecifier,
        impl: Callable[Concatenate[Any, P], Awaitable[R]],
        mailbox: Mailbox,
        propagator: Propagator,
        explicit_response_port: bool,
    ) -> None:
        super().__init__(propagator)
        self._actor_mesh = actor_mesh_ref
        self._name = name
        self._signature: inspect.Signature = inspect.signature(impl)
        self._mailbox = mailbox
        self._explicit_response_port = explicit_response_port

    def _supervise(self, r: HyPortReceiver | OncePortReceiver) -> Any:
        mesh = self._actor_mesh._actor_mesh
        return r if mesh is None else mesh.supervise(r)

    def _call_name(self) -> Any:
        return self._name

    def _check_arguments(self, args, kwargs):
        if self._explicit_response_port:
            self._signature.bind(None, None, *args, **kwargs)
        else:
            self._signature.bind(None, *args, **kwargs)

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
        self._check_arguments(args, kwargs)
        objects, bytes = flatten((args, kwargs), _is_ref_or_mailbox)
        if all(not hasattr(obj, "__monarch_ref__") for obj in objects):
            message = PythonMessage(
                PythonMessageKind.CallMethod(
                    self._name, None if port is None else port._port_ref
                ),
                bytes,
            )
            self._actor_mesh.cast(message, selection)
        else:
            actor_send(self, bytes, objects, port, selection)
        shape = self._actor_mesh._shape
        return Extent(shape.labels, shape.ndslice.sizes)

    def _port(self, once: bool = False) -> "PortTuple[R]":
        p, r = PortTuple.create(self._mailbox, once)
        if TYPE_CHECKING:
            assert isinstance(
                r._receiver, (HyPortReceiver | OncePortReceiver)
            ), "unexpected receiver type"
        return PortTuple(p, PortReceiver(self._mailbox, self._supervise(r._receiver)))

    def _rref(self, args, kwargs):
        self._check_arguments(args, kwargs)
        refs, bytes = flatten((args, kwargs), _is_ref_or_mailbox)

        return actor_rref(self, bytes, refs)


@overload
def as_endpoint(
    not_an_endpoint: Callable[P, R],
    *,
    propagate: Propagator = None,
    explicit_response_port: Literal[False] = False,
) -> Endpoint[P, R]: ...


@overload
def as_endpoint(
    not_an_endpoint: Callable[Concatenate["PortProtocol[R]", P], None],
    *,
    propagate: Propagator = None,
    explicit_response_port: Literal[True],
) -> Endpoint[P, R]: ...


def as_endpoint(
    not_an_endpoint: Any,
    *,
    propagate: Propagator = None,
    explicit_response_port: bool = False,
):
    if not isinstance(not_an_endpoint, NotAnEndpoint):
        raise ValueError("expected an method of a spawned actor")
    kind = (
        MethodSpecifier.ExplicitPort
        if explicit_response_port
        else MethodSpecifier.ReturnsResponse
    )
    return ActorEndpoint(
        not_an_endpoint._ref._actor_mesh_ref,
        kind(not_an_endpoint._name),
        getattr(not_an_endpoint._ref, not_an_endpoint._name),
        not_an_endpoint._ref._mailbox,
        propagate,
        explicit_response_port,
    )


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


class Port(Generic[R]):
    def __init__(
        self,
        port_ref: PortRef | OncePortRef,
        mailbox: Mailbox,
        rank: Optional[int],
    ) -> None:
        self._port_ref = port_ref
        self._mailbox = mailbox
        self._rank = rank

    def send(self, obj: R) -> None:
        self._port_ref.send(
            self._mailbox,
            PythonMessage(PythonMessageKind.Result(self._rank), _pickle(obj)),
        )

    def exception(self, obj: Exception) -> None:
        # we deliver each error exactly once, so if there is no port to respond to,
        # the error is sent to the current actor as an exception.
        self._port_ref.send(
            self._mailbox,
            PythonMessage(PythonMessageKind.Exception(self._rank), _pickle(obj)),
        )


class DroppingPort:
    """
    Used in place of a real port when the message has no response port.
    Makes sure any exception sent to it causes the actor to report an exception.
    """

    def __init__(self):
        pass

    def send(self, obj: Any) -> None:
        pass

    def exception(self, obj: Exception) -> None:
        # we deliver each error exactly once, so if there is no port to respond to,
        # the error is sent to the current actor as an exception.
        raise obj from None


R = TypeVar("R")

T = TypeVar("T")

if TYPE_CHECKING:
    # Python <= 3.10 cannot inherit from Generic[R] and NamedTuple at the same time.
    # we only need it for type checking though, so copypasta it until 3.11.
    class PortTuple(NamedTuple, Generic[R]):
        sender: "Port[R]"
        receiver: "PortReceiver[R]"

        @staticmethod
        def create(mailbox: Mailbox, once: bool = False) -> "PortTuple[Any]":
            handle, receiver = mailbox.open_once_port() if once else mailbox.open_port()
            port_ref = handle.bind()
            return PortTuple(
                Port(port_ref, mailbox, rank=None),
                PortReceiver(mailbox, receiver),
            )
else:

    class PortTuple(NamedTuple):
        sender: "Port[Any]"
        receiver: "PortReceiver[Any]"

        @staticmethod
        def create(mailbox: Mailbox, once: bool = False) -> "PortTuple[Any]":
            handle, receiver = mailbox.open_once_port() if once else mailbox.open_port()
            port_ref = handle.bind()
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
        method_spec: MethodSpecifier,
        message: bytes,
        panic_flag: PanicFlag,
        local_state: Iterable[Any],
        port: "PortProtocol",
    ) -> None:
        # response_port can be None. If so, then sending to port will drop the response,
        # and raise any exceptions to the caller.
        try:
            ctx: MonarchContext = MonarchContext(
                mailbox, mailbox.actor_id.proc_id, Point(rank, shape)
            )
            _context.set(ctx)

            DebugContext.set(DebugContext())

            args, kwargs = unflatten(message, local_state)

            match method_spec:
                case MethodSpecifier.Init():
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
                case MethodSpecifier.ReturnsResponse(name=method):
                    pass
                case MethodSpecifier.ExplicitPort(name=method):
                    args = (port, *args)
                    port = DroppingPort()

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
            the_method = getattr(self.instance, method)
            if isinstance(the_method, EndpointProperty):
                module = the_method._method.__module__
                the_method = functools.partial(the_method._method, self.instance)
            else:
                module = the_method.__module__

            if inspect.iscoroutinefunction(the_method):

                async def instrumented():
                    enter_span(
                        module,
                        method,
                        str(ctx.mailbox.actor_id),
                    )
                    try:
                        result = await the_method(*args, **kwargs)
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
                enter_span(module, method, str(ctx.mailbox.actor_id))
                with fake_sync_state():
                    result = the_method(*args, **kwargs)
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

    def _new_with_shape(self, shape: Shape) -> "ActorMeshRef":
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )


class ActorMeshRef(MeshTrait):
    def __init__(
        self, Class: Type[T], actor_mesh_ref: _ActorMeshRefImpl, mailbox: Mailbox
    ) -> None:
        self.__name__: str = Class.__name__
        self._class: Type[T] = Class
        self._actor_mesh_ref: _ActorMeshRefImpl = actor_mesh_ref
        self._mailbox: Mailbox = mailbox
        for attr_name in dir(self._class):
            attr_value = getattr(self._class, attr_name, None)
            if isinstance(attr_value, EndpointProperty):
                # Convert string method name to appropriate MethodSpecifier
                kind = (
                    MethodSpecifier.ExplicitPort
                    if attr_value._explicit_response_port
                    else MethodSpecifier.ReturnsResponse
                )
                setattr(
                    self,
                    attr_name,
                    ActorEndpoint(
                        self._actor_mesh_ref,
                        kind(attr_name),
                        attr_value._method,
                        self._mailbox,
                        attr_value._propagator,
                        attr_value._explicit_response_port,
                    ),
                )

    def __getattr__(self, attr: str) -> NotAnEndpoint:
        if attr in dir(self._class):
            return NotAnEndpoint(self, attr)
        raise AttributeError(attr)

    def _create(
        self,
        args: Iterable[Any],
        kwargs: Dict[str, Any],
    ) -> None:
        async def null_func(*_args: Iterable[Any], **_kwargs: Dict[str, Any]) -> None:
            return None

        ep = ActorEndpoint(
            self._actor_mesh_ref,
            MethodSpecifier.Init(),
            null_func,
            self._mailbox,
            None,
            False,
        )
        send(ep, (self._class, *args), kwargs)

    def __reduce_ex__(
        self, protocol: ...
    ) -> "Tuple[Type[ActorMeshRef], Tuple[Any, ...]]":
        return ActorMeshRef, (
            self._class,
            self._actor_mesh_ref,
            self._mailbox,
        )

    @property
    def _ndslice(self) -> NDSlice:
        return self._actor_mesh_ref._shape.ndslice

    @property
    def _labels(self) -> Iterable[str]:
        return self._actor_mesh_ref._shape.labels

    def _new_with_shape(self, shape: Shape) -> "ActorMeshRef":
        return ActorMeshRef(
            self._class,
            _ActorMeshRefImpl.from_actor_ref_with_shape(self._actor_mesh_ref, shape),
            self._mailbox,
        )

    def __repr__(self) -> str:
        return f"ActorMeshRef(class={self._class}, shape={self._actor_mesh_ref._shape})"

    async def stop(self):
        await self._actor_mesh_ref.stop()


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
