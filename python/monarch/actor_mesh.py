# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
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
    Callable,
    cast,
    Concatenate,
    Coroutine,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    ParamSpec,
    Tuple,
    Type,
    TypeVar,
)

import monarch
from monarch import ActorFuture as Future
from monarch._rust_bindings.hyperactor_extension.telemetry import enter_span, exit_span

from monarch._rust_bindings.monarch_hyperactor.actor import PanicFlag, PythonMessage
from monarch._rust_bindings.monarch_hyperactor.actor_mesh import PythonActorMesh
from monarch._rust_bindings.monarch_hyperactor.mailbox import (
    Mailbox,
    OncePortReceiver,
    PortId,
    PortReceiver as HyPortReceiver,
)
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.monarch_hyperactor.shape import Point as HyPoint, Shape

from monarch.common.pickle_flatten import flatten, unflatten
from monarch.common.shape import MeshTrait, NDSlice

logger = logging.getLogger(__name__)

Allocator = monarch.ProcessAllocator | monarch.LocalAllocator

try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
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


# this was implemented in python 3.12 as an argument to task
# but I have to backport to 3.10/3.11.
def create_eager_task(coro: Coroutine[Any, None, Any]) -> asyncio.Future:
    iter = coro.__await__()
    try:
        first_yield = next(iter)
        return asyncio.create_task(RestOfCoroutine(first_yield, iter).run())
    except StopIteration as e:
        t = asyncio.Future()
        t.set_result(e.value)
        return t


class RestOfCoroutine(Generic[T1, T2]):
    def __init__(self, first_yield: T1, iter: Generator[T2, None, T2]) -> None:
        self.first_yield: T1 | None = first_yield
        self.iter: Generator[T2, None, T2] = iter

    def __await__(self) -> Generator[T1, None, T1] | Generator[T2, None, T2]:
        first_yield = self.first_yield
        assert first_yield is not None
        yield first_yield
        self.first_yield = None
        while True:
            try:
                yield next(self.iter)
            except StopIteration as e:
                return e.value

    async def run(self) -> T1 | T2:
        return await self


T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
A = TypeVar("A")

# keep this load balancing deterministic, but
# equally distributed.
_load_balancing_seed = random.Random(4)


Selection = Literal["all", "choose"]  # TODO: replace with real selection objects


# standin class for whatever is the serializable python object we use
# to name an actor mesh. Hacked up today because ActorMesh
# isn't plumbed to non-clients
class _ActorMeshRefImpl:
    def __init__(
        self,
        mailbox: Mailbox,
        hy_actor_mesh: Optional[PythonActorMesh],
        shape: Shape,
        actor_ids: List[ActorId],
    ) -> None:
        self._mailbox = mailbox
        self._actor_mesh = hy_actor_mesh
        self._shape = shape
        self._please_replace_me_actor_ids = actor_ids

    @staticmethod
    def from_hyperactor_mesh(
        mailbox: Mailbox, hy_actor_mesh: PythonActorMesh
    ) -> "_ActorMeshRefImpl":
        shape: Shape = hy_actor_mesh.shape
        return _ActorMeshRefImpl(
            mailbox,
            hy_actor_mesh,
            hy_actor_mesh.shape,
            [cast(ActorId, hy_actor_mesh.get(i)) for i in range(len(shape))],
        )

    @staticmethod
    def from_actor_id(mailbox: Mailbox, actor_id: ActorId) -> "_ActorMeshRefImpl":
        return _ActorMeshRefImpl(mailbox, None, singleton_shape, [actor_id])

    @staticmethod
    def from_actor_ref_with_shape(
        ref: "_ActorMeshRefImpl", shape: Shape
    ) -> "_ActorMeshRefImpl":
        return _ActorMeshRefImpl(
            ref._mailbox, None, shape, ref._please_replace_me_actor_ids
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

    def send(self, rank: int, message: PythonMessage) -> None:
        actor = self._please_replace_me_actor_ids[rank]
        self._mailbox.post(actor, message)

    def cast(
        self,
        message: PythonMessage,
        selection: Selection,
    ) -> None:
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
            return
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
        else:
            raise ValueError(f"invalid selection: {selection}")

    def __len__(self) -> int:
        return len(self._shape)


class Endpoint(Generic[P, R]):
    def __init__(
        self,
        actor_mesh_ref: _ActorMeshRefImpl,
        name: str,
        impl: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
        mailbox: Mailbox,
    ) -> None:
        self._actor_mesh = actor_mesh_ref
        self._name = name
        self._signature: inspect.Signature = inspect.signature(impl)
        self._mailbox = mailbox

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
        send(self, args, kwargs, port=p, selection="choose")
        return r.recv()

    def call_one(self, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        if len(self._actor_mesh) != 1:
            raise ValueError(
                f"Can only use 'call_one' on a single Actor but this actor has shape {self._actor_mesh._shape}"
            )
        return self.choose(*args, **kwargs)

    def call(self, *args: P.args, **kwargs: P.kwargs) -> "Future[ValueMesh[R]]":
        p, r = port(self)
        # pyre-ignore
        send(self, args, kwargs, port=p, rank_in_response=True)

        async def process():
            results = [None] * len(self._actor_mesh)
            for _ in range(len(self._actor_mesh)):
                rank, value = await r.recv()
                results[rank] = value
            call_shape = Shape(
                self._actor_mesh._shape.labels,
                NDSlice.new_row_major(self._actor_mesh._shape.ndslice.sizes),
            )
            return ValueMesh(call_shape, results)

        return Future(process)

    async def stream(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[R, R]:
        """
        Broadcasts to all actors and yields their responses as a stream / generator.

        This enables processing results from multiple actors incrementally as
        they become available. Returns an async generator of response values.
        """
        p, r = port(self)
        # pyre-ignore
        send(self, args, kwargs, port=p)
        for _ in range(len(self._actor_mesh)):
            yield await r.recv()

    def broadcast(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """
        Broadcast to all actors and wait for each to acknowledge receipt.

        This behaves like `cast`, but ensures that each actor has received and
        processed the message by awaiting a response from each one. Does not
        return any results.
        """
        # pyre-ignore
        send(self, args, kwargs)


class Accumulator(Generic[P, R, A]):
    def __init__(
        self, endpoint: Endpoint[P, R], identity: A, combine: Callable[[A, R], A]
    ):
        self._endpoint = endpoint
        self._identity = identity
        self._combine = combine

    def accumulate(self, *args: P.args, **kwargs: P.kwargs) -> "Future[A]":
        gen = self._endpoint.stream(*args, **kwargs)

        async def impl():
            value = self._identity
            async for x in gen:
                value = self._combine(value, x)
            return value

        return Future(impl)


class ValueMesh(MeshTrait, Generic[R]):
    def __init__(self, shape: Shape, values: List[R]) -> None:
        self._shape = shape
        self._values = values

    def _new_with_shape(self, shape: Shape) -> "ValueMesh[R]":
        return ValueMesh(shape, self._values)

    def item(self, **kwargs):
        coordinates = [kwargs.pop(label) for label in self._labels]
        if kwargs:
            raise KeyError(f"item has extra dimensions: {list(kwargs.keys())}")

        return self._values[self._ndslice.nditem(coordinates)]

    def __iter__(self):
        for rank in self._shape.ranks():
            yield Point(rank, self._shape), self._values[rank]

    def __len__(self):
        return len(self._shape)

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
    port: "Optional[PortId]" = None,
    selection: Selection = "all",
    rank_in_response: bool = False,
) -> None:
    """
    Fire-and-forget broadcast invocation of the endpoint across all actors in the mesh.

    This sends the message to all actors but does not wait for any result.
    """
    endpoint._signature.bind(None, *args, **kwargs)
    message = PythonMessage(
        endpoint._name, _pickle((args, kwargs)), port, rank_in_response
    )
    endpoint._actor_mesh.cast(message, selection)


class EndpointProperty(Generic[P, R]):
    def __init__(self, method: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]):
        self._method = method

    def __get__(self, instance, owner) -> Endpoint[P, R]:
        # this is a total lie, but we have to actually
        # recognize this was defined as an endpoint,
        # and also lookup the method
        return cast(Endpoint[P, R], self)


def endpoint(
    method: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> EndpointProperty[P, R]:
    return EndpointProperty(method)


class Port:
    def __init__(self, port: PortId, mailbox: Mailbox, rank_in_response: bool) -> None:
        self._port = port
        self._mailbox = mailbox
        self._rank_in_response = rank_in_response

    def send(self, method: str, obj: object) -> None:
        if self._rank_in_response:
            obj = (MonarchContext.get().point.rank, obj)
        self._mailbox.post(
            self._port,
            PythonMessage(method, _pickle(obj), None),
        )


# advance lower-level API for sending messages. This is intentially
# not part of the Endpoint API because they way it accepts arguments
# and handles concerns is different.
def port(endpoint: Endpoint[P, R], once=False) -> Tuple["PortId", "PortReceiver[R]"]:
    handle, receiver = (
        endpoint._mailbox.open_once_port() if once else endpoint._mailbox.open_port()
    )
    port_id: PortId = handle.bind()
    return port_id, PortReceiver(endpoint._mailbox, receiver)


class PortReceiver(Generic[R]):
    def __init__(
        self,
        mailbox: Mailbox,
        receiver: HyPortReceiver | OncePortReceiver,
    ):
        self._mailbox = mailbox
        self._receiver = receiver

    async def _recv(self) -> R:
        return self._process(await self._receiver.recv())

    def _blocking_recv(self) -> R:
        return self._process(self._receiver.blocking_recv())

    def _process(self, msg: PythonMessage):
        # TODO: Try to do something more structured than a cast here
        payload = cast(R, _unpickle(msg.message, self._mailbox))
        if msg.method == "result":
            return payload
        else:
            assert msg.method == "exception"
            if isinstance(payload, tuple):
                # If the payload is a tuple, it's because we requested the rank
                # to be included in the response; just ignore it.
                raise payload[1]
            else:
                # pyre-ignore
                raise payload

    def recv(self) -> "Future[R]":
        return Future(lambda: self._recv(), self._blocking_recv)


singleton_shape = Shape([], NDSlice(offset=0, sizes=[], strides=[]))


class _Actor:
    def __init__(self) -> None:
        self.instance: object | None = None
        self.active_requests: asyncio.Queue[asyncio.Future[object]] = asyncio.Queue()
        self.complete_task: asyncio.Task | None = None

    def handle(
        self, mailbox: Mailbox, message: PythonMessage, panic_flag: PanicFlag
    ) -> Optional[Coroutine[Any, Any, Any]]:
        return self.handle_cast(mailbox, 0, singleton_shape, message, panic_flag)

    def handle_cast(
        self,
        mailbox: Mailbox,
        rank: int,
        shape: Shape,
        message: PythonMessage,
        panic_flag: PanicFlag,
    ) -> Optional[Coroutine[Any, Any, Any]]:
        port = (
            Port(message.response_port, mailbox, message.rank_in_response)
            if message.response_port
            else None
        )
        try:
            ctx = MonarchContext(mailbox, mailbox.actor_id.proc_id, Point(rank, shape))
            _context.set(ctx)

            args, kwargs = _unpickle(message.message, mailbox)
            if message.method == "__init__":
                Class, *args = args
                self.instance = Class(*args, **kwargs)
                return None
            else:
                the_method = getattr(self.instance, message.method)._method

                if not inspect.iscoroutinefunction(the_method):
                    enter_span(
                        the_method.__module__, message.method, str(ctx.mailbox.actor_id)
                    )
                    result = the_method(self.instance, *args, **kwargs)
                    exit_span()
                    if port is not None:
                        port.send("result", result)
                    return None

                async def instrumented():
                    enter_span(
                        the_method.__module__, message.method, str(ctx.mailbox.actor_id)
                    )
                    result = await the_method(self.instance, *args, **kwargs)
                    exit_span()
                    return result

                return self.run_async(
                    ctx,
                    self.run_task(port, instrumented(), panic_flag),
                )
        except Exception as e:
            traceback.print_exc()
            s = ActorError(e)

            # The exception is delivered to exactly one of:
            # (1) our caller, (2) our supervisor
            if port is not None:
                port.send("exception", s)
            else:
                raise s from None

    async def run_async(
        self,
        ctx: MonarchContext,
        coroutine: Coroutine[Any, None, Any],
    ) -> None:
        _context.set(ctx)
        if self.complete_task is None:
            self.complete_task = asyncio.create_task(self._complete())
        await self.active_requests.put(create_eager_task(coroutine))

    async def run_task(self, port, coroutine, panic_flag):
        try:
            result = await coroutine
            if port is not None:
                port.send("result", result)
        except Exception as e:
            traceback.print_exc()
            s = ActorError(e)

            # The exception is delivered to exactly one of:
            # (1) our caller, (2) our supervisor
            if port is not None:
                port.send("exception", s)
            else:
                raise s from None
        except BaseException as e:
            # A BaseException can be thrown in the case of a Rust panic.
            # In this case, we need a way to signal the panic to the Rust side.
            # See [Panics in async endpoints]
            try:
                panic_flag.signal_panic(e)
            except Exception:
                # The channel might be closed if the Rust side has already detected the error
                pass
            raise

    async def _complete(self) -> None:
        while True:
            task = await self.active_requests.get()
            await task


def _is_mailbox(x: object) -> bool:
    return isinstance(x, Mailbox)


def _pickle(obj: object) -> bytes:
    _, msg = flatten(obj, _is_mailbox)
    return msg


def _unpickle(data: bytes, mailbox: Mailbox) -> Any:
    # regardless of the mailboxes of the remote objects
    # they all become the local mailbox.
    return unflatten(data, itertools.repeat(mailbox))


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
        self.__name__ = Class.__name__
        self._class = Class
        self._actor_mesh_ref = actor_mesh_ref
        self._mailbox = mailbox
        for attr_name in dir(self._class):
            attr_value = getattr(self._class, attr_name, None)
            if isinstance(attr_value, EndpointProperty):
                setattr(
                    self,
                    attr_name,
                    Endpoint(
                        self._actor_mesh_ref,
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
                endpoint = Endpoint(
                    self._actor_mesh_ref,
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

    def _create(self, args: Iterable[Any], kwargs: Dict[str, Any]) -> None:
        async def null_func(*_args: Iterable[Any], **_kwargs: Dict[str, Any]) -> None:
            return None

        ep = Endpoint(
            self._actor_mesh_ref,
            "__init__",
            null_func,
            self._mailbox,
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


class ActorError(Exception):
    """
    Deterministic problem with the user's code.
    For example, an OOM resulting in trying to allocate too much GPU memory, or violating
    some invariant enforced by the various APIs.
    """

    def __init__(
        self,
        exception: Exception,
        message: str = "A remote actor call has failed asynchronously.",
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
