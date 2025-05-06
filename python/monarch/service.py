import asyncio
import contextvars
import inspect

import itertools
import random
import traceback
import warnings

from dataclasses import dataclass
from traceback import extract_tb, StackSummary
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
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

import monarch._monarch.hyperactor as hyperactor
from monarch.common.pickle_flatten import flatten, unflatten
from monarch.common.shape import MeshTrait, NDSlice, Shape

Allocator = hyperactor.ProcessAllocator | hyperactor.LocalAllocator

try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
except ImportError:
    IN_PAR = False

T1 = TypeVar("T1")
T2 = TypeVar("T2")


@dataclass
class MonarchContext:
    mailbox: hyperactor.Mailbox
    proc_id: str
    rank: int
    shape: Shape

    @staticmethod
    def get() -> "MonarchContext":
        return _context.get()


_context: contextvars.ContextVar[MonarchContext] = contextvars.ContextVar(
    "monarch.service._context"
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
class ActorMeshRef:
    def __init__(
        self,
        mailbox: hyperactor.Mailbox,
        hy_actor_mesh: Optional[hyperactor.PythonActorMesh],
        shape: Shape,
        actor_ids: List[hyperactor.ActorId],
    ) -> None:
        self._mailbox = mailbox
        self._actor_mesh = hy_actor_mesh
        self._shape = shape
        self._please_replace_me_actor_ids = actor_ids

    @staticmethod
    def from_hyperactor_mesh(
        mailbox: hyperactor.Mailbox, hy_actor_mesh: hyperactor.PythonActorMesh
    ) -> "ActorMeshRef":
        shape: Shape = hy_actor_mesh.shape
        return ActorMeshRef(
            mailbox,
            hy_actor_mesh,
            hy_actor_mesh.shape,
            [
                cast(hyperactor.ActorId, hy_actor_mesh.get(i))
                for i in range(len(shape.ndslice))
            ],
        )

    @staticmethod
    def from_actor_id(
        mailbox: hyperactor.Mailbox, actor_id: hyperactor.ActorId
    ) -> "ActorMeshRef":
        return ActorMeshRef(mailbox, None, singleton_shape, [actor_id])

    @staticmethod
    def from_actor_ref_with_shape(ref: "ActorMeshRef", shape: Shape) -> "ActorMeshRef":
        return ActorMeshRef(ref._mailbox, None, shape, ref._please_replace_me_actor_ids)

    def __getstate__(
        self,
    ) -> Tuple[Shape, List[hyperactor.ActorId], hyperactor.Mailbox]:
        return self._shape, self._please_replace_me_actor_ids, self._mailbox

    def __setstate__(
        self,
        state: Tuple[Shape, List[hyperactor.ActorId], hyperactor.Mailbox],
    ) -> None:
        self._actor_mesh = None
        self._shape, self._please_replace_me_actor_ids, self._mailbox = state

    def send(self, rank: int, message: hyperactor.PythonMessage) -> None:
        actor = self._please_replace_me_actor_ids[rank]
        self._mailbox.post(actor, message)

    def cast(
        self,
        message: hyperactor.PythonMessage,
        selection: Selection,
    ) -> None:
        if selection == "choose":
            idx = _load_balancing_seed.randrange(len(self._shape.ndslice))
            actor_rank = self._shape.ndslice[idx]
            self._mailbox.post(self._please_replace_me_actor_ids[actor_rank], message)
            return
        elif selection == "all":
            if self._actor_mesh is None:
                # replace me with actual remote actor mesh
                for rank in self._shape.ranks():
                    self._mailbox.post(self._please_replace_me_actor_ids[rank], message)
            else:
                self._actor_mesh.cast(message)
        else:
            raise ValueError(f"invalid selection: {selection}")

    @property
    def len(self) -> int:
        return len(self._shape.ndslice)


class Endpoint(Generic[P, R]):
    def __init__(
        self,
        actor_mesh_ref: ActorMeshRef,
        name: str,
        impl: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
        mailbox: hyperactor.Mailbox,
    ) -> None:
        self._actor_mesh = actor_mesh_ref
        self._name = name
        self._signature: inspect.Signature = inspect.signature(impl)
        self._mailbox = mailbox

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "Message[P, R]":
        self._signature.bind(None, *args, **kwargs)
        return Message(self, args, kwargs)

    def _port(self, once: bool) -> Tuple["Port", "PortReceiver[R]"]:
        handle, receiver = (
            self._mailbox.open_once_port() if once else self._mailbox.open_port()
        )
        port_id: hyperactor.PortId = handle.bind()
        return Port(port_id, self._mailbox), PortReceiver(self._mailbox, receiver)


class Message(Generic[P, R]):
    def __init__(self, endpoint: Endpoint[P, R], args: object, kwargs: object):
        self._endpoint = endpoint
        self._used = False
        self._args = args
        self._kwargs = kwargs

    def port(self, once: bool = False) -> Tuple["Port", "PortReceiver"]:
        return self._endpoint._port(once)

    def __del__(self):
        if not self._used:
            warnings.warn("This message was not used.", UserWarning, stacklevel=2)

    def broadcast(
        self, *, port: "Optional[Port]" = None, selection: Selection = "all"
    ) -> None:
        """
        Fire-and-forget broadcast invocation of the endpoint across all actors in the mesh.

        This sends the message to all actors but does not wait for any result.
        """
        if self._used:
            raise ValueError(
                "message has already been sent, create a new message to sent it a different way"
            )
        self._used = True
        message = hyperactor.PythonMessage(
            self._endpoint._name, _pickle((self._args, self._kwargs, port))
        )
        self._endpoint._actor_mesh.cast(message, selection)
        del self._args, self._kwargs

    def choose(self) -> "Future[R]":
        """
        Load balanced sends a message to one chosen actor and awaits a result.

        Load balanced RPC-style entrypoint for request/response messaging.
        """
        port, receiver = self.port(once=True)
        self.broadcast(selection="choose", port=port)
        return receiver.recv()

    def call(self) -> "Future[R]":
        if self._endpoint._actor_mesh.len != 1:
            raise ValueError(
                f"Can only use 'call' on a single Actor but this actor has shape {self._endpoint._actor_mesh._shape}"
            )
        return self.choose()

    # TODO: convert into something equivalent that works with futures
    async def stream(self) -> AsyncGenerator[R, R]:
        """
        Broadcasts to all actors and yields their responses as a stream / generator.

        This enables processing results from multiple actors incrementally as
        they become available. Returns an async generator of response values.
        """
        port, receiver = self.port()
        self.broadcast(port=port)
        for _ in range(self._endpoint._actor_mesh.len):
            yield await receiver.recv()

    def accumulate(
        self,
        identity: A,
        combine: Callable[[A, R], A],
    ) -> "Future[A]":
        gen = self.stream()

        async def impl():
            value = identity
            async for x in gen:
                value = combine(value, x)
            return value

        return Future(impl())

    def broadcast_and_wait(self) -> "Future[None]":
        """
        Broadcast to all actors and wait for each to acknowledge receipt.

        This behaves like `cast`, but ensures that each actor has received and
        processed the message by awaiting a response from each one. Does not
        return any results.
        """
        return self.accumulate(None, lambda x, _: x)


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
    def __init__(self, port: hyperactor.PortId, mailbox: hyperactor.Mailbox) -> None:
        self._port = port
        self._mailbox = mailbox

    def send(self, method: str, obj: object) -> None:
        self._mailbox.post(
            self._port,
            hyperactor.PythonMessage(method, _pickle(obj)),
        )


class PortReceiver(Generic[R]):
    def __init__(
        self,
        mailbox: hyperactor.Mailbox,
        receiver: hyperactor.PortReceiver | hyperactor.OncePortReceiver,
    ):
        self._mailbox = mailbox
        self._receiver = receiver

    async def _recv(self) -> R:
        return self._process(await self._receiver.recv())

    def _blocking_recv(self) -> R:
        return self._process(self._receiver.blocking_recv())

    def _process(self, msg: hyperactor.PythonMessage):
        # TODO: Try to do something more structured than a cast here
        payload = cast(R, _unpickle(msg.message, self._mailbox))
        if msg.method == "result":
            return payload
        else:
            assert msg.method == "exception"
            # pyre-ignore do something more structured here
            raise payload

    def recv(self) -> "Future[R]":
        return Future(self._recv(), self._blocking_recv)


class Future(Generic[R]):
    def __init__(self, impl, blocking_impl=None):
        self._impl = impl
        self._blocking_impl = blocking_impl

    def get(self) -> R:
        if self._blocking_impl is not None:
            return self._blocking_impl()
        return asyncio.run(self._impl)

    def __await__(self):
        return self._impl.__await__()


singleton_shape = Shape([], NDSlice(offset=0, sizes=[], strides=[]))


class _Actor:
    def __init__(self) -> None:
        self.instance: object | None = None
        self.active_requests: asyncio.Queue[asyncio.Future[object]] = asyncio.Queue()
        self.complete_task: object | None = None

    def handle(
        self, mailbox: hyperactor.Mailbox, message: hyperactor.PythonMessage
    ) -> Optional[Coroutine[Any, Any, Any]]:
        return self.handle_cast(mailbox, 0, singleton_shape, message)

    def handle_cast(
        self,
        mailbox: hyperactor.Mailbox,
        rank: int,
        shape: Shape,
        message: hyperactor.PythonMessage,
    ) -> Optional[Coroutine[Any, Any, Any]]:
        port = None
        try:
            args, kwargs, port = _unpickle(message.message, mailbox)

            ctx = MonarchContext(mailbox, mailbox.actor_id.proc_id, rank, shape)
            _context.set(ctx)

            if message.method == "__init__":
                Class, *args = args
                self.instance = Class(*args, **kwargs)
                return None
            else:
                the_method = getattr(self.instance, message.method)._method
                result = the_method(self.instance, *args, **kwargs)
                if not inspect.iscoroutinefunction(the_method):
                    if port is not None:
                        port.send("result", result)
                    return None

                return self.run_async(ctx, self.run_task(port, result))
        except Exception as e:
            s = ServiceCallFailedException(e)
            if port is not None:
                port.send("exception", s)
            raise s from None

    async def run_async(self, ctx, coroutine):
        _context.set(ctx)
        if self.complete_task is None:
            asyncio.create_task(self._complete())
        await self.active_requests.put(create_eager_task(coroutine))

    async def run_task(self, port, coroutine):
        try:
            result = await coroutine
            if port is not None:
                port.send("result", result)
        except Exception as e:
            print("EXCEPTING", e)
            s = ServiceCallFailedException(e)
            if port is not None:
                port.send("exception", s)
            raise s from None

    async def _complete(self) -> None:
        while True:
            task = await self.active_requests.get()
            await task


def _is_mailbox(x: object) -> bool:
    return isinstance(x, hyperactor.Mailbox)


def _pickle(obj: object) -> bytes:
    _, msg = flatten(obj, _is_mailbox)
    return msg


def _unpickle(data: bytes, mailbox: hyperactor.Mailbox) -> Any:
    # regardless of the mailboxes of the remote objects
    # they all become the local mailbox.
    return unflatten(data, itertools.repeat(mailbox))


class Actor(MeshTrait):
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

    def _new_with_shape(self, shape: Shape) -> "Service":
        raise NotImplementedError(
            "actor implementations are not meshes, but we can't convince the typechecker of it..."
        )


class Service(MeshTrait):
    def __init__(
        self, Class: Type[T], actor_mesh_ref: ActorMeshRef, mailbox: hyperactor.Mailbox
    ) -> None:
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

    def _create(self, args: Iterable[Any], kwargs: Dict[str, Any]) -> None:
        async def null_func(*_args: Iterable[Any], **_kwargs: Dict[str, Any]) -> None:
            return None

        ep = Endpoint(
            self._actor_mesh_ref,
            "__init__",
            null_func,
            self._mailbox,
        )
        # pyre-ignore
        ep(self._class, *args, **kwargs).broadcast()

    def __reduce_ex__(self, protocol: ...) -> "Tuple[Type[Service], Tuple[Any, ...]]":
        return Service, (
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

    def _new_with_shape(self, shape: Shape) -> "Service":
        return Service(
            self._class,
            ActorMeshRef.from_actor_ref_with_shape(self._actor_mesh_ref, shape),
            self._mailbox,
        )


class ServiceCallFailedException(Exception):
    """
    Deterministic problem with the user's code.
    For example, an OOM resulting in trying to allocate too much GPU memory, or violating
    some invariant enforced by the various APIs.
    """

    def __init__(
        self,
        exception: Exception,
        message: str = "A remote service call has failed asynchronously.",
    ) -> None:
        self.exception = exception
        self.service_frames: StackSummary = extract_tb(exception.__traceback__)
        self.message = message

    def __str__(self) -> str:
        exe = str(self.exception)
        service_tb = "".join(traceback.format_list(self.service_frames))
        return (
            f"{self.message}\n"
            f"Traceback of where the service call failed (most recent call last):\n{service_tb}{type(self.exception).__name__}: {exe}"
        )


def current_actor_name() -> str:
    return str(MonarchContext.get().mailbox.actor_id)


def current_rank() -> Dict[str, int]:
    ctx = MonarchContext.get()
    return ctx.shape.coordinates(ctx.rank)


def current_size() -> Dict[str, int]:
    ctx = MonarchContext.get()
    return dict(zip(ctx.shape.labels, ctx.shape.ndslice.sizes))
