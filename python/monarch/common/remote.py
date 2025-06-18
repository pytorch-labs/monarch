# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
import logging
import warnings

from logging import Logger
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    Optional,
    overload,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

import monarch.common.messages as messages

import torch

from monarch.common import _coalescing, device_mesh, messages, stream

if TYPE_CHECKING:
    from monarch.common.client import Client

from monarch.common.device_mesh import RemoteProcessGroup
from monarch.common.fake import fake_call

from monarch.common.function import (
    Propagator,
    resolvable_function,
    ResolvableFunction,
    ResolvableFunctionFromPath,
)
from monarch.common.function_caching import (
    hashable_tensor_flatten,
    tensor_placeholder,
    TensorGroup,
    TensorPlaceholder,
)
from monarch.common.future import Future
from monarch.common.messages import Dims
from monarch.common.tensor import dtensor_check, dtensor_dispatch
from monarch.common.tree import flatten, tree_map
from torch import autograd, distributed as dist
from typing_extensions import ParamSpec

logger: Logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

Propagator = Callable | Literal["mocked", "cached", "inspect"] | None


class Remote(Generic[P, R]):
    def __init__(self, impl: Any, propagator_arg: Propagator):
        self._remote_impl = impl
        self._propagator_arg = propagator_arg
        self._cache: Optional[dict] = None

    @property
    def _resolvable(self):
        return resolvable_function(self._remote_impl)

    def _propagate(self, args, kwargs, fake_args, fake_kwargs):
        if self._propagator_arg is None or self._propagator_arg == "cached":
            if self._cache is None:
                self._cache = {}
            return _cached_propagation(self._cache, self._resolvable, args, kwargs)
        elif self._propagator_arg == "inspect":
            return None
        elif self._propagator_arg == "mocked":
            raise NotImplementedError("mocked propagation")
        else:
            return fake_call(self._propagator_arg, *fake_args, **fake_kwargs)

    def _fetch_propagate(self, args, kwargs, fake_args, fake_kwargs):
        if self._propagator_arg is None:
            return  # no propgator provided, so we just assume no mutations
        return self._propagate(args, kwargs, fake_args, fake_kwargs)

    def _pipe_propagate(self, args, kwargs, fake_args, fake_kwargs):
        if not callable(self._propagator_arg):
            raise ValueError("Must specify explicit callable for pipe")
        return self._propagate(args, kwargs, fake_args, fake_kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return dtensor_dispatch(
            self._resolvable,
            self._propagate,
            args,
            kwargs,
            device_mesh._active,
            stream._active,
        )

    def call_on_shard_and_fetch(
        self, *args, shard: Dict[str, int] | None = None, **kwargs
    ) -> Future[R]:
        return _call_on_shard_and_fetch(
            self._resolvable, self._fetch_propagate, *args, shard=shard, **kwargs
        )


# This can't just be Callable because otherwise we are not
# allowed to use type arguments in the return value.
class RemoteIfy(Protocol):
    def __call__(self, function: Callable[P, R]) -> Remote[P, R]: ...


@overload
def remote(
    function: Callable[P, R], *, propagate: Propagator = None
) -> "Remote[P, R]": ...


@overload
def remote(
    function: str, *, propagate: Literal["mocked", "cached", "inspect"] | None = None
) -> "Remote": ...


@overload
def remote(function: str, *, propagate: Callable[P, R]) -> Remote[P, R]: ...


@overload
def remote(*, propagate: Propagator = None) -> RemoteIfy: ...  # type: ignore


# ignore because otherwise it claims that the actual implementation doesn't
# accept the above list of arguments


def remote(function: Any = None, *, propagate: Propagator = None) -> Any:
    if function is None:
        return functools.partial(remote, propagate=propagate)
    return Remote(function, propagate)


def _call_on_shard_and_fetch(
    rfunction: ResolvableFunction | None,
    propagator: Any,
    /,
    *args: object,
    shard: dict[str, int] | None = None,
    **kwargs: object,
) -> Future:
    """
    Call `function` at the coordinates `shard` of the current device mesh, and retrieve the result as a Future.
        function - the remote function to call
        *args/**kwargs - arguments to the function
        shard - a dictionary from mesh dimension name to coordinate of the shard
                If None, this will fetch from coordinate 0 for all dimensions (useful after all_reduce/all_gather)
    """
    ambient_mesh = device_mesh._active

    if rfunction is None:
        preprocess_message = None
        rfunction = ResolvableFunctionFromPath("ident")
    else:
        preprocess_message = rfunction
    _, dtensors, mutates, mesh = dtensor_check(
        propagator, rfunction, args, kwargs, ambient_mesh, stream._active
    )

    client: "Client" = mesh.client
    if _coalescing.is_active(client):
        raise NotImplementedError("NYI: fetching results during a coalescing block")
    return client.fetch(
        mesh,
        stream._active._to_ref(client),
        shard,
        preprocess_message,
        args,
        kwargs,
        mutates,
        dtensors,
    )


@remote
def _propagate(
    function: ResolvableFunction, args: Tuple[Any, ...], kwargs: Dict[str, Any]
):
    """
    RF preprocess function
    """
    fn = function.resolve()

    # XXX - in addition to the functional properties,
    # and info about if any of the input tensors got mutated.
    arg_tensors, _ = flatten((args, kwargs), lambda x: isinstance(x, torch.Tensor))
    input_group = TensorGroup(arg_tensors)
    result = fn(*args, **kwargs)
    result_tensors, unflatten_result = flatten(
        result, lambda x: isinstance(x, torch.Tensor)
    )

    output_group = TensorGroup(result_tensors, parent=input_group)

    the_result = unflatten_result([tensor_placeholder for _ in result_tensors])
    return (
        the_result,
        output_group.pattern,
    )


class DummyProcessGroup(dist.ProcessGroup):
    def __init__(self, dims: Dims, world_size: int):
        # pyre-ignore
        super().__init__(0, world_size)
        self.dims = dims
        self.world_size = world_size

    def allreduce(self, tensor, op=dist.ReduceOp.SUM, async_op=False):
        class DummyWork:
            def wait(self):
                return tensor

        return DummyWork()

    def _allgather_base(self, output_tensor, input_tensor, opts):
        class DummyWork:
            def wait(self):
                return output_tensor

        return DummyWork()

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts):
        class DummyWork:
            def wait(self):
                return output_tensor

        return DummyWork()

    def __getstate__(self):
        return {"dims": self.dims, "world_size": self.world_size}

    def __setstate__(self, state):
        self.__init__(state["dims"], state["world_size"])


def _mock_pgs(x):
    if isinstance(x, autograd.function.FunctionCtx):
        for attr in dir(x):
            if not attr.startswith("__") and isinstance(attr, RemoteProcessGroup):
                setattr(x, attr, DummyProcessGroup(attr.dims, attr.size()))
        return x
    if isinstance(x, RemoteProcessGroup):
        return DummyProcessGroup(x.dims, x.size())
    return x


# for testing
_miss = 0
_hit = 0


def _cached_propagation(_cache, rfunction, args, kwargs):
    tensors, shape_key = hashable_tensor_flatten(args, kwargs)
    inputs_group = TensorGroup([t._fake for t in tensors])
    requires_grads = tuple(t.requires_grad for t in tensors)
    key = (shape_key, inputs_group.pattern, requires_grads)

    global _miss, _hit
    if key not in _cache:
        _miss += 1
        args_no_pg, kwargs_no_pg = tree_map(_mock_pgs, (args, kwargs))
        result_with_placeholders, output_pattern = _propagate.call_on_shard_and_fetch(
            function=rfunction, args=args_no_pg, kwargs=kwargs_no_pg
        ).result()

        _, unflatten_result = flatten(
            result_with_placeholders, lambda x: isinstance(x, TensorPlaceholder)
        )
        _cache[key] = (unflatten_result, output_pattern)
    else:
        _hit += 1
    # return fresh fake result every time to avoid spurious aliasing
    unflatten_result, output_pattern = _cache[key]

    output_tensors = fake_call(output_pattern.empty, [inputs_group.tensors])
    return unflatten_result(output_tensors)
