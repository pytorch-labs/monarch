# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

# Import this to initialize actor runtime before anything else.
import monarch._src.actor._extension  # noqa: F401

# Import before monarch to pre-load torch DSOs as, in exploded wheel flows,
# our RPATHs won't correctly find them.
import torch  # noqa: F401

# submodules of monarch should not be imported in this
# top-level file because it will cause them to get
# loaded even if they are not actually being used.
# for instance if we import monarch.common.functions,
# we might not want to also import monarch.common.tensor,
# which recursively imports torch.

# Instead to expose functionality as part of the
# monarch.* API, import it inside the TYPE_CHECKING
# guard (so typechecker works), and then add it
# to the _public_api dict and __all__ list. These
# entries will get loaded on demand.


if TYPE_CHECKING:
    from monarch import timer
    from monarch._src.actor.allocator import LocalAllocator, ProcessAllocator
    from monarch._src.actor.shape import NDSlice, Shape
    from monarch._src.tensor_engine.common._coalescing import coalescing

    from monarch._src.tensor_engine.common.device_mesh import (
        DeviceMesh,
        get_active_mesh,
        no_mesh,
        RemoteProcessGroup,
        slice_mesh,
        to_mesh,
    )

    from monarch._src.tensor_engine.common.function import (
        resolvers as function_resolvers,
    )

    from monarch._src.tensor_engine.common.future import Future

    from monarch._src.tensor_engine.common.invocation import RemoteException
    from monarch._src.tensor_engine.common.opaque_ref import OpaqueRef
    from monarch._src.tensor_engine.common.pipe import (
        create_pipe,
        Pipe,
        remote_generator,
    )
    from monarch._src.tensor_engine.common.remote import remote
    from monarch._src.tensor_engine.common.selection import Selection
    from monarch._src.tensor_engine.common.stream import get_active_stream, Stream
    from monarch._src.tensor_engine.common.tensor import reduce, reduce_, Tensor
    from monarch.fetch import fetch_shard, inspect, show
    from monarch.gradient_generator import grad_function, grad_generator
    from monarch.notebook import mast_mesh, reserve_torchx as mast_reserve
    from monarch.python_local_mesh import python_local_mesh
    from monarch.rust_backend_mesh import (
        rust_backend_mesh,
        rust_backend_meshes,
        rust_mast_mesh,
    )
    from monarch.rust_local_mesh import local_mesh, local_meshes, SocketType
    from monarch.simulator.config import set_meta  # noqa
    from monarch.simulator.interface import Simulator
    from monarch.world_mesh import world_mesh


_public_api = {
    "coalescing": ("monarch._src.tensor_engine.common._coalescing", "coalescing"),
    "remote": ("monarch._src.tensor_engine.common.remote", "remote"),
    "DeviceMesh": ("monarch._src.tensor_engine.common.device_mesh", "DeviceMesh"),
    "get_active_mesh": (
        "monarch._src.tensor_engine.common.device_mesh",
        "get_active_mesh",
    ),
    "no_mesh": ("monarch._src.tensor_engine.common.device_mesh", "no_mesh"),
    "RemoteProcessGroup": (
        "monarch._src.tensor_engine.common.device_mesh",
        "RemoteProcessGroup",
    ),
    "function_resolvers": ("monarch._src.tensor_engine.common.function", "resolvers"),
    "Future": ("monarch._src.tensor_engine.common.future", "Future"),
    "RemoteException": (
        "monarch._src.tensor_engine.common.invocation",
        "RemoteException",
    ),
    "Shape": ("monarch._src.actor.shape", "Shape"),
    "NDSlice": ("monarch._src.actor.shape", "NDSlice"),
    "Selection": ("monarch._src.tensor_engine.common.selection", "Selection"),
    "OpaqueRef": ("monarch._src.tensor_engine.common.opaque_ref", "OpaqueRef"),
    "create_pipe": ("monarch._src.tensor_engine.common.pipe", "create_pipe"),
    "Pipe": ("monarch._src.tensor_engine.common.pipe", "Pipe"),
    "remote_generator": ("monarch._src.tensor_engine.common.pipe", "remote_generator"),
    "get_active_stream": (
        "monarch._src.tensor_engine.common.stream",
        "get_active_stream",
    ),
    "Stream": ("monarch._src.tensor_engine.common.stream", "Stream"),
    "Tensor": ("monarch._src.tensor_engine.common.tensor", "Tensor"),
    "reduce": ("monarch._src.tensor_engine.common.tensor", "reduce"),
    "reduce_": ("monarch._src.tensor_engine.common.tensor", "reduce_"),
    "to_mesh": ("monarch._src.tensor_engine.common.device_mesh", "to_mesh"),
    "slice_mesh": ("monarch._src.tensor_engine.common.device_mesh", "slice_mesh"),
    "call_on_shard_and_fetch": ("monarch.fetch", "call_on_shard_and_fetch"),
    "fetch_shard": ("monarch.fetch", "fetch_shard"),
    "inspect": ("monarch.fetch", "inspect"),
    "show": ("monarch.fetch", "show"),
    "grad_function": ("monarch.gradient_generator", "grad_function"),
    "grad_generator": ("monarch.gradient_generator", "grad_generator"),
    "python_local_mesh": ("monarch.python_local_mesh", "python_local_mesh"),
    "mast_mesh": ("monarch.notebook", "mast_mesh"),
    "mast_reserve": ("monarch.notebook", "reserve_torchx"),
    "rust_backend_mesh": ("monarch.rust_backend_mesh", "rust_backend_mesh"),
    "rust_backend_meshes": ("monarch.rust_backend_mesh", "rust_backend_meshes"),
    "local_mesh": ("monarch.rust_local_mesh", "local_mesh"),
    "local_meshes": ("monarch.rust_local_mesh", "local_meshes"),
    "SocketType": ("monarch.rust_local_mesh", "SocketType"),
    "rust_mast_mesh": ("monarch.rust_backend_mesh", "rust_mast_mesh"),
    "set_meta": ("monarch.simulator.config", "set_meta"),
    "Simulator": ("monarch.simulator.interface", "Simulator"),
    "world_mesh": ("monarch.world_mesh", "world_mesh"),
    "timer": ("monarch.timer", "timer"),
    "ProcessAllocator": ("monarch._src.actor.allocator", "ProcessAllocator"),
    "LocalAllocator": ("monarch._src.actor.allocator", "LocalAllocator"),
    "builtins": ("monarch.builtins", "builtins"),
}


def __getattr__(name):
    if name in _public_api:
        module_path, attr_name = _public_api[name]
        module = _import_module(module_path)
        result = getattr(module, attr_name)
        globals()[name] = result
        return result
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


try:
    from __manifest__ import fbmake  # noqa

    IN_PAR = True
except ImportError:
    IN_PAR = False

# we have to explicitly list this rather than just take the keys of the _public_api
# otherwise tools think the imports are unused
__all__ = [
    "coalescing",
    "DeviceMesh",
    "get_active_mesh",
    "no_mesh",
    "remote",
    "RemoteProcessGroup",
    "function_resolvers",
    "Future",
    "RemoteException",
    "Shape",
    "Selection",
    "NDSlice",
    "OpaqueRef",
    "create_pipe",
    "Pipe",
    "remote_generator",
    "get_active_stream",
    "Stream",
    "Tensor",
    "reduce",
    "reduce_",
    "to_mesh",
    "slice_mesh",
    "call_on_shard_and_fetch",
    "fetch_shard",
    "inspect",
    "show",
    "grad_function",
    "grad_generator",
    "python_local_mesh",
    "mast_mesh",
    "mast_reserve",
    "rust_backend_mesh",
    "rust_backend_meshes",
    "local_mesh",
    "local_meshes",
    "SocketType",
    "rust_mast_mesh",
    "set_meta",
    "Simulator",
    "world_mesh",
    "timer",
    "ProcessAllocator",
    "LocalAllocator",
    "builtins",
]
assert sorted(__all__) == sorted(_public_api)
