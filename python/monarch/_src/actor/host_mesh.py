# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from math import prod

from typing import Callable, Dict, Optional, Tuple

from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec

from monarch._src.actor.actor_mesh import context
from monarch._src.actor.allocator import AllocateMixin, AllocHandle, LocalAllocator
from monarch._src.actor.proc_mesh import _get_bootstrap_args, ProcessAllocator, ProcMesh
from monarch._src.actor.shape import MeshTrait, NDSlice, Shape


def this_host() -> "HostMesh":
    """
    The current machine.

    This is just shorthand for looking it up via the context
    """
    return context().actor_instance.proc.host_mesh


def this_proc() -> "ProcMesh":
    """
    The current singleton process that this specific actor is
    running on
    """
    return context().actor_instance.proc


def create_local_host_mesh() -> "HostMesh":
    cmd, args, env = _get_bootstrap_args()
    return HostMesh(Shape.unity(), ProcessAllocator(cmd, args, env))


class HostMesh(MeshTrait):
    def __init__(self, shape: Shape, allocator: AllocateMixin):
        self._allocator = allocator
        self._shape = shape
        self._spawned = 0

    def _alloc(self, hosts: int, gpus: int) -> "AllocHandle":
        spec: AllocSpec = AllocSpec(AllocConstraints(), hosts=hosts, gpus=gpus)
        return self._allocator.allocate(spec)

    def spawn_procs(
        self,
        per_host: Optional[Dict[str, int]] = None,
        bootstrap: Optional[Callable[[], None]] = None,
    ) -> "ProcMesh":
        """
        Start new processes on this host mesh. By default this starts one proc
        on each host in the mesh. Additional procs can be started using `per_host` to
        specify the local shape, e.g.
            per_host = {'gpus': 8}
        Will create a proc mesh with an additional 'gpus' dimension.

        `bootstrap` is a function that will be run at startup on each proc and can be used to e.g.
        configure CUDA or NCCL. We guarantee that CUDA has not been initialized before boostrap is called.
        """
        if per_host is None:
            per_host = {}
        if self._spawned > 0 and len(self._ndslice) > 1:
            warnings.warn(
                "spawning multiple procs on the same host mesh is kinda fake at the moment, there is no guarentee that the two different spawns will be on shared hosts",
                stacklevel=2,
            )
        self._spawned += 1
        hosts = len(self._ndslice)
        flat_per_host = prod(per_host.values())
        alloc_handle = self._alloc(hosts, flat_per_host)

        new_extent = dict(zip(self._labels, self._ndslice.sizes))
        new_extent.update(per_host)
        return ProcMesh.from_alloc(alloc_handle.reshape(new_extent), bootstrap)

    @property
    def _ndslice(self) -> NDSlice:
        return self._shape.ndslice

    @property
    def _labels(self) -> Tuple[str, ...]:
        return tuple(self._shape.labels)

    def _new_with_shape(self, shape: Shape) -> "HostMesh":
        warnings.warn(
            "Slicing a host mesh is kinda fake at the moment, there is no guarentee that procs in the slice will end up on the corresponding hosts",
            stacklevel=2,
        )
        return HostMesh(
            Shape(self._labels, NDSlice.new_row_major(self._ndslice.sizes)),
            self._allocator,
        )


def fake_in_process_host() -> "HostMesh":
    return HostMesh(Shape.unity(), LocalAllocator())
