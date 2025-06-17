# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from datetime import timedelta
from typing import Optional

from monarch._rust_bindings.hyperactor_extension.alloc import Alloc, AllocSpec
from typing_extensions import Self

class ProcessAllocatorBase:
    def __init__(
        self,
        program: str,
        args: Optional[list[str]] = None,
        envs: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Create a new process allocator.

        Arguments:
        - `program`: The program for each process to run. Must be a hyperactor
                    bootstrapped program.
        - `args`: The arguments to pass to the program.
        - `envs`: The environment variables to set for the program.
        """
        ...

    async def allocate_nonblocking(self, spec: AllocSpec) -> Alloc:
        """
        Allocate a process according to the provided spec.

        Arguments:
        - `spec`: The spec to allocate according to.
        """
        ...

    def allocate_blocking(self, spec: AllocSpec) -> Alloc:
        """
        Allocate a process according to the provided spec, blocking until an
        alloc is returned.

        Arguments:
        - `spec`: The spec to allocate according to.
        """
        ...

class LocalAllocatorBase:
    async def allocate_nonblocking(self, spec: AllocSpec) -> Alloc:
        """
        Allocate a process according to the provided spec.

        Arguments:
        - `spec`: The spec to allocate according to.
        """
        ...

    def allocate_blocking(self, spec: AllocSpec) -> Alloc:
        """
        Allocate a process according to the provided spec, blocking until an
        alloc is returned.

        Arguments:
        - `spec`: The spec to allocate according to.
        """
        ...

class RemoteAllocatorBase:
    def __new__(
        cls,
        world_id: str,
        initializer: "monarch.allocator.RemoteAllocInitializer",  # pyre-ignore[11] defined in monarch/python/monarch/allocator.py
        heartbeat_interval: timedelta = timedelta(seconds=5),
    ) -> Self:
        """
        Create a new (client-side) allocator instance that submits allocation requests to
        remote hosts that are running hyperactor's RemoteProcessAllocator.

        Arguments:
        - `world_id`: The world id to use for the remote allocator.
        - `initializer`: Returns the server addresses to send allocation requests to.
        - `heartbeat_interval`: Heartbeat interval used to maintain health status of remote hosts.
        """
        ...

    async def allocate_nonblocking(self, spec: AllocSpec) -> Alloc:
        """
        Allocate a process according to the provided spec.

        Arguments:
        - `spec`: The spec to allocate according to.
        """
        ...

    def allocate_blocking(self, spec: AllocSpec) -> Alloc:
        """
        Allocate a process according to the provided spec, blocking until an
        alloc is returned.

        Arguments:
        - `spec`: The spec to allocate according to.
        """
        ...
