# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
    DEFAULT_PORT: int
    DEFAULT_HEARTBEAT_INTERVAL_MILLIS: int

    def __new__(
        cls,
        world_id: str,
        addrs: list[str],
        heartbeat_interval_millis: int = 5000,
    ) -> Self:
        """
        Create a new (client-side) allocator instance that submits allocation requests to
        remote hosts that are running hyperactor's RemoteProcessAllocator.

        Arguments:
        - `world_id`: The world id to use for the remote allocator.
        - `addrs`: The addresses of the remote process allocators.
            Each address is of the form `{transport}!{addr}(:{port})`.
            This is the string form of `hyperactor::channel::ChannelAddr` (Rust).
            For example, `tcp!127.0.0.1:1234`.  All the addresses must have the
            same transport type.
        - `heartbeat_interval_ millis`: Heartbeat interval in milliseconds.
            Used to maintain health status of remote hosts.
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
