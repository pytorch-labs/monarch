# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import final

from monarch import ActorFuture as Future
from monarch._rust_bindings.hyperactor_extension.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    Alloc,
    AllocSpec,
)

from monarch._rust_bindings.monarch_hyperactor.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    LocalAllocatorBase,
    ProcessAllocatorBase,
    RemoteAllocatorBase,
)


@final
class ProcessAllocator(ProcessAllocatorBase):
    """
    An allocator that allocates by spawning local processes.
    """

    def allocate(self, spec: AllocSpec) -> Future[Alloc]:
        """
        Allocate a process according to the provided spec.

        Arguments:
        - `spec`: The spec to allocate according to.

        Returns:
        - A future that will be fulfilled when the requested allocation is fulfilled.
        """
        return Future(
            lambda: self.allocate_nonblocking(spec),
            lambda: self.allocate_blocking(spec),
        )


@final
class LocalAllocator(LocalAllocatorBase):
    """
    An allocator that allocates by spawning actors into the current process.
    """

    def allocate(self, spec: AllocSpec) -> Future[Alloc]:
        """
        Allocate a process according to the provided spec.

        Arguments:
        - `spec`: The spec to allocate according to.

        Returns:
        - A future that will be fulfilled when the requested allocation is fulfilled.
        """
        return Future(
            lambda: self.allocate_nonblocking(spec),
            lambda: self.allocate_blocking(spec),
        )


class RemoteAllocInitializer(abc.ABC):
    """Subclass-able Python interface for `hyperactor_mesh::alloc::remoteprocess:RemoteProcessAllocInitializer`.

    NOTE: changes to method signatures of this class must be made to the call-site at
    `PyRemoteProcessAllocInitializer.py_initialize_alloc()` in `monarch/monarch_hyperactor/src/alloc.rs`
    """

    @abc.abstractmethod
    async def initialize_alloc(self) -> list[str]:
        """
        Return the addresses of the servers that should be used to allocate processes
        for the proc mesh. The addresses should be running hyperactor's RemoteProcessAllocator.

        Each address is of the form `{transport}!{addr}(:{port})`.
        This is the string form of `hyperactor::channel::ChannelAddr` (Rust).
        For example, `tcp!127.0.0.1:1234`.

        NOTE: Currently, all the addresses must have the same transport type and port
        NOTE: Although this method is currently called once at the initialization of the Allocator,
            in the future this method can be called multiple times and should return the current set of
            addresses that are eligible to handle allocation requests.

        """
        ...


class StaticRemoteAllocInitializer(RemoteAllocInitializer):
    """
    Returns the static list of server addresses that this initializer
    was constructed with on each `initialize_alloc()` call.
    """

    def __init__(self, *addrs: str) -> None:
        super().__init__()
        self.addrs: list[str] = list(addrs)

    async def initialize_alloc(self) -> list[str]:
        return list(self.addrs)


@final
class RemoteAllocator(RemoteAllocatorBase):
    """
    An allocator that allocates by spawning actors on a remote host.
    The remote host must be running hyperactor's remote-process-allocator.
    """

    def allocate(self, spec: AllocSpec) -> Future[Alloc]:
        """
        Allocate a process according to the provided spec.

        Arguments:
        - `spec`: The spec to allocate according to.

        Returns:
        - A future that will be fulfilled when the requested allocation is fulfilled.
        """
        return Future(
            lambda: self.allocate_nonblocking(spec),
            lambda: self.allocate_blocking(spec),
        )
