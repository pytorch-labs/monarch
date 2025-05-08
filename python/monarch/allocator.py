from typing import final

from monarch import ActorFuture as Future

from monarch._monarch.hyperactor import (
    Alloc,
    AllocSpec,
    LocalAllocatorBase,
    ProcessAllocatorBase,
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
            self.allocate_nonblocking(spec), lambda spec: self.allocate_blocking(spec)
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
            self.allocate_nonblocking(spec), lambda spec: self.allocate_blocking(spec)
        )
