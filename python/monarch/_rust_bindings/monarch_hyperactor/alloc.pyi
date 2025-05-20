# pyre-strict

from typing import Optional

from monarch._rust_bindings.hyperactor_extension import Alloc, AllocSpec

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
