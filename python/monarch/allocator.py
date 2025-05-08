from typing import final

from monarch._monarch.hyperactor import LocalAllocatorBase, ProcessAllocatorBase


@final
class ProcessAllocator(ProcessAllocatorBase):
    """
    An allocator that allocates by spawning local processes.
    """

    pass


@final
class LocalAllocator(LocalAllocatorBase):
    """
    An allocator that allocates by spawning actors into the current process.
    """

    pass
