from typing import Dict, final, List, Optional, Type

from monarch._monarch.hyperactor import Alloc, AllocSpec

@final
class MastAllocatorConfig:
    DEFAULT_REMOTE_ALLOCATOR_PORT: int

    def __init__(
        self,
        job_name: Optional[str] = None,
        transport: Optional[str] = None,
        remote_allocator_port: Optional[int] = None,
    ) -> None:
        """
        Create a new Mast allocator config.

        Arguments:
        - `job_name`: The name of the mast job to allocate on. Defaults to current job.
        - `transport`: The transport to use {tcp|metatls}. Defaults to metatls.
        - `remote_allocator_port`: The port to use for the remote allocator. Defaults to DEFAULT_REMOTE_ALLOCATOR_PORT.
        """
        ...

@final
class MastAllocator:
    ALLOC_LABEL_TASK_GROUP: str

    def __init__(
        self,
        config: Optional[MastAllocatorConfig] = None,
    ) -> None:
        """
        Create a new mast allocator.

        Arguments:
        - `config`: Configurations to use. Defaults to default configs.
        """
        ...

    async def allocate(self, spec: AllocSpec) -> Alloc:
        """
        Allocate a process according to the provided spec. MastAllocator.ALLOC_LABEL_TASK_GROUP
        must be present in the spec's labels.

        Arguments:
        - `spec`: The spec to allocate according to. MastAllocator.ALLOC_LABEL_TASK_GROUP is
                  required.
        """
        ...
