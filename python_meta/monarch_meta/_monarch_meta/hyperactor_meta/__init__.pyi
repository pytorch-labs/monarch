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

@final
class MockMast:
    def __init__(self) -> None:
        """
        Create a new mock mast that can be used to emulate mast setup locally.
        """
        ...

    async def add_local_task_group(self, name: str, num_tasks: int) -> None:
        """
        Add a local task group to the mock mast using LocalAllocator.

        Arguments:
        - `name`: The name of the task group.
        - `num_tasks`: The number of tasks in the task group.
        """
        ...

    async def add_process_task_group(
        self,
        name: str,
        num_tasks: int,
        cmd: str,
        args: Optional[dict[str, str]] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Add a local task group to the mock mast using ProcessAllocator.

        Arguments:
        - `name`: The name of the task group.
        - `num_tasks`: The number of tasks in the task group.
        - `cmd`: The command to run for each task.
        - `args`: The arguments to pass to the command.
        - `env`: The environment variables to set for each task.
        """
        ...

    async def stop_task_group(self, name: str) -> None:
        """
        Stop a task group.

        Arguments:
        - `name`: The name of the task group.
        """
        ...

    async def get_mast_allocator(self, config: MastAllocatorConfig) -> MastAllocator:
        """
        Get a mast allocator for the mock mast. `config` must have job_name set.

        Arguments:
        - `config`: The configuration to use for the mast allocator. Must have job_name set.
        """
        ...
