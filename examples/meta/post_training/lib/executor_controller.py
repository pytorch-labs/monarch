import gc
import logging
from typing import Any, Dict, List, Optional

from monarch import fetch_shard

from .communication.communication_channel import CommunicationChannel

# from .communication.constants import RECV_OPS, SEND_OPS
from .executor import Executor

from .executor_context import ExecutorContext

logger: logging.Logger = logging.getLogger(__name__)


class ExecutorController:
    """
    The ExecutorController class is responsible for managing the execution of tasks
    across multiple executors. It initializes communication channels, manages the
    execution steps, and handles profiling and memory snapshot settings.

    Attributes:
        executor_context (ExecutorContext): The context containing executors and their configurations.
        checkpoint_executor (Executor): The executor responsible for handling checkpoints.
        communication_channels (List[CommunicationChannel]): Channels for communication between executors.
        max_steps (int): Maximum number of execution steps.
        nsteps_per_checkpoint (int): Number of steps between checkpoints.
        profile_freq (int): Frequency of profiling, <= 0 means no profiling.
        mem_snapshot_start_step (int): Step to start memory snapshot profiling, <= 0 means no profiling.
        mem_snapshot_profiling_duration (int): Duration of memory snapshot profiling.
        mem_snapshot_max_entries (int): Maximum entries for memory snapshot profiling.
        init_communication_channels (Optional[List[CommunicationChannel]]): Initial communication channels.
        align_gc (bool): Flag to align garbage collection with execution steps.
        global_step (int): Current global execution step.

    Note:
        This implementation focuses on the major execution and communication logic.
        Features like checkpointing, profiling, and memory snapshot will be added in the future.
    """

    def __init__(
        self,
        executor_context: ExecutorContext,
        checkpoint_executor: Executor,
        communication_channels: List[CommunicationChannel],
        max_steps: int,
        nsteps_per_checkpoint: int,
        profile_freq: int = -1,  # <= 0 means no profiling
        mem_snapshot_start_step: int = -1,  # <= 0 means no profiling
        mem_snapshot_profiling_duration: int = 0,
        mem_snapshot_max_entries: int = 0,
        init_communication_channels: Optional[List[CommunicationChannel]] = None,
        align_gc: bool = True,
    ) -> None:
        self.executor_context = executor_context
        self.max_steps = max_steps
        self.nsteps_per_checkpoint = nsteps_per_checkpoint
        self.model_state_dict_key_to_shape: Optional[Dict[str, Any]] = None
        self.model_state_dict_key_to_tensor: Optional[Dict[str, Any]] = None
        self.keys_not_supported: List[str] = []
        self.communication_channels: List[CommunicationChannel] = communication_channels
        self.init_communication_channels: List[CommunicationChannel] = (
            init_communication_channels or []
        )
        self.checkpoint_executor = checkpoint_executor
        self.profile_freq: int = profile_freq
        self.mem_snapshot_start_step: int = mem_snapshot_start_step
        self.mem_snapshot_profiling_duration: int = mem_snapshot_profiling_duration
        self.mem_snapshot_max_entries: int = mem_snapshot_max_entries
        self.align_gc: bool = align_gc
        self.global_step = 0
        self.init_input_dict = self.init()

    def init(self) -> Dict[str, Any]:
        self._maybe_disable_auto_gc()
        input_dict = {}
        for communication_channel in self.init_communication_channels:
            input_dict[communication_channel.inbound_executor_name] = (
                communication_channel.step(step=self.global_step, blocking=True)
            )
        return input_dict

    def run(self) -> None:
        logger.info("Starting ExecutorController")
        while self.global_step < self.max_steps:
            input_dict = {}
            if self.global_step > 0:
                # Communication ops are only executed after the first step.
                for communication_channel in self.communication_channels:
                    input_dict[communication_channel.inbound_executor_name] = (
                        communication_channel.step(step=self.global_step, blocking=True)
                    )
            else:
                input_dict = self.init_input_dict

            logger.debug(
                f"step:{self.global_step} after execute all channels: input_dict: {input_dict}"
            )

            # The executors run concurrently in each iteration, and we ensure all executors
            # finish before starting the next iteration.
            handles = []
            for (
                executor_name,
                executor,
            ) in self.executor_context.name_to_executor.items():
                with self.executor_context.name_to_mesh[executor_name].activate():
                    executor_input_dict = input_dict.get(executor_name, {})
                    logger.info(
                        f"step:{self.global_step} executor {executor_name} runs"
                    )
                    handle = executor.step(self.global_step, executor_input_dict)
                    handles.append(handle)
            for handle in handles:
                fetch_shard(handle).result()
            self.global_step += 1

    def log_metrics(self, step_start_time: int | float) -> None:
        pass

    def _maybe_disable_auto_gc(self) -> None:
        if self.align_gc:
            gc.disable()
