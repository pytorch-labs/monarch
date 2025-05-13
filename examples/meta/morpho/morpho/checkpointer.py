# pyre-unsafe
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from morpho.droppable import Droppable, null_droppable

CheckpointSchedule = Callable[[int, bool], bool]


StateDict = Dict[str, Any]


@dataclass
class CheckpointState:
    step: int
    model_state: StateDict
    dataloader_cursor: Any
    optimizer_state: StateDict
    report_state: StateDict


_fake_checkpoint = None


class Checkpointer:
    def __init__(self, output_directory: str, schedule: CheckpointSchedule):
        self.output_directory = output_directory
        self.schedule = schedule
        self.fake_checkpoint = None

    def load(self) -> Optional[CheckpointState]:
        return _fake_checkpoint

    # we do not fold this into save, because
    # the trainer might have to do work of its own
    # arrange the data into a checkpointable format
    def should_save(self, step: int, last: bool):
        return self.schedule(step, last)

    def save(
        self,
        state: CheckpointState,
    ) -> Droppable:
        global _fake_checkpoint
        _fake_checkpoint = state
        return null_droppable
