# pyre-unsafe

import copy
import itertools
import logging
import traceback
from collections import deque
from typing import cast, Dict, List, Optional, Sequence, Tuple

import numpy as np
from monarch.simulator.config import META_VAL
from monarch.simulator.profiling import RuntimeEstimator
from monarch.simulator.task import Borrow, EventTask, Task, WorkerTaskManager
from monarch.simulator.tensor import (
    FakeTensorTracker,
    StreamMemoryTracker,
    TensorManager,
    WorkerStorageTracker,
)
from monarch.simulator.trace import TraceEvent

logger = logging.getLogger(__name__)


class Stream:
    """Represents a worker stream."""

    def __init__(
        self,
        ident: int,
        name: str,
        fake_tensor_tracker: FakeTensorTracker,
        storage_tracker: WorkerStorageTracker,
        cpu_tensors: TensorManager,
    ) -> None:
        self.id = ident
        self.name = name
        self.task_queue = deque()
        self.last_task: Optional[Task] = None
        self.now = 0
        self.events: List[TraceEvent] = []
        self.memory = StreamMemoryTracker(storage_tracker)
        # Local tensors created on this stream. tTe value means which tasks
        # or borrows (int) are using this tensor.
        self.tensors = TensorManager(fake_tensor_tracker, self.memory)
        self.cpu_tensors = cpu_tensors
        self.fake_tensor_tracker = fake_tensor_tracker

    def add_task(self, task: Task) -> None:
        """
        Add a task to this stream. A task is always pending in the beginning and
        will be executed only if it is ready and is the first task in the stream.
        """
        task.start_time = max(self.now, task.start_time)

        for output in set(task.outputs) - set(task.inputs):
            self.tensors.add(output, (task,), task.start_time)

        # Input must be from the previous tasks on the same stream or from
        # the borrowed tensors.
        for tensor in task.inputs:
            if tensor in self.cpu_tensors:
                self.cpu_tensors.incr_ref(tensor, task)
            else:
                self.tensors.incr_ref(tensor, task)

        if self.task_queue:
            task.dependencies.append(self.task_queue[-1])
        elif self.last_task:
            task.dependencies.append(self.last_task)

        self.task_queue.append(task)

    def lend(self, borrow: Borrow) -> None:
        self.tensors.incr_ref(borrow.tensor_src_id, borrow.ident)

    def return_borrow(self, borrow: Borrow) -> None:
        self.tensors.decr_ref(borrow.tensor_src_id, borrow.ident, self.now, None)

    def borrow(self, borrow: Borrow) -> None:
        # We don't care about the timestamp as borrow should not incur any memory
        # usage change.
        self.tensors.add(borrow.tensor_dst_id, (), -1)
        self.tensors.first_use(borrow.tensor_dst_id, -1, None)

    def borrow_drop(self, borrow: Borrow) -> None:
        # We don't care about the timestamp as borrow should not incur any memory
        # usage change.
        # self.tensors.delete(borrow.tensor_dst_id, -1)
        pass

    def delete_refs(self, tensor_ids: List[int], now: int) -> None:
        tb = traceback.extract_stack()
        for tensor_id in tensor_ids:
            if tensor_id not in self.tensors:
                continue
            now = max(self.now, now)
            self.tensors.delete(tensor_id, now, tb)

    def maybe_set_ready(self) -> bool:
        if self.task_queue:
            return self.task_queue[0].maybe_set_ready()
        return False

    def maybe_execute(self) -> bool:
        """
        Check if we can execute the first task of this stream. Return True if
        the first task's state is changed from READY to EXECUTING.
        """
        if self.task_queue:
            task = self.task_queue[0]
            executing = task.maybe_execute()
            if executing:
                for output in set(task.outputs) - set(task.inputs):
                    self.tensors.first_use(output, task.start_time, task.traceback)
        return False

    def maybe_finish(self) -> Tuple[Optional[Task], Optional[Task]]:
        """
        Check if we can finish the first task of this stream. Return the task if
        the first task's state is changed from EXECUTING to EXECUTED else return
        None.
        """
        if not self.task_queue:
            return (None, None)

        task = self.task_queue[0]
        if not task.maybe_finish():
            return (None, None)

        task = self.task_queue.popleft()
        original_last_task = self.last_task
        self.last_task = task

        # Update the tensor and memory usage.
        if isinstance(task, EventTask):
            borrow = task.borrow
            if borrow is not None and borrow.tensor_src_id in self.tensors:
                self.tensors.decr_ref(
                    borrow.tensor_src_id, borrow.ident, task.end_time, task.traceback
                )
        else:
            removed_tensors = set()
            for tensor in itertools.chain(task.inputs, task.outputs):
                if tensor in self.cpu_tensors:
                    self.cpu_tensors.decr_ref(
                        tensor, task, task.end_time, task.traceback
                    )
                    removed_tensors.add(tensor)
                elif tensor not in self.tensors:
                    raise RuntimeError(f"tensor {tensor} not in self.tensors.")
                elif tensor not in removed_tensors:
                    # We also remove the reference even if the tensor is in
                    # outputs -- the tensor is not going to be deleted until
                    # DeleteRef is received.
                    self.tensors.decr_ref(tensor, task, task.end_time, task.traceback)
                    removed_tensors.add(tensor)

        # Add TraceEvent.
        if task.end_time > task.start_time:
            runtime = task.end_time - task.start_time
            self.events.append(
                TraceEvent(
                    task.start_time, runtime, task.meta, task.command_id, task.traceback
                )
            )

        # update the stream timestamp
        self.now = task.end_time
        return (original_last_task, task)

    def wait_event(self, event: EventTask) -> None:
        self.add_task(event)

    def record_event(self) -> Task:
        if self.task_queue:
            return self.task_queue[-1]
        elif self.last_task:
            return self.last_task
        else:
            raise RuntimeError("No tasks can be recorded.")

    def clone(
        self,
        task_manager: WorkerTaskManager,
        storage_tracker: WorkerStorageTracker,
        cpu_tensors: TensorManager,
    ) -> "Stream":
        ret = Stream(
            ident=self.id,
            name=self.name,
            fake_tensor_tracker=self.fake_tensor_tracker,
            storage_tracker=storage_tracker,
            cpu_tensors=cpu_tensors,
        )
        for task in self.task_queue:
            ret.task_queue.append(task_manager.tasks[task.task_id])
        if self.last_task:
            assert self.last_task.task_id is not None
            ret.last_task = task_manager.tasks[self.last_task.task_id]
        ret.now = self.now
        ret.events = copy.copy(self.events)
        ret.memory = self.memory.clone(storage_tracker)
        ret.tensors = self.tensors.clone(task_manager, ret.memory)
        return ret


class Worker:
    """Represents a worker."""

    def __init__(
        self,
        fake_tensor_tracker: FakeTensorTracker,
        runtime: RuntimeEstimator,
    ) -> None:
        self.runtime = runtime
        self.streams: Dict[int, Stream] = {}
        self.default_stream_id = 0
        self.events: List[TraceEvent] = []
        self.wait_events: Dict[int, EventTask] = {}
        self.fake_tensor_tracker = fake_tensor_tracker
        self.storage_tracker = WorkerStorageTracker(fake_tensor_tracker)
        # We don't track the CPU, memory usage. So pass None as the memory
        # argument.
        self.cpu_tensors = TensorManager(fake_tensor_tracker, None)
        self.borrows: Dict[int, Borrow] = {}

        self.task_manager = WorkerTaskManager()

    def record_command(
        self,
        command: str,
        command_id: int,
        now: int,
        traceback: Sequence[traceback.FrameSummary],
    ) -> None:
        # This is a CPU activity event.
        self.events.append(
            TraceEvent(
                now,
                self.runtime.get_runtime("kernel_launch"),
                [command] + META_VAL,
                command_id,
                traceback,
            )
        )

    def create_stream(self, ident: int, name: str, default: bool) -> None:
        if ident in self.streams:
            raise ValueError(f"{ident} is already created.")
        self.streams[ident] = Stream(
            ident,
            name,
            self.fake_tensor_tracker,
            self.storage_tracker,
            self.cpu_tensors,
        )
        if default:
            self.default_stream_id = ident

    def add_task(self, task: Task, now: int, stream: Optional[int] = None) -> None:
        self.record_command(task.meta[0], task.command_id, now, task.traceback)
        if stream is None:
            stream = self.default_stream_id
        self.streams[stream].add_task(task)
        self.task_manager.add(task)

    def borrow(self, task: EventTask, borrow: Borrow) -> None:
        from_stream = task.event_stream
        to_stream = task.wait_stream
        self.streams[from_stream].lend(borrow)
        self.streams[to_stream].borrow(borrow)

        # Record the event from the source stream so that the destination stream
        # can wait for it when the borrowed tensor is first used.
        # TODO: can we unify the separate data structures that keep tasks?
        self.wait_events[borrow.ident] = task
        self.task_manager.add(task)
        self.borrows[borrow.ident] = borrow

    def borrow_first_use(self, borrow_id: int, now: int) -> None:
        task = self.wait_events[borrow_id]
        to_stream = task.wait_stream

        # The destination stream needs to wait for the event before it can use
        # the borrowed tensor.
        self.record_command(task.meta[0], task.command_id, now, task.traceback)
        self.streams[to_stream].wait_event(task)

    def borrow_last_use(self, task: EventTask, borrow_id: int) -> None:
        # Record the last use event from the destination stream so that the
        # source stream can wait for it when the borrow is dropped.
        self.wait_events[borrow_id] = task
        self.task_manager.add(task)

    def borrow_drop(self, borrow_id: int, now: int) -> None:
        task = self.wait_events[borrow_id]
        from_stream = task.wait_stream
        to_stream = task.event_stream

        # Wait for the last usage.
        borrow = self.borrows[borrow_id]
        self.record_command(task.meta[0], task.command_id, now, task.traceback)
        self.streams[from_stream].wait_event(task)
        self.streams[from_stream].return_borrow(borrow)
        self.streams[to_stream].borrow_drop(borrow)

    def add_cpu_tensor(self, tensor_id: int, ts: int) -> None:
        # Currently we don't simulate any CPU ops and memory, so this is the
        # API to add CPU tensors. We also don't add the dependency of the
        # creation task as it is a CPU op (e.g., dataloader).
        self.cpu_tensors.add(tensor_id, (), ts)

    def delete_refs(self, tensor_ids: List[int], ts: int) -> None:
        for tensor_id in tensor_ids:
            if tensor_id in self.cpu_tensors:
                self.cpu_tensors.delete(tensor_id, ts, None)

        for stream in self.streams.values():
            stream.delete_refs(tensor_ids, ts)

    def maybe_set_ready(self) -> bool:
        """
        Check if we can set ready for tasks on the streams of the worker. Return
        True if we execute at least one task.
        """
        return any(s.maybe_set_ready() for s in self.streams.values())

    def maybe_execute(self) -> bool:
        """
        Check if we can execute tasks on the streams of the worker. Return
        True if we execute at least one task.
        """
        return any(s.maybe_execute() for s in self.streams.values())

    def maybe_finish(self) -> bool:
        """
        Check if we can finish any task on the streams of the worker. Return
        True if we finish at least one task.
        """
        ret = False
        for stream in self.streams.values():
            last_task, task = stream.maybe_finish()
            if task:
                ret = True
                if last_task:
                    self.task_manager.remove(last_task)
        return ret


class WorkerGroup(Worker):
    def __init__(
        self,
        workers,
        fake_tensor_tracker: FakeTensorTracker,
        runtime: RuntimeEstimator,
    ) -> None:
        super().__init__(fake_tensor_tracker, runtime)
        self.workers = workers

    def clone(self, workers) -> "WorkerGroup":
        ret = WorkerGroup(workers, self.fake_tensor_tracker, self.runtime)
        ret.default_stream_id = self.default_stream_id
        ret.events = copy.copy(self.events)
        ret.borrows = copy.copy(self.borrows)
        ret.task_manager = self.task_manager.clone()
        ret.storage_tracker = self.storage_tracker.clone()
        ret.cpu_tensors = self.cpu_tensors.clone(ret.task_manager, None)
        for ident, task in self.wait_events.items():
            assert task.task_id is not None
            ret.wait_events[ident] = cast(
                EventTask, ret.task_manager.tasks[task.task_id]
            )
        for sid, stream in self.streams.items():
            ret.streams[sid] = stream.clone(
                ret.task_manager, ret.storage_tracker, ret.cpu_tensors
            )
        return ret

    def split(self, split_set) -> "WorkerGroup":
        assert len(np.setdiff1d(split_set, self.workers, assume_unique=True)) == 0
        self.workers = np.setdiff1d(self.workers, split_set, assume_unique=True)
        return self.clone(split_set)
