import json
import logging
import os
import pickle
import sqlite3
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from monarch import opaque_object
from monarch.common import opaque_ref

logger: logging.Logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkQueue:
    def __init__(self, db_path: str = "work_queue.db"):
        """Initialize the work queue with the specified SQLite database path."""
        self.db_path = db_path
        self._initialize_db()
        self.clean_completed_tasks(older_than_seconds=600)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute(
            "PRAGMA journal_mode=WAL"
        )  # Write-Ahead Logging for better concurrency
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    priority INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    worker_id TEXT,
                    result TEXT,
                    error TEXT
                )
            """
            )

            # Create indexes for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_topic ON tasks(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)"
            )

    def publish(self, topic: str, payload: Any, priority: int = 0) -> str:
        """
        Publish a task to the specified topic.

        Args:
            topic: The topic/channel to publish to
            payload: The task payload (will be JSON serialized)
            priority: Task priority (higher number = higher priority)

        Returns:
            The ID of the published task
        """
        task_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        try:
            pickle_payload = pickle.dumps(payload)
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tasks
                    (id, topic, payload, priority, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        topic,
                        pickle_payload,
                        priority,
                        TaskStatus.PENDING.value,
                        now,
                        now,
                    ),
                )
        except sqlite3.Error as e:
            logger.info(f"publish SQLite error occurred: {e}")
            # Handle or log the error as needed
            raise  # Optionally re-raise the exception
        except pickle.PickleError as e:
            logger.info(f"publish Pickle error occurred: {e}")
            # Handle or log the error as needed
            raise  # Optionally re-raise the exception
        except Exception as e:
            logger.info(f"publish An unexpected error occurred: {e}")
            # Handle or log the error as needed
            raise  # Optionally re-raise the exception
        return task_id

    def _try_get_task(
        self, topics: Union[str, List[str]], worker_id: Optional[str] = None
    ) -> Optional[Dict]:
        if isinstance(topics, str):
            topics = [topics]

        if worker_id is None:
            worker_id = str(uuid.uuid4())

        # Use a transaction to ensure atomicity
        with self._get_connection() as conn:
            # Prepare the placeholders for the SQL IN clause
            placeholders = ",".join("?" for _ in topics)

            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id FROM tasks
                WHERE status = ? AND topic IN ({placeholders})
                ORDER BY priority DESC, created_at DESC
                LIMIT 1
                """,
                (
                    TaskStatus.PENDING.value,
                    *topics,
                ),
            )
            updated_id = cursor.fetchone()[0]

            cursor.execute("SELECT * FROM tasks WHERE id = ?", (updated_id,))
            row = cursor.fetchone()

            if row is not None:
                # Convert row to dictionary
                task = dict(row)
                task["payload"] = pickle.loads(task["payload"])
                return task
            else:
                return None

    def subscribe(
        self,
        topics: Union[str, List[str]],
        worker_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Optional[Dict]:
        """
        Subscribe to one or more topics and get the next available task.

        Args:
            topics: A single topic or list of topics to subscribe to
            worker_id: An optional worker identifier

        Returns:
            A task dictionary or None if no tasks are available
        """
        # logger.info(f"Subscribe to {topics} with worker_id {worker_id}")
        task = self._try_get_task(topics, worker_id)
        if task is not None:
            return task

        start = time.time()
        while time.time() - start < timeout:
            task = self._try_get_task(topics, worker_id)
            if task is not None:
                return task
            time.sleep(0.1)

        raise TimeoutError("No tasks available within timeout period")

    def complete_task(self, task_id: str, result: Optional[Any] = None) -> bool:
        """
        Mark a task as completed with an optional result.

        Args:
            task_id: The ID of the task to complete
            result: Optional result data (will be JSON serialized)

        Returns:
            True if the task was updated, False otherwise
        """
        now = datetime.utcnow().isoformat()
        result_json = json.dumps(result) if result is not None else None

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = ?, result = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    TaskStatus.COMPLETED.value,
                    result_json,
                    now,
                    task_id,
                    TaskStatus.IN_PROGRESS.value,
                ),
            )

            return cursor.rowcount > 0

    def fail_task(self, task_id: str, error: str) -> bool:
        """
        Mark a task as failed with an error message.

        Args:
            task_id: The ID of the task to mark as failed
            error: Error message or reason for failure

        Returns:
            True if the task was updated, False otherwise
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    TaskStatus.FAILED.value,
                    error,
                    now,
                    task_id,
                    TaskStatus.IN_PROGRESS.value,
                ),
            )

            return cursor.rowcount > 0

    def requeue_task(self, task_id: str) -> bool:
        """
        Requeue a task that may have failed or timed out.

        Args:
            task_id: The ID of the task to requeue

        Returns:
            True if the task was requeued, False otherwise
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = ?, worker_id = NULL, updated_at = ?
                WHERE id = ? AND (status = ? OR status = ?)
                """,
                (
                    TaskStatus.PENDING.value,
                    now,
                    task_id,
                    TaskStatus.IN_PROGRESS.value,
                    TaskStatus.FAILED.value,
                ),
            )

            return cursor.rowcount > 0

    def requeue_stale_tasks(self, older_than_seconds: int = 300) -> int:
        """
        Requeue tasks that have been in progress for too long.

        Args:
            older_than_seconds: Requeue tasks that haven't been updated in this many seconds

        Returns:
            Number of tasks requeued
        """
        cutoff_time = datetime.utcnow().timestamp() - older_than_seconds
        cutoff_iso = datetime.fromtimestamp(cutoff_time).isoformat()
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = ?, worker_id = NULL, updated_at = ?
                WHERE status = ? AND updated_at < ?
                """,
                (
                    TaskStatus.PENDING.value,
                    now,
                    TaskStatus.IN_PROGRESS.value,
                    cutoff_iso,
                ),
            )

            return cursor.rowcount

    def get_task(self, task_id: str) -> Optional[Dict]:
        """
        Get a task by its ID.

        Args:
            task_id: The ID of the task to retrieve

        Returns:
            The task as a dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            task = dict(row)
            task["payload"] = json.loads(task["payload"])
            if task["result"] is not None:
                task["result"] = json.loads(task["result"])

            return task

    def get_tasks_by_status(self, status: TaskStatus, limit: int = 100) -> List[Dict]:
        """
        Get tasks with the specified status.

        Args:
            status: The task status to filter by
            limit: Maximum number of tasks to return

        Returns:
            A list of task dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                (status.value, limit),
            )

            tasks = []
            for row in cursor:
                task = dict(row)
                task["payload"] = pickle.loads(task["payload"])
                if task["result"] is not None:
                    task["result"] = json.loads(task["result"])
                tasks.append(task)

            return tasks

    def clean_completed_tasks(self, older_than_seconds: int = 86400) -> int:
        """
        Delete completed tasks older than the specified time.

        Args:
            older_than_seconds: Delete completed tasks older than this many seconds

        Returns:
            Number of tasks deleted
        """
        cutoff_time = datetime.utcnow().timestamp() - older_than_seconds
        cutoff_iso = datetime.fromtimestamp(cutoff_time).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM tasks
                WHERE status = ? AND updated_at < ?
                """,
                (TaskStatus.COMPLETED.value, cutoff_iso),
            )

            return cursor.rowcount


class InterMeshPipeSQLLiteImpl:
    def __init__(self, db_path=None):
        if db_path is not None:
            self.db_path = db_path
        else:
            user = os.getenv("USER")
            self.db_path = f"/dev/shm/{user}/work_queue.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        print(self.db_path)
        logger.info("monarch work queue being built on " + self.db_path)
        self.queue = WorkQueue(self.db_path)

    def put(self, topic, obj):
        if type(obj) is opaque_ref.OpaqueRef:
            logger.info(f"Sending message1: {obj.value} to {topic}")
            task_id = self.queue.publish(topic, obj.value)
        else:
            logger.info(f"Sending message2: {obj} to {topic}")
            task_id = self.queue.publish(topic, obj)
        logger.info(f"put message: done {task_id=}")
        return torch.tensor(0, dtype=torch.int64)

    def put_model(self, topic, model_ref):
        assert isinstance(model_ref, opaque_ref.OpaqueRef)
        assert issubclass(type(model_ref.value), torch.nn.Module)
        self.queue.publish(topic, model_ref.value.state_dict())

    def get(self, topic, block=True, timeout=10.0):
        logger.info(f"get: {topic}")
        task = self.queue.subscribe(topic, timeout=timeout)
        try:
            payload = task["payload"]
            logger.info(f"Received message: {payload} from {topic}")
            self.queue.complete_task(task["id"], "done")
        except Exception as e:
            print(f"Error loading tensor: {e} {task}")
            self.queue.fail_task(task["id"], str(e))
            return None
        return opaque_ref.OpaqueRef(payload), torch.tensor(1.0, dtype=torch.float)


class InterMeshPipe(opaque_object.OpaqueObject):
    @opaque_object.opaque_method
    def put(self, topic, obj):
        return torch.tensor(1.0, dtype=torch.float)

    @opaque_object.opaque_method
    def put_model(self, topic, model_ref):
        pass

    @opaque_object.opaque_method
    def get(self, topic, block=True, timeout=None):
        return opaque_ref.OpaqueRef(None), torch.tensor(1.0, dtype=torch.float)


def attach_to_inter_mesh_pipe(db_path_in=None) -> InterMeshPipe:
    return InterMeshPipe(
        "post_training.lib.communication.monarch_comms.InterMeshPipeSQLLiteImpl",
        db_path=db_path_in,
    )
