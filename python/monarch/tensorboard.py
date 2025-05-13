# pyre-unsafe
import logging
from typing import Any

from monarch.common.device_mesh import DeviceMesh
from monarch.remote_class import ControllerRemoteClass, WorkerRemoteClass
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Tensorboard(ControllerRemoteClass):
    def __init__(self, coordinator: DeviceMesh, path: str, *args, **kwargs) -> None:
        from monarch import IN_PAR

        self.path = path
        self.url: str = ""
        self.coordinator = coordinator
        if path.startswith("manifold://"):
            if not IN_PAR:
                raise RuntimeError(
                    "Cannot save tensorboard to manifold with conda environment. "
                    "Save to the local filesystem or oilfs instead"
                )

            manifold_url = f"https://internalfb.com/intern/tensorboard/?dir={path}"
            self.url = manifold_url
        else:
            self.url = path

        # Only create tensorboard for the coordinator rank.
        with self.coordinator.activate():
            super().__init__(
                "monarch.tensorboard._WorkerSummaryWriter",
                path,
                *args,
                **kwargs,
            )

        logger.info("Run `tensorboard --logdir %s` to launch the tensorboard.", path)

    @ControllerRemoteClass.remote_method
    def _log(self, name: str, data: Any, step: int) -> None:
        pass

    @ControllerRemoteClass.remote_method
    def _flush(self) -> None:
        pass

    @ControllerRemoteClass.remote_method
    def _close(self) -> None:
        pass

    def log(self, name: str, data: Any, step: int) -> None:
        with self.coordinator.activate():
            self._log(name, data, step)

    def flush(self) -> None:
        with self.coordinator.activate():
            self._flush()

    def close(self) -> None:
        with self.coordinator.activate():
            self._close()


class _WorkerSummaryWriter(WorkerRemoteClass):
    def __init__(self, path: str, *args, **kwargs) -> None:
        self._writer = SummaryWriter(path, *args, **kwargs)

    def _log(self, name: str, data: Any, step: int) -> None:
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def _flush(self) -> None:
        self._writer.flush()

    def _close(self) -> None:
        self._writer.close()
