# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import gc
import os
import time
import traceback
from importlib import import_module

from typing import Dict, Tuple
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from monarch._testing import TestingContext
from nanoGPT import train
from nanoGPT.config import NanoGPTConfig


class MockPipe:
    def recv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randint(
                100,
                (12, 256),
                dtype=torch.int64,
            ),
            torch.randint(
                100,
                (12, 256),
                dtype=torch.int64,
            ),
        )


class TestNanogpt(TestCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["TORCH_SUPERVISOR_HEARTBEAT_INTERVAL"] = "0.1"
        self.exceeded_frames = False
        self.last_start_time = 0.0

        def timing_gc_callback(phase: str, info: Dict[str, int]) -> None:
            # just small per optimization
            if not self.exceeded_frames and phase == "start":
                self.last_start_time = time.time()
            if not self.exceeded_frames and phase == "stop":
                duration = time.time() - self.last_start_time
                if duration > 0.1:
                    frame_summaries = [
                        obj
                        for obj in gc.get_objects()
                        if isinstance(obj, traceback.FrameSummary)
                    ]
                    # this number is for one iteration and should be gc'd
                    if len(frame_summaries) > 200000:
                        self.exceeded_frames = True

        gc.callbacks.append(timing_gc_callback)

    def tearDown(self) -> None:
        os.environ.pop("TORCH_SUPERVISOR_HEARTBEAT_INTERVAL")
        super().tearDown()

    def test_main(self) -> None:
        config_module = import_module(
            "monarch.examples.nanoGPT.config.train_shakespeare_char_small"
        )
        args = args = [
            config_module.__file__,
            "--max_iters=2",
            "--mocked=True",
        ]
        train.main(args)
        # nothing to assert, just make sure we can run it

    @patch("monarch.common.pipe.Pipe")
    def test_train(self, mocked_pipe: MagicMock) -> None:
        # tests both that we can train in reasonable time and that we don't have memory leak
        # More of an integration test
        mocked_pipe.return_value = MockPipe()

        config_module = import_module(
            "monarch.examples.nanoGPT.config.train_shakespeare_char_small"
        )
        args = [
            config_module.__file__,
            "--n_gpus=1",
            "--max_iters=2",
            # no data_root_dir for tests, we use mocked_pipe
            "--data_root_dir=''",
        ]
        NanoGPTConfig.configure(args)

        start = time.time()
        with TestingContext() as tc:
            with tc.local_rust_device_mesh(
                NanoGPTConfig.n_hosts,
                NanoGPTConfig.n_gpus,
                activate=False,
            ) as device_mesh:
                train.run(device_mesh, args)

        # For now, simply test we can complete two iterations in 120 seconds
        self.assertLess(time.time() - start, 120)
        self.assertFalse(self.exceeded_frames)
