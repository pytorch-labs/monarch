# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import os
import random
import tempfile
import time
from importlib import import_module
from typing import Tuple
from unittest import TestCase

from llama3 import train
from pyre_extensions import none_throws


class TestLlama3(TestCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["TORCH_SUPERVISOR_HEARTBEAT_INTERVAL"] = "0.1"

    def tearDown(self) -> None:
        os.environ.pop("TORCH_SUPERVISOR_HEARTBEAT_INTERVAL")
        super().tearDown()

    def test_train_monarch(self) -> None:
        start = time.time()
        time_per_iter, return_code = _run_llama3("python")

        # For now, simply test we can complete in 120 seconds
        # TODO add in loss validation
        self.assertLess(time.time() - start, 120)
        # TODO lower this threshold (perf bar is too low)
        self.assertLess(time_per_iter, 1)
        self.assertTrue(return_code == 0)

    def test_train_monarch_rust(self) -> None:
        start = time.time()
        time_per_iter, return_code = _run_llama3("rust_test")

        # For now, simply test we can complete in 120 seconds
        # TODO add in loss validation
        self.assertLess(time.time() - start, 120)
        # TODO lower this threshold (perf bar is too low)
        self.assertLess(time_per_iter, 1)
        self.assertTrue(return_code == 0)


def _run_llama3(mesh_type: str) -> Tuple[float, int]:
    data_dir: str = tempfile.mkdtemp()
    shakespeare_dir = os.path.join(data_dir, "shakespeare")
    os.makedirs(shakespeare_dir)

    for split in ("train", "val"):
        with open(f"{shakespeare_dir}/{split}.bin", "wb") as f:
            f.write(bytes(random.getrandbits(8) for _ in range(120000)))

    config_module = import_module("llama3.configs.llama8b")
    path = none_throws(config_module.__file__)

    return train.main(
        [
            path,
            "--n_gpus=2",
            "--tp=1",
            "--dp=2",
            "--pp=1",
            "--n_hosts=1",
            "--n_layer=2",
            # 20 iterations with 20 / 10 = 2 total fetch shards per eval
            # This will make sure the client to wait for the workers to come back with losses
            # so that we know the workers are alive doing their jobs.
            "--max_iters=20",
            "--eval_interval=10",
            "--eval_iters=1",
            # TE introduces other dependencies; exclude it to make the dependency lean
            "--use_te=False",
            # Use small batch and block sizes to avoid OOMs
            "--batch_size=2",
            "--block_size=256",
            f"--mesh_type={mesh_type}",
            "--xlformers_data=''",
            "--xlformers_tokenizer=''",
            f"--data_root_dir={data_dir}",
            "--dataset=shakespeare",
        ]
    )
