# pyre-strict
from unittest import TestCase

import pytest

from post_training.grpo import train_grpo
from post_training.grpo.config import GRPOConfig


@pytest.mark.timeout(200)
class TestGRPO(TestCase):
    def test_grpo_inter_mesh(self) -> None:
        config = GRPOConfig(multi_mesh=True)
        train_grpo.main(config)

    def test_grpo_sub_mesh(self) -> None:
        config = GRPOConfig(multi_mesh=False)
        train_grpo.main(config)
