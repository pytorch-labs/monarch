"""E2E tests for RL + Monarch examples.

To run:

buck2 run @//mode/opt fbcode//monarch/examples/tests:test_rl
"""
# pyre-strict

import logging

logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger(__name__)

import asyncio
from unittest import TestCase

import pytest
from rl import async_pipelined


@pytest.mark.timeout(200)
class TestRL(TestCase):
    def test_async_pipelined(self) -> None:
        logger.setLevel(logging.DEBUG)
        asyncio.run(async_pipelined.main(max_gpus=4))
