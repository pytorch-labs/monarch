# pyre-strict
from unittest import TestCase

import pytest

from ddp import ddp


@pytest.mark.timeout(120)
class TestDDP(TestCase):
    def test_ddp(self) -> None:
        ddp.main(running_as_unittest=True)
