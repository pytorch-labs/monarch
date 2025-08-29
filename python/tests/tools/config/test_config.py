# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import shutil
import tempfile
import unittest
from pathlib import Path

from monarch.tools.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="TestConfig_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_workspace_is_None(self) -> None:
        with self.assertWarns(FutureWarning):
            config = Config(workspace=None)  # pyre-ignore[6] BC testing
        self.assertDictEqual({}, config.workspace.dirs)
        self.assertIsNone(config.workspace.env)

    def test_workspace_is_str(self) -> None:
        with self.assertWarns(FutureWarning):
            # pyre-ignore[6] BC testing
            config = Config(workspace=str(self.tmpdir / "torch"))

        self.assertDictEqual({self.tmpdir / "torch": ""}, config.workspace.dirs)
