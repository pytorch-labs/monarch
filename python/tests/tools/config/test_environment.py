# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from monarch.tools.config.environment import CondaEnvironment


class TestCondaEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="TestEnvironment_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_no_prefix_no_active_env_throws(self) -> None:
        # clears any CONDA_PREFIX env vars
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AssertionError):
                CondaEnvironment().conda_prefix

    def test_no_prefix_active_env(self) -> None:
        mock_conda_prefix = str(self.tmpdir / ".conda" / "testenv")
        with mock.patch.dict(os.environ, {"CONDA_PREFIX": mock_conda_prefix}):
            self.assertEqual(mock_conda_prefix, CondaEnvironment().conda_prefix)

    def test_conda_prefix(self) -> None:
        current_active = str(self.tmpdir / ".conda" / "foo")
        override = str(self.tmpdir / ".conda" / "bar")
        with mock.patch.dict(os.environ, {"CONDA_PREFIX": current_active}):
            self.assertEqual(override, CondaEnvironment(override).conda_prefix)

    def test_currently_active_env(self) -> None:
        mock_conda_prefix = str(self.tmpdir / ".conda" / "testenv")
        env = CondaEnvironment()

        with mock.patch.dict(os.environ, {"CONDA_PREFIX": mock_conda_prefix}):
            self.assertEqual(str(self.tmpdir / ".conda" / "testenv"), env.conda_prefix)

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AssertionError):
                env.conda_prefix
