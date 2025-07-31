# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from monarch.tools.utils import conda, MONARCH_HOME


class TestUtils(unittest.TestCase):
    # guard against MONARCH_HOME set outside the test
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_MONARCH_HOME_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            user_home = Path(tmpdir) / "sally"
            with mock.patch("pathlib.Path.home", return_value=user_home):
                monarch_home = MONARCH_HOME()
                self.assertEqual(monarch_home, user_home / ".monarch")
                self.assertTrue(monarch_home.exists())

    def test_MONARCH_HOME_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            override_monarch_home = Path(tmpdir) / "test" / ".monarch"
            with mock.patch.dict(
                os.environ, {"MONARCH_HOME": str(override_monarch_home)}
            ):
                monarch_home = MONARCH_HOME()
                conda_pack_out = MONARCH_HOME("conda-pack", "out")

                self.assertEqual(override_monarch_home, monarch_home)
                self.assertEqual(monarch_home / "conda-pack" / "out", conda_pack_out)

                self.assertTrue(monarch_home.is_dir())
                self.assertTrue(conda_pack_out.is_dir())


class TestCondaUtils(unittest.TestCase):
    def test_conda_active_env_name(self) -> None:
        with mock.patch.dict(
            os.environ, {"CONDA_PREFIX": "/home/USER/.conda/envs/bar-py3"}, clear=True
        ):
            self.assertEqual(conda.active_env_name(), "bar-py3")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(conda.active_env_name())

    def test_conda_active_env_dir(self) -> None:
        with mock.patch.dict(
            os.environ, {"CONDA_PREFIX": "/home/USER/.conda/envs/foo"}, clear=True
        ):
            self.assertEqual(conda.active_env_dir(), "/home/USER/.conda/envs/foo")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(conda.active_env_dir())
