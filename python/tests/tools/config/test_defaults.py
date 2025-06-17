# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from pathlib import Path

from monarch.tools.config import (  # @manual=//monarch/python/monarch/tools/config/meta:defaults
    defaults,
)
from torchx.specs.builders import _create_args_parser


class TestDefaults(unittest.TestCase):
    def test_default_config(self) -> None:
        for scheduler in defaults.scheduler_factories():
            with self.subTest(scheduler=scheduler):
                config = defaults.config(scheduler)

                # make sure that we've set the scheduler name when returning the config
                self.assertEqual(scheduler, config.scheduler)

                # make sure a new Config is returned each time
                # by modifying the returned config
                #   -> re-getting the default configs for the same scheduler
                #   -> validating the changes are not persisted in the new config
                self.assertNotIn("foo", config.scheduler_args)
                config.scheduler_args["foo"] = "bar"
                self.assertNotIn("foo", defaults.config(scheduler).scheduler_args)

    def test_default_config_workspace(self) -> None:
        current_working_dir = str(Path.cwd())
        config = defaults.config("local_cwd", current_working_dir)
        self.assertEqual(current_working_dir, config.workspace)

    def test_default_scheduler_factories(self) -> None:
        # just make sure the common schedulers are present
        self.assertIn("local_cwd", defaults.scheduler_factories())
        self.assertIn("slurm", defaults.scheduler_factories())

    def test_default_component(self) -> None:
        # just make sure there exists a default component for each configured scheduler
        # and that the returned default component is a valid component
        for scheduler in defaults.scheduler_factories():
            with self.subTest(scheduler=scheduler):
                component_fn = defaults.component_fn(scheduler)

                # the following will fail if the component_fn is not a valid torchx component
                with self.assertRaises(SystemExit):
                    _create_args_parser(component_fn).parse_args(["--help"])
