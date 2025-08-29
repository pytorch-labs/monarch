# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from pathlib import Path
from typing import Any, Callable

from monarch.tools.config import (  # @manual=//monarch/python/monarch/tools/config/meta:defaults
    defaults,
    UnnamedAppDef,
)
from torchx.specs import AppDef
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

    def test_default_config_appdef(self) -> None:
        for scheduler, _ in {
            "mast": {"image": "_DUMMY_FBPKG_:0"},
            "whatever": {"image": "_DUMMY_FBPKG_:0"},
            "mast_conda": {},
        }.items():
            config = defaults.config(
                scheduler,
            )
            self.assertEqual(UnnamedAppDef(), config.appdef)

    def test_default_scheduler_factories(self) -> None:
        # just make sure the common schedulers are present
        self.assertIn("local_cwd", defaults.scheduler_factories())
        self.assertIn("slurm", defaults.scheduler_factories())

    def test_default_component(self) -> None:
        # just make sure there exists a default component for each configured scheduler
        # and that the returned default component is a valid component

        for scheduler in defaults.scheduler_factories():
            with self.subTest(scheduler=scheduler):
                component_fn: Callable[..., UnnamedAppDef] = defaults.component_fn(
                    scheduler
                )

                def _component_fn_wrapper(
                    component_fn: Callable[..., UnnamedAppDef] = component_fn,
                    *args: Any,
                    **kwargs: Any,
                ) -> AppDef:
                    name = kwargs.pop("name", None)
                    unnamed_appdef = component_fn(*args, **kwargs)
                    return AppDef(
                        name=name,
                        roles=unnamed_appdef.roles,
                        metadata=unnamed_appdef.metadata,
                    )

                # the following will fail if the component_fn is not a valid torchx component
                with self.assertRaises(SystemExit):
                    _create_args_parser(_component_fn_wrapper).parse_args(["--help"])
