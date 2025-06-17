# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest import mock

from monarch.tools import commands
from monarch.tools.commands import component_args_from_cli

from monarch.tools.config import (  # @manual=//monarch/python/monarch/tools/config/meta:defaults
    defaults,
)
from monarch.tools.mesh_spec import MeshSpec, ServerSpec
from torchx.specs import AppDef, AppDryRunInfo, AppState, AppStatus, Role


class TestCommands(unittest.TestCase):
    def test_component_args_from_cli(self) -> None:
        def fn(h: str, num_hosts: int) -> AppDef:
            return AppDef("_unused_", roles=[Role("_unused_", "_unused_")])

        args = component_args_from_cli(fn, ["h=gpu.medium", "num_hosts=4"])

        # should be able to call the component function with **args as kwargs
        self.assertIsNotNone(fn(**args))
        self.assertDictEqual({"h": "gpu.medium", "num_hosts": 4}, args)

    def test_create_dryrun(self) -> None:
        config = defaults.config("slurm")
        config.dryrun = True

        dryrun_info = commands.create(config)()
        # need only assert that the return type of dryrun is a dryrun info object
        # since we delegate to torchx for job submission
        self.assertIsInstance(dryrun_info, AppDryRunInfo)

    @mock.patch(
        "torchx.schedulers.slurm_scheduler.SlurmScheduler.schedule",
        return_value="test_job_id",
    )
    def test_create(self, mock_schedule: mock.MagicMock) -> None:
        config = defaults.config("slurm")
        server_handle = commands.create(config)()

        mock_schedule.assert_called_once()
        self.assertEqual(server_handle, "slurm:///test_job_id")

    @mock.patch("monarch.tools.commands.Runner.cancel")
    def test_kill(self, mock_cancel: mock.MagicMock) -> None:
        handle = "slurm:///test_job_id"
        commands.kill(handle)
        mock_cancel.assert_called_once_with(handle)

    @mock.patch("monarch.tools.commands.Runner.status", return_value=None)
    def test_info_non_existent_server(self, _: mock.MagicMock) -> None:
        self.assertIsNone(commands.info("slurm:///job-does-not-exist"))

    @mock.patch("monarch.tools.commands.Runner.describe")
    @mock.patch("monarch.tools.commands.Runner.status")
    def test_info(
        self, mock_status: mock.MagicMock, mock_describe: mock.MagicMock
    ) -> None:
        appstatus = AppStatus(state=AppState.RUNNING)
        mock_status.return_value = appstatus

        appdef = AppDef(
            name="monarch_test_123",
            roles=[
                Role(
                    name="trainer",
                    image="__unused__",
                    num_replicas=4,
                    port_map={"mesh": 26501},
                )
            ],
            metadata={
                "monarch/meshes/trainer/host_type": "gpu.medium",
                "monarch/meshes/trainer/gpus": "2",
            },
        )
        mock_describe.return_value = appdef

        self.assertEqual(
            ServerSpec(
                name="monarch_test_123",
                state=appstatus.state,
                meshes=[
                    MeshSpec(
                        name="trainer",
                        num_hosts=4,
                        host_type="gpu.medium",
                        gpus=2,
                        port=26501,
                    )
                ],
            ),
            commands.info("slurm:///job-id"),
        )
