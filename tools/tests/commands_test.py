# pyre-strict

import argparse
import json
import unittest
from unittest import mock

from facebook.hpc_scheduler.hpcscheduler.types import HpcJobDefinition

from monarch.tools import commands
from monarch.tools.commands import component_args_from_cli, Config
from monarch.tools.mesh_spec import MeshSpec, ServerSpec
from torchx.specs import AppDef, AppDryRunInfo, AppState, AppStatus, macros, Role


def test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scheduler")
    parser.add_argument(
        "-cfg",
        "--scheduler_args",
        action="append",
    )
    parser.add_argument("--dryrun", action="store_true")
    return parser


class CommandsTest(unittest.TestCase):
    def test_config_apply_cli_args(self) -> None:
        args = test_parser().parse_args(
            [
                # supports both 'comma'-delimited and repeated '-cfg'
                "-cfg=conda_fbpkg_id=foo,conda_dir=bar",
                "-cfg=hpcIdentity=baz",
            ]
        )

        config = Config()
        config.apply_cli_args(args)

        self.assertEqual(config.scheduler_args.pop("conda_fbpkg_id"), "foo")
        self.assertEqual(config.scheduler_args.pop("conda_dir"), "bar")
        self.assertEqual(config.scheduler_args.pop("hpcIdentity"), "baz")

        # check the rest against defaults
        default = Config()
        for key, value in config.scheduler_args.items():
            self.assertEqual(value, default.scheduler_args[key])

    def test_config_cannot_override_enable_legacy(self) -> None:
        args = test_parser().parse_args(["-cfg", "enableLegacy=True"])
        config = Config()

        with self.assertRaisesRegex(
            AssertionError, "`enableLegacy` cannot be overridden"
        ):
            config.apply_cli_args(args)

    def test_component_args_from_cli(self) -> None:
        def fn(h: str, num_hosts: int) -> AppDef:
            return AppDef("_unused_", roles=[Role("_unused_", "_unused_")])

        args = component_args_from_cli(fn, ["h=gtt_any", "num_hosts=4"])

        # should be able to call the component function with **args as kwargs
        self.assertIsNotNone(fn(**args))
        self.assertDictEqual({"h": "gtt_any", "num_hosts": 4}, args)

    def test_create_dryrun(self) -> None:
        config = Config(dryrun=True)
        dryrun_info = commands.create(config)()
        # need only assert that the return type of dryrun is a dryrun info object
        # since we delegate to torchx for job submission
        self.assertIsInstance(dryrun_info, AppDryRunInfo)

    def test_create_dryrun_is_json(self) -> None:
        # assert that the dryrun info can be loaded as JSON
        config = Config(dryrun=True)
        dryrun_info = commands.create(config)()
        self.assertTrue(json.loads(str(dryrun_info)))

    def test_mast_conda(self) -> None:
        config = Config()
        config.scheduler_args["conda_fbpkg_id"] = "_DUMMY_FBPKG_:0"
        config.scheduler_args["conda_path_in_fbpkg"] = "dummy/conda"

        config.dryrun = True

        dryrun_info: AppDryRunInfo = commands.create(config)(  # pyre-ignore[9]
            meshes=[
                "mesh0:1:gtt_any",
                "mesh1:1:gtt_any",
            ],
            env={"IMG_ROOT_MACRO": macros.img_root},
        )

        # these assertions only apply for scheduler == mast_conda
        self.assertEqual("mast_conda", dryrun_info._scheduler)

        self.assertIsNotNone(dryrun_info._app)
        appdef: AppDef = dryrun_info._app

        for role in appdef.roles:
            conda_fbpkg_id = config.scheduler_args["conda_fbpkg_id"]
            conda_fbpkg_name = conda_fbpkg_id.split(":")[0]  # pyre-ignore[16]
            conda_path = config.scheduler_args["conda_path_in_fbpkg"]
            workspace_fbpkg_name = config.scheduler_args["workspace_fbpkg_name"]

            self.assertEqual(
                f"/packages/{conda_fbpkg_name}/{conda_path}", role.env["CONDA_DIR"]
            )
            self.assertEqual(
                f"/packages/{workspace_fbpkg_name}", role.env["WORKSPACE_DIR"]
            )

        hpc_job_def: HpcJobDefinition = dryrun_info.request
        for task_group_def in hpc_job_def.hpcTaskGroups:
            env = task_group_def.spec.env
            self.assertEqual(env["WORKSPACE_DIR"], env["IMG_ROOT_MACRO"])

    @mock.patch("torchx.schedulers.fb.mast_conda_scheduler.MastCondaScheduler.schedule")
    def test_create(self, mock_schedule: mock.MagicMock) -> None:
        mock_schedule.return_value = "test_job_id"
        config = Config()
        config.scheduler_args["conda_fbpkg_id"] = "_UNUSED_"

        server_handle = commands.create(config)()
        mock_schedule.assert_called_once()
        self.assertEqual(server_handle, "mast_conda:///test_job_id")

    @mock.patch("monarch.tools.commands.Runner.cancel")
    def test_kill(self, mock_cancel: mock.MagicMock) -> None:
        handle = "mast_conda:///test_job_id"
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
            roles=[Role(name="trainer", image="__unused__", num_replicas=4)],
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
                    )
                ],
            ),
            commands.info("slurm:///job-id"),
        )
