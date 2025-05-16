# pyre-strict

import contextlib
import io
import json
import unittest
from importlib import resources
from typing import Generator
from unittest import mock

from monarch.tools.cli import main
from monarch.tools.mesh_spec import MeshSpec, ServerSpec
from torchx.specs import AppState

_DISABLE_WORKSPACE = ""  # passing --workspace="" disables it for tests


@contextlib.contextmanager
def capture_stdout() -> Generator[io.StringIO, None, None]:
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        yield buf


class MainTest(unittest.TestCase):
    def test_help(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            main(["--help"])
            self.assertEqual(cm.exception.code, 0)

    def test_create_dryrun_default(self) -> None:
        with capture_stdout() as buf:
            main(
                [
                    "create",
                    "--dryrun",
                    f"--workspace={_DISABLE_WORKSPACE}",
                    "-cfg=conda_fbpkg_id=_unused_:1",
                ]
            )
            out = buf.getvalue()
        # correctness of the create command is tested in commands_test.py
        # so just make sure the output of the cli is sane
        self.assertTrue(json.loads(out))

    def test_create_dryrun_two_meshes(self) -> None:
        num_hosts = 4

        with resources.path("monarch.tools.components", "conda.py") as component_file:
            with capture_stdout() as buf:
                main(
                    [
                        "create",
                        "--dryrun",
                        "-cfg=conda_fbpkg_id=_unused_:1",
                        f"--workspace={_DISABLE_WORKSPACE}",
                        f"--component={component_file}:hyperactor",
                        f"-arg=meshes=trainers:{num_hosts}:gtt_any,generators:{num_hosts*2}:gtt_any",
                    ]
                )
                out = buf.getvalue()

            tgs = {tg["name"]: tg for tg in json.loads(out)["hpcTaskGroups"]}
            self.assertEqual(2, len(tgs))
            self.assertEqual(num_hosts, tgs["trainers"]["taskCount"])
            self.assertEqual(num_hosts * 2, tgs["generators"]["taskCount"])

            # for mast_conda scheduler only
            for _, tg in tgs.items():
                env = tg["spec"]["env"]
                for key in ["PYTHON_EXEC", "CONDA_DIR", "WORKSPACE_DIR"]:
                    self.assertIn(key, env)
                    self.assertIsNotNone(env[key])

    @mock.patch("monarch.tools.cli.info")
    def test_info(self, mock_cmd_info: mock.MagicMock) -> None:
        job_name = "imaginary-test-job"
        mock_cmd_info.return_value = ServerSpec(
            name=job_name,
            state=AppState.RUNNING,
            meshes=[
                MeshSpec(name="trainer", num_hosts=4, host_type="gpu.medium", gpus=2),
                MeshSpec(name="generator", num_hosts=16, host_type="gpu.small", gpus=1),
            ],
        )
        with capture_stdout() as buf:
            main(["info", f"slurm:///{job_name}"])
            out = buf.getvalue()
            # CLI does not pretty-print json so that the output can be piped for
            # further processing. Read the captured stdout and pretty-format
            # json so that the expected value reads better
            expected = """
{
  "name": "imaginary-test-job",
  "state": "RUNNING",
  "meshes": {
    "trainer": {
      "host_type": "gpu.medium",
      "hosts": 4,
      "gpus": 2
    },
    "generator": {
      "host_type": "gpu.small",
      "hosts": 16,
      "gpus": 1
    }
  }
}
"""
            self.assertEqual(
                expected.strip("\n"),
                json.dumps(json.loads(out), indent=2),
            )

    @mock.patch("monarch.tools.cli.kill")
    def test_kill(self, mock_cmd_kill: mock.MagicMock) -> None:
        handle = "mast_conda:///test-job-id"
        main(["kill", handle])
        mock_cmd_kill.assert_called_once_with(handle)
