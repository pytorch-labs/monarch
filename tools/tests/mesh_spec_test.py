# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import json
import unittest
from dataclasses import asdict

from monarch.tools.mesh_spec import (
    mesh_spec_from_metadata,
    mesh_spec_from_str,
    MeshSpec,
    ServerSpec,
    tag_as_metadata,
)

from torchx import specs

UNUSED = "__UNUSED_FOR_TEST__"


class MeshSpecTest(unittest.TestCase):
    def test_tag_as_metadata(self) -> None:
        mesh_spec = MeshSpec(
            name="trainer", num_hosts=2, host_type="gpu.medium", gpus=2
        )
        appdef = specs.AppDef(name=UNUSED)
        tag_as_metadata(mesh_spec, appdef)

        self.assertDictEqual(
            {
                "monarch/meshes/trainer/host_type": "gpu.medium",
                "monarch/meshes/trainer/gpus": "2",
            },
            appdef.metadata,
        )

    def test_mesh_spec_from_str_incomplete_spec(self) -> None:
        for spec_str in [
            "2:gpu.medium",  # missing_mesh_name
            "trainer:gpu.medium",  # missing_num_hosts
            "trainer:2",  # missing_host_type
        ]:
            with self.assertRaisesRegex(
                AssertionError, r"not of the form 'NAME:NUM_HOSTS:HOST_TYPE'"
            ):
                mesh_spec_from_str(spec_str)

    def test_mesh_spec_from_str_num_hosts_not_an_integer(self) -> None:
        with self.assertRaisesRegex(AssertionError, "is not a number"):
            mesh_spec_from_str("trainer:four:gpu.medium")

    def test_mesh_spec_from_str(self) -> None:
        mesh_spec_str = "trainer:4:gpu.small"
        mesh_spec = mesh_spec_from_str(mesh_spec_str)

        self.assertEqual("trainer", mesh_spec.name)
        self.assertEqual(4, mesh_spec.num_hosts)
        self.assertEqual("gpu.small", mesh_spec.host_type)
        self.assertEqual(1, mesh_spec.gpus)

    def test_mesh_spec_from_metadata(self) -> None:
        appdef = specs.AppDef(
            name=UNUSED,
            roles=[specs.Role(name="trainer", image=UNUSED, num_replicas=4)],
            metadata={
                "monarch/meshes/trainer/host_type": "gpu.medium",
                "monarch/meshes/trainer/gpus": "2",
            },
        )
        trainer_mesh_spec = mesh_spec_from_metadata(appdef, "trainer")
        self.assertIsNotNone(trainer_mesh_spec)
        self.assertEqual(4, trainer_mesh_spec.num_hosts)
        self.assertEqual("gpu.medium", trainer_mesh_spec.host_type)
        self.assertEqual(2, trainer_mesh_spec.gpus)

        # no generator role in appdef
        self.assertIsNone(mesh_spec_from_metadata(appdef, "generator"))

    def test_mesh_spec_can_dump_as_json(self) -> None:
        mesh_spec = MeshSpec(
            name="trainer", num_hosts=4, host_type="gpu.medium", gpus=2
        )
        expected = """
{
  "name": "trainer",
  "num_hosts": 4,
  "host_type": "gpu.medium",
  "gpus": 2,
  "port": 26600
}
"""
        self.assertEqual(expected.strip("\n"), json.dumps(asdict(mesh_spec), indent=2))


class ServerSpecTest(unittest.TestCase):
    def get_test_server_spec(self) -> ServerSpec:
        return ServerSpec(
            name="monarch-foo-1a2b3c",
            state=specs.AppState.RUNNING,
            meshes=[
                MeshSpec(name="trainer", num_hosts=4, host_type="gpu.medium", gpus=2),
                MeshSpec(name="generator", num_hosts=8, host_type="gpu.small", gpus=1),
            ],
        )

    def test_get_mesh_spec(self) -> None:
        server_spec = self.get_test_server_spec()
        mesh_spec = server_spec.get_mesh_spec("trainer")

        self.assertEqual("trainer", mesh_spec.name)
        self.assertEqual(4, mesh_spec.num_hosts)
        self.assertEqual(2, mesh_spec.gpus)
        self.assertEqual("gpu.medium", mesh_spec.host_type)

    def test_get_mesh_spec_not_found(self) -> None:
        server_spec = self.get_test_server_spec()
        with self.assertRaisesRegex(
            ValueError,
            r"Mesh: 'worker' not found in job: monarch-foo-1a2b3c. Try one of: \['trainer', 'generator'\]",
        ):
            server_spec.get_mesh_spec("worker")
