import json
import unittest
from dataclasses import asdict

from monarch.tools.mesh_spec import mesh_spec_from_metadata, MeshSpec, tag_as_metadata

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
  "gpus": 2
}
"""
        self.assertEqual(expected.strip("\n"), json.dumps(asdict(mesh_spec), indent=2))
