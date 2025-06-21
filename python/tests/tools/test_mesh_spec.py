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
from monarch.tools.network import get_sockaddr

from torchx import specs

UNUSED = "__UNUSED_FOR_TEST__"


class TestMeshSpec(unittest.TestCase):
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
                "monarch/meshes/trainer/transport": "tcp",
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
                "monarch/meshes/trainer/transport": "metatls",
            },
        )
        trainer_mesh_spec = mesh_spec_from_metadata(appdef, "trainer")
        self.assertIsNotNone(trainer_mesh_spec)
        self.assertEqual(4, trainer_mesh_spec.num_hosts)
        self.assertEqual("gpu.medium", trainer_mesh_spec.host_type)
        self.assertEqual(2, trainer_mesh_spec.gpus)
        self.assertEqual("metatls", trainer_mesh_spec.transport)

        # no generator role in appdef
        self.assertIsNone(mesh_spec_from_metadata(appdef, "generator"))

    def test_mesh_spec_can_dump_as_json(self) -> None:
        mesh_spec = MeshSpec(
            name="trainer",
            num_hosts=4,
            host_type="gpu.medium",
            gpus=2,
            hostnames=["n0", "n1", "n2", "n3"],
        )
        expected = """
{
  "name": "trainer",
  "num_hosts": 4,
  "host_type": "gpu.medium",
  "gpus": 2,
  "transport": "tcp",
  "port": 26600,
  "hostnames": [
    "n0",
    "n1",
    "n2",
    "n3"
  ]
}
"""
        self.assertEqual(expected.strip("\n"), json.dumps(asdict(mesh_spec), indent=2))

    def test_mesh_spec_server_addrs_empty_hostnames(self) -> None:
        for transport in ["tcp", "metatls"]:
            with self.subTest(transport=transport):
                mesh_spec = MeshSpec(name="x", num_hosts=2, transport=transport)
                self.assertListEqual([], mesh_spec.server_addrs())

    def test_mesh_spec_server_addrs_unsupported_transport(self) -> None:
        for transport in ["unix", "local"]:
            with self.subTest(transport=transport):
                mesh_spec = MeshSpec(
                    name="x",
                    num_hosts=2,
                    transport=transport,
                    hostnames=["node0", "node1"],
                )
                with self.assertRaises(ValueError):
                    mesh_spec.server_addrs()

    def test_mesh_spec_server_addrs_tcp(self) -> None:
        mesh_spec = MeshSpec(
            name="x",
            num_hosts=1,
            port=29000,
            hostnames=["localhost"],
        )

        self.assertListEqual(
            [f"tcp!{get_sockaddr('localhost', 29000)}"],
            mesh_spec.server_addrs(),
        )

    def test_mesh_spec_server_addrs_transport_port_override(self) -> None:
        mesh_spec = MeshSpec(
            name="x",
            num_hosts=1,
            port=29000,
            transport="tcp",
            hostnames=["devgpu001.abc.facebook.com"],
        )
        self.assertListEqual(
            ["metatls!devgpu001.abc.facebook.com:29001"],
            mesh_spec.server_addrs(transport="metatls", port=29001),
        )

    def test_mesh_spec_server_addrs_metatls(self) -> None:
        mesh_spec = MeshSpec(
            name="x",
            num_hosts=1,
            transport="metatls",
            port=29000,
            hostnames=["devgpu001.abc.facebook.com"],
        )
        self.assertListEqual(
            ["metatls!devgpu001.abc.facebook.com:29000"],
            mesh_spec.server_addrs(),
        )


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
