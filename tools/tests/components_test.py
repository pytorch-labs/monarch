# pyre-strict

import getpass
import unittest

from monarch.tools.components import base, conda
from monarch.tools.mesh_spec import DEFAULT_REMOTE_ALLOCATOR_PORT
from torchx import specs

from torchx.components.component_test_base import ComponentTestCase


class ComponentsTest(unittest.TestCase):
    def test_create_default(self) -> None:
        appdef = conda.hyperactor()
        job_name = f"monarch-{getpass.getuser()}"

        roles = appdef.roles
        self.assertEqual(1, len(roles))

        mesh = roles[0]

        # --- check name, # replicas, resources ---
        self.assertEqual(appdef.name, job_name)
        self.assertEqual(1, len(roles))

        # defaults to a single host mesh on gtt_any
        self.assertEqual(mesh.name, "mesh_0")
        self.assertEqual(1, mesh.num_replicas)
        self.assertEqual(mesh.resource, specs.resource(h="gtt_any"))

        # --- check port settings ---
        default_port = DEFAULT_REMOTE_ALLOCATOR_PORT
        self.assertDictEqual(mesh.port_map, {"mesh": default_port})

        # --- check entrypoint --
        self.assertEqual(
            "/packages/monarch/hyperactor", mesh.entrypoint.split(" && ")[-1]
        )
        self.assertEqual(
            [
                "--num-hosts=1",
                "mesh-worker",
                f"--port={default_port}",
                f"--program={specs.macros.img_root}/projects/monarch/scripts/run_monarch_bootstrap.sh",
            ],
            mesh.args,
        )

        # --- check common env vars ---
        for env_var in {
            "PRELOAD_PATH",
            "TORCHX_RUN_PYTHONPATH",
            "XLF_SYSTEMD_SERVICES",
            "DUMP_DIR",
            *conda._DEFAULT_ENV.keys(),
        }:
            self.assertIn(env_var, mesh.env)

        # --- check mesh spec tags ---
        appdef.metadata["monarch/meshes/mesh_0/host_type"] = "gtt_any"
        appdef.metadata["monarch/meshes/mesh_0/gpus"] = "8"

    def test_create_two_meshes(self) -> None:
        mesh_specs = [
            "trainer:2:gtt_any_8",
            "generator:8:gtt_any_4",
        ]
        appdef = conda.hyperactor(meshes=mesh_specs)

        self.assertEqual(len(mesh_specs), len(appdef.roles))

        for i, mesh_spec_literal in enumerate(mesh_specs):
            name, num_hosts, host_type = mesh_spec_literal.split(":")

            # just check the mesh spec arguments in this test
            mesh_role = appdef.roles[i]
            self.assertEqual(name, mesh_role.name)
            self.assertEqual(int(num_hosts), mesh_role.num_replicas)
            self.assertEqual(specs.resource(h=host_type), mesh_role.resource)

    def test_bad_mesh_spec(self) -> None:
        missing_field_msg = r"not of the form 'NAME:NUM_HOSTS:HOST_TYPE'"
        with self.assertRaisesRegex(AssertionError, missing_field_msg):
            missing_mesh_name = ["4:gtt_any_8"]
            base.hyperactor(meshes=missing_mesh_name)

        with self.assertRaisesRegex(AssertionError, missing_field_msg):
            missing_host_type = ["trainer:4"]
            base.hyperactor(meshes=missing_host_type)

        with self.assertRaisesRegex(AssertionError, missing_field_msg):
            missing_num_hosts = ["trainer:gtt_any_8"]
            base.hyperactor(meshes=missing_num_hosts)

        with self.assertRaisesRegex(AssertionError, r"is not a number"):
            num_hosts_not_a_number = ["trainer:two:gtt_any"]
            base.hyperactor(meshes=num_hosts_not_a_number)

        with self.assertRaisesRegex(KeyError, r"No named resource found for `foobar`"):
            invalid_host_type = ["trainer:2:foobar"]
            base.hyperactor(meshes=invalid_host_type)


class ValidateComponentFunctionSyntax(ComponentTestCase):
    def test_hyperactor(self) -> None:
        self.validate(conda, "hyperactor")
        self.validate(base, "hyperactor")
