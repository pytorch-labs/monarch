"""
TorchX component definition for Monarch server.
Basically a job definition for creating a Monarch server.

See: https://pytorch.org/torchx/main/basics.html#components
"""

# pyre-strict
import copy
import getpass
from typing import Dict, List, Optional

import torchx.components.fb.conda as conda
import torchx.components.fb.conda_transforms as conda_transforms
import torchx.specs as specs
from monarch.tools.mesh_spec import MeshSpec, tag_as_metadata

_TAGS = ["monarch"]

_IGNORED = 1

_DEFAULT_MESHES = ["mesh_0:1:gtt_any"]


# TODO kiuk@ move to torchx.components.fb.conda
class Packages:
    """Builder for adding various types of fbpkgs to the job spec."""

    def __init__(self) -> None:
        self.packages: list[str] = []
        self.python_paths: list[str] = []
        self.preload_paths: list[str] = []

    def _name(self, fbpkg: str) -> str:
        return fbpkg.split(":")[0]

    def add_python_lib(self, fbpkg: str, libdir: str = "lib") -> None:
        """Use for adding python library fbpkgs that need to be added to PYTHONPATH."""
        self.packages.append(fbpkg)
        self.python_paths.append(f"/packages/{self._name(fbpkg)}/{libdir}")

    def add_shared_lib(self, fbpkg: str, libpath: str) -> None:
        """Use for adding *.so libs that need to be added to PRELOAD_PATH."""
        self.packages.append(fbpkg)
        self.preload_paths.append(f"/packages/{self._name(fbpkg)}/{libpath}")

    def add_package(self, fbpkg: str) -> None:
        """Use to add fbpkgs with scripts or binaries that do not need any **PATH manipulations."""
        self.packages.append(fbpkg)

    def PRELOAD_PATH(self) -> str:
        return ":".join(self.preload_paths)

    def PYTHONPATH(self) -> str:
        return ":".join(self.python_paths)

    def image(self) -> str:
        return ";".join(self.packages)


_EMPTY_PACKAGES = Packages()


def hyperactor(
    # TODO kiuk@ figure out a better way to pass mesh specs
    #  right now TorchX component function signature is limited to
    #  primitives and list, dict of primitives so we encode
    #  mesh specs as ':' delimited str for now
    meshes: List[str] = _DEFAULT_MESHES,
    env: Optional[Dict[str, str]] = None,
    port: int = 29600,
    hyperactor_fbpkg: str = "monarch:prod",
    additional_packages: Packages = _EMPTY_PACKAGES,
    pre_launch_cmd: Optional[str] = None,
    # TODO kiuk@ [3/n][monarch] create run_monarch_mesh_worker.sh and hook it up here
    program: str = f"{specs.macros.img_root}/projects/monarch/scripts/run_monarch_bootstrap.sh",
    systemd_services: Optional[Dict[str, bool]] = None,
) -> specs.AppDef:
    """Creates a Monarch server on hyperactor per the given parameters.

    Args:
        meshes: list of mesh specs of the form "{name}:{num_hosts}:{host_type}"
    """

    systemd_services = systemd_services or {}
    env = env or {}

    packages = additional_packages
    packages.add_python_lib(hyperactor_fbpkg)

    launch_cmd = "/packages/" + hyperactor_fbpkg.split(":")[0] + "/hyperactor"
    # TODO it might not always be &&
    entrypoint = (
        launch_cmd if pre_launch_cmd is None else pre_launch_cmd + " && " + launch_cmd
    )

    job_name = f"monarch-{getpass.getuser()}"

    mesh_specs = []
    for mesh in meshes:
        parts = mesh.split(":")
        assert (
            len(parts) == 3
        ), f"`{mesh}` is not of the form 'NAME:NUM_HOSTS:HOST_TYPE'"

        name, num_hosts, host_type = parts

        assert num_hosts.isdigit(), f"`{num_hosts}` is not a number in: {mesh}"

        mesh_specs.append(MeshSpec(name, int(num_hosts), host_type))

    # get conda-on-mast role template
    appdef = conda.run(
        *[entrypoint],
        name=job_name,
        h="IGNORED",  # overridden later
        num_nodes=_IGNORED,  # overridden later
        env=env,
        run_as_root=True,
        enable_ttls=True,
    )
    template_role = appdef.roles[0]
    template_role.image = packages.image()
    template_role.port_map |= {"mesh": port}

    mesh_roles = []
    for mesh in mesh_specs:
        mesh_role = copy.deepcopy(template_role)
        mesh_role.name = mesh.name
        mesh_role.resource = specs.resource(h=mesh.host_type)
        mesh_role.num_replicas = mesh.num_hosts
        mesh_role.args = [
            # not needed for mesh-worker subcmd but is a global arg
            # TODO extract mesh-worker subcmd into its own entrypoint
            f"--num-hosts={_IGNORED}",
            "mesh-worker",
            f"--port={port}",
            f"--program={program}",
        ]
        mesh.gpus = mesh_role.resource.gpu
        tag_as_metadata(mesh, appdef)
        mesh_roles.append(mesh_role)

    # overwrite appdef's roles with the mesh roles we created
    appdef.roles = mesh_roles

    conda_transforms.append_tb_logdir_metadata(appdef)
    appdef.metadata["tags"] = ",".join(_TAGS)
    return appdef


# TODO kiuk@ move to torchx.workspace.fb.conda_workspace
def make_fbpkg(
    fbpkg_name: str,
    paths: list[str],
) -> Optional[str]:
    if not paths:
        return None
    else:
        from torchx.workspace.fb import fbpkg_utils

        return fbpkg_utils.build_fbpkg(
            fbpkg_name=fbpkg_name,
            paths=paths,
            expiration="4w",
        )
