# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import string
from dataclasses import dataclass
from typing import Any, Optional

from torchx import specs

DEFAULT_REMOTE_ALLOCATOR_PORT = 26600

_TAG_MESHES_PREFIX = "monarch/meshes/${mesh_name}/"
_TAG_HOST_TYPE: str = _TAG_MESHES_PREFIX + "host_type"
_TAG_GPUS: str = _TAG_MESHES_PREFIX + "gpus"


@dataclass
class MeshSpec:
    """Doubles as the 'input' specifications of how to setup the mesh role
    when submitting the job and as the 'info' (describe) API's return value.
    """

    name: str
    num_hosts: int
    host_type: str
    gpus: int
    port: int = DEFAULT_REMOTE_ALLOCATOR_PORT


def _tag(mesh_name: str, tag_template: str) -> str:
    return string.Template(tag_template).substitute(mesh_name=mesh_name)


def tag_as_metadata(mesh_spec: MeshSpec, appdef: specs.AppDef) -> None:
    appdef.metadata[_tag(mesh_spec.name, _TAG_HOST_TYPE)] = mesh_spec.host_type
    appdef.metadata[_tag(mesh_spec.name, _TAG_GPUS)] = str(mesh_spec.gpus)


def mesh_spec_from_metadata(appdef: specs.AppDef, mesh_name: str) -> Optional[MeshSpec]:
    for role in appdef.roles:
        if role.name == mesh_name:
            return MeshSpec(
                name=mesh_name,
                num_hosts=role.num_replicas,
                host_type=appdef.metadata.get(_tag(mesh_name, _TAG_HOST_TYPE), ""),
                gpus=int(appdef.metadata.get(_tag(mesh_name, _TAG_GPUS), "-1")),
                port=role.port_map.get("mesh", DEFAULT_REMOTE_ALLOCATOR_PORT),
            )

    return None


def mesh_spec_from_str(mesh_spec_str: str) -> MeshSpec:
    """Parses the given string into a MeshSpec.

    Args:
        mesh_spec_str: A string representation of the mesh specification
            in the format 'NAME:NUM_HOSTS:HOST_TYPE' (e.g. 'trainer:8:gpu.medium').
    """
    parts = mesh_spec_str.split(":")
    assert (
        len(parts) == 3
    ), f"`{mesh_spec_str}` is not of the form 'NAME:NUM_HOSTS:HOST_TYPE'"

    name, num_hosts, host_type = parts
    gpus = specs.resource(h=host_type).gpu

    assert num_hosts.isdigit(), f"`{num_hosts}` is not a number in: {mesh_spec_str}"

    return MeshSpec(name, int(num_hosts), host_type, gpus)


@dataclass
class ServerSpec:
    """Holds information (as returned by the 'describe' API of the scheduler)
    about the monarch server. This is the return value of ``monarch.tools.commands.info` API.
    """

    name: str
    state: specs.AppState
    meshes: list[MeshSpec]

    def get_mesh_spec(self, mesh_name: str) -> MeshSpec:
        for mesh_spec in self.meshes:
            if mesh_spec.name == mesh_name:
                return mesh_spec

        raise ValueError(
            f"Mesh: '{mesh_name}' not found in job: {self.name}. Try one of: {self.get_mesh_names()}"
        )

    def get_mesh_names(self) -> list[str]:
        return [m.name for m in self.meshes]

    def to_json(self) -> dict[str, Any]:
        """Returns the JSON form of this struct that can be printed to console by:

        .. code-block:: python

            import json

            server_spec = ServerSpec(...)
            print(json.dumps(server_spec, indent=2))
        """

        return {
            "name": self.name,
            "state": self.state.name,
            "meshes": {
                mesh.name: {
                    "host_type": mesh.host_type,
                    "hosts": mesh.num_hosts,
                    "gpus": mesh.gpus,
                }
                for mesh in self.meshes
            },
        }
