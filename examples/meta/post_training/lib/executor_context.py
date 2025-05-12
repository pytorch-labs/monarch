import asyncio
import logging
import pickle
from typing import Any, Dict, Tuple

import torch
from monarch.rust_local_mesh import (
    local_mesh_provider,
    LoggingLocation,
    SocketType,
    SupervisionParams,
)

from .communication.monarch_comms import attach_to_inter_mesh_pipe

from .executor import make_monarch_executor

logger: logging.Logger = logging.getLogger(__name__)


def reconstruct_model_state_dict_key_to_shape(
    shape_tensor: torch.Tensor,
) -> Dict[str, Any]:
    serialized_shape_map = bytes(shape_tensor.tolist())
    _model_state_dict_key_to_shape = pickle.loads(serialized_shape_map)  # noqa: PYTHONPICKLEISBAD
    return _model_state_dict_key_to_shape


class ExecutorContext:
    """
    ExecutorContext is responsible for initializing and managing the execution context for a group of executors.
    It sets up the necessary meshes and executors based on the provided configuration group.

    Attributes:
        executor_configs_group: A list of executor configuration dictionaries.
        total_number_devices: Total number of devices calculated from the executor configurations.
        name_to_mesh: A dictionary mapping executor names to their corresponding mesh objects.
        name_to_executor: A dictionary mapping executor names to their initialized executor objects.
        name_to_state_dict_key_to_shape: A dictionary mapping executor names to their model's state
                                         dictionary key-to-shape mappings.
    """

    def __init__(self, executor_configs_group, params=None):
        logger.info("#### start executor context init ####")
        self.executor_configs_group = executor_configs_group
        self.params = params
        self.multi_mesh = (
            params is not None
            and hasattr(self.params, "multi_mesh")
            and self.params.multi_mesh
        )
        self.meshes = []
        self.o_mesh = None
        for executor_config in executor_configs_group:
            assert hasattr(
                executor_config, "num_processes"
            ), "num_processes is required"
            assert hasattr(executor_config, "path"), "path is required"
            assert hasattr(executor_config, "name"), "name is required"
        self.total_number_devices = sum(
            executor_config.num_processes for executor_config in executor_configs_group
        )
        logger.info("#### start create executor meshes ####")
        (
            self.name_to_mesh,
            self.name_to_executor,
            self.name_to_state_dict_key_to_shape,
            self.name_to_pipe,
        ) = self._get_global_meshes_for_executors(executor_configs_group)
        logger.info(
            f"#### finish create executor group #### \n{self.name_to_mesh=}, {self.name_to_executor=}, {self.name_to_state_dict_key_to_shape=} {self.name_to_pipe=}"
        )

    def shutdown(self):
        """
        Shuts down the executor context by exiting and deactivating the mesh if it exists.
        Logs a message indicating the completion of the mesh destruction process.
        """
        if self.o_mesh is not None:
            self.o_mesh.exit()
            self.o_mesh.deactivate()
        else:
            for mesh in self.meshes:
                mesh.exit()
                mesh.deactivate()
        logger.info("#### finish destroy meshes ####")

    def _get_global_meshes_for_executors(
        self, executor_configs_group
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Creates and returns global meshes for executors based on the provided configuration group.

        Args:
            executor_configs_group: A list of executor configuration dictionaries, each containing
                                    parameters like 'name', 'path', 'num_processes', and 'params'.

        Returns:
            A tuple containing:
            - name_to_mesh: A dictionary mapping executor names to their corresponding mesh objects.
            - name_to_executor: A dictionary mapping executor names to their initialized executor objects.
            - name_to_state_dict_key_to_shape: A dictionary mapping executor names to their model's state
              dictionary key-to-shape mappings.
            - name_to_pipe: A dictionary mapping executor names to their inter-mesh pipe objects.
        """
        num_meshes = len(executor_configs_group)
        if self.multi_mesh:
            mesh_provider, bootstrap = local_mesh_provider(
                meshes=num_meshes,
                hosts_per_mesh=1,
                gpus_per_host=executor_configs_group[0].num_processes,
                socket_type=SocketType.UNIX,
                logging_location=LoggingLocation.DEFAULT,
                supervision_params=SupervisionParams(
                    update_timeout_in_sec=10,  # Fail fast
                    query_interval_in_sec=1,
                    update_interval_in_sec=1,
                ),
                auto_epoch=True,
            )
            executor_meshes = []
            for _ in range(num_meshes):
                mesh = mesh_provider.new_mesh()
                executor_meshes.append(mesh)
        else:
            mesh_provider, bootstrap = local_mesh_provider(
                meshes=num_meshes,
                hosts_per_mesh=1,
                gpus_per_host=self.total_number_devices,
                socket_type=SocketType.UNIX,
                logging_location=LoggingLocation.DEFAULT,
                supervision_params=SupervisionParams(
                    update_timeout_in_sec=10,  # Fail fast
                    query_interval_in_sec=1,
                    update_interval_in_sec=1,
                ),
                auto_epoch=True,
            )
            self.o_mesh = mesh_provider.new_mesh()
            meshes = self.o_mesh.flatten("gpu").split(
                gpu=("executors", "dp", "pp", "tp"),
                executors=len(executor_configs_group),
                pp=executor_configs_group[0].pipeline_parallel_size,
                tp=executor_configs_group[0].model_parallel_size,
            )
            torch.set_default_device("cuda")

            executor_meshes = [
                meshes(executors=i)
                for i in range(
                    len(executor_configs_group),
                )
            ]
        torch.set_default_device("cuda")
        self.meshes = executor_meshes
        logger.info(f"#### executor_meshes: {executor_meshes} ####")
        monarch_executors = []
        name_to_mesh = {}
        name_to_executor = {}
        name_to_pipe = {}
        name_to_state_dict_key_to_shape = {}

        for executor_mesh, executor_config in zip(
            executor_meshes, executor_configs_group
        ):
            name_to_mesh[executor_config.name] = executor_mesh
            with executor_mesh.activate():
                monarch_executor = make_monarch_executor(
                    executor_config.path,
                    name=executor_config.name,
                    num_processes=executor_config.num_processes,
                    params=executor_config,
                )
                monarch_executor.init()
                monarch_executors.append(monarch_executor)
                name_to_executor[executor_config.name] = monarch_executor

                state_dict_key_to_shape_tensor = (
                    monarch_executor.call_method_on_shard_and_fetch(
                        "get_model_state_dict_key_to_shape"
                    ).result()
                )
                state_dict_key_to_shape = reconstruct_model_state_dict_key_to_shape(
                    state_dict_key_to_shape_tensor
                )
                logger.info(f"state_dict_key_to_shape: {state_dict_key_to_shape}")
                name_to_state_dict_key_to_shape[executor_config.name] = (
                    state_dict_key_to_shape
                )
                if self.multi_mesh:
                    name_to_pipe[executor_config.name] = attach_to_inter_mesh_pipe()

        return (
            name_to_mesh,
            name_to_executor,
            name_to_state_dict_key_to_shape,
            name_to_pipe,
        )
