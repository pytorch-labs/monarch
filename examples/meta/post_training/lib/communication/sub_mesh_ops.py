import logging

import os

import pickle
from typing import Any, Dict, List, Tuple

import torch
from monarch import Tensor
from post_training.lib.executor import MonarchExecutor

logger: logging.Logger = logging.getLogger(__name__)


DUMMY_MODEL_OPS = int(os.getenv("DUMMY_MODEL_OPS", "0")) == 1


def reconstruct_model_state_dict_key_to_shape(
    shape_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """
    Reconstructs a dictionary mapping model state dict keys to their shapes.

    Args:
        shape_tensor (torch.Tensor): A tensor containing serialized shape information.

    Returns:
        Dict[str, Any]: A dictionary mapping model state dict keys to their corresponding shapes.
    """
    serialized_shape_map = bytes(shape_tensor.tolist())
    _model_state_dict_key_to_shape = pickle.loads(serialized_shape_map)  # noqa: PYTHONPICKLEISBAD
    return _model_state_dict_key_to_shape


def submesh_weight_send_op(
    channel_name,
    outbound_mesh,
    inbound_mesh,
    outbound_executor: MonarchExecutor,
    inbound_executor: MonarchExecutor,
    outbound_executor_pipe,
    inbound_executor_pipe,
    model_state_dict_key_to_shape,
) -> Tuple[Dict[str, Any], List[Tensor]]:
    """
    Sends model weights from one mesh to another using the communication channel.

    Args:
        communication_channel (CommunicationChannel): The channel used for communication between meshes.
        executor_context (ExecutorContext): The context containing mesh information.
        executor (Executor): The executor responsible for handling model weights.
        executor_name (str): The name of the executor.

    Returns:
        Tuple[Dict[str, Any], List[Tensor]]: A tuple containing:
            - A dictionary mapping model state dict keys to their corresponding weights on the other mesh.
            - A list of monarch Tensors stored on the remote mesh, which can be used as handles by calling fetch_shard(handle).result()

    """
    if DUMMY_MODEL_OPS:
        logger.info("skipping model weight send")
        return {}, []

    output_dict = {}
    for key, shape in model_state_dict_key_to_shape.items():  # noqa: 16
        with outbound_mesh.activate():
            weight = outbound_executor.get_model_weight(key, shape).to("cuda")
            weight_on_other_mesh = weight.to_mesh(inbound_mesh)  # noqa: 16
            logger.info(
                f"submesh_weight_send_op: key: {key}, shape: {shape} %s %s",
                weight,
                weight_on_other_mesh,
            )
            output_dict[key] = weight_on_other_mesh
    return output_dict, []


def submesh_weight_recv_op(
    channel_name,
    outbound_mesh,
    inbound_mesh,
    outbound_executor: MonarchExecutor,
    inbound_executor: MonarchExecutor,
    outbound_executor_pipe,
    inbound_executor_pipe,
    output_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tensor]]:
    """
    Receives model weights from another mesh and loads them into the executor's state dict.

    Args:
        communication_channel (CommunicationChannel): The channel used for communication between meshes.
        executor_context (ExecutorContext): The context containing mesh information.
        executor (Executor): The executor responsible for handling model weights.
        executor_name (str): The name of the executor.
        output_dict (Dict[str, Any]): A dictionary mapping model state dict keys to their corresponding weights.

    Returns:
        Tuple[Dict[str, Any], List[Tensor]]: A tuple containing:
            - A dictionary mapping model state dict keys to their corresponding weights on the other mesh.
            - A list of monarch Tensors stored on the remote mesh, which can be used as handles by calling fetch_shard(handle).result()
    """
    if DUMMY_MODEL_OPS:
        logger.info("skipping model weight recv")
        return {}, []
    handles = []
    with inbound_mesh.activate():
        handles.append(inbound_executor.load_state_dict(output_dict))

    return {}, handles


def submesh_send_op(
    channel_name,
    outbound_mesh,
    inbound_mesh,
    outbound_executor: MonarchExecutor,
    inbound_executor: MonarchExecutor,
    outbound_executor_pipe,
    inbound_executor_pipe,
) -> Tuple[Dict[str, Any], List[Tensor]]:
    """
    Sends output tensors from one mesh to another using the communication channel.

    Args:
        communication_channel (CommunicationChannel): The channel used for communication between meshes.
        executor_context (ExecutorContext): The context containing mesh information.
        executor (Executor): The executor responsible for handling output tensors.
        executor_name (str): The name of the executor.

    Returns:
        Tuple[Dict[str, Any], List[Tensor]]: A tuple containing:
            - A dictionary mapping output keys to their corresponding tensors on the other mesh.
            - A list of monarch Tensors stored on the remote mesh, which can be used as handles by calling fetch_shard(handle).result()
    """
    if DUMMY_MODEL_OPS:
        logger.info("skipping model weight send")
        return {}, []

    with outbound_mesh.activate():
        output_dict_key_to_shape_tensor = (
            outbound_executor.call_method_on_shard_and_fetch(  # noqa: 16
                "get_model_output_dict_key_to_shape"
            ).result()
        )
        output_dict_key_to_shape = reconstruct_model_state_dict_key_to_shape(
            output_dict_key_to_shape_tensor
        )
        logger.info(f"output_dict_key_to_shape: {output_dict_key_to_shape}")

        output_dict = {}
        for key, shape in output_dict_key_to_shape.items():
            with outbound_mesh.activate():
                weight = outbound_executor.get_output_tensor(key, shape).to("cuda")
                weight_on_other_mesh = weight.to_mesh(inbound_mesh)  # noqa: 16
                logger.info(
                    f"submesh_send_op: key: {key}, shape: {shape} %s %s",
                    weight,
                    weight_on_other_mesh,
                )
                output_dict[key] = weight_on_other_mesh
        return output_dict, []


def submesh_recv_op(
    channel_name,
    outbound_mesh,
    inbound_mesh,
    outbound_executor: MonarchExecutor,
    inbound_executor: MonarchExecutor,
    outbound_executor_pipe,
    inbound_executor_pipe,
    output_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tensor]]:
    """
    Receives output tensors from another mesh and returns them.

    Args:
        communication_channel (CommunicationChannel): The channel used for communication between meshes.
        executor_context (ExecutorContext): The context containing mesh information.
        executor (Executor): The executor responsible for handling output tensors.
        executor_name (str): The name of the executor.
        output_dict (Dict[str, Any]): A dictionary mapping output tensor keys to their corresponding tensors.

    Returns:
        Tuple[Dict[str, Any], List[Tensor]]: A tuple containing:
            - A dictionary mapping output keys to their corresponding tensors on the other mesh.
            - A list of monarch Tensors stored on the remote mesh, which can be used as handles by calling fetch_shard(handle).result()
    """
    if DUMMY_MODEL_OPS:
        logger.info("skipping model weight recv")
        return {}, []
    return output_dict, []
