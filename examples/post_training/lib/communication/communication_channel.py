import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from monarch import fetch_shard, Tensor

from post_training.lib.executor_context import ExecutorContext

from .inter_mesh_ops import (
    inter_mesh_recv_op,
    inter_mesh_send_op,
    inter_mesh_weight_recv_op,
    inter_mesh_weight_send_op,
)

from .sub_mesh_ops import (
    submesh_recv_op,
    submesh_send_op,
    submesh_weight_recv_op,
    submesh_weight_send_op,
)

logger: logging.Logger = logging.getLogger(__name__)


class CommunicationType(Enum):
    # Enum value for point-to-point communication within a sub-mesh
    SUB_MESH_P2P = 0
    # Enum value for weight-based point-to-point communication within a sub-mesh
    SUB_MESH_WEIGHT_P2P = 1
    # Enum value for point-to-point communication between different meshes
    INTER_MESH_P2P = 2
    # Enum value for weight-based point-to-point communication between different meshes
    INTER_MESH_WEIGHT_P2P = 3


SEND_OPS = {
    CommunicationType.SUB_MESH_P2P: submesh_send_op,
    CommunicationType.SUB_MESH_WEIGHT_P2P: submesh_weight_send_op,
    CommunicationType.INTER_MESH_P2P: inter_mesh_send_op,
    CommunicationType.INTER_MESH_WEIGHT_P2P: inter_mesh_weight_send_op,
}
RECV_OPS = {
    CommunicationType.SUB_MESH_P2P: submesh_recv_op,
    CommunicationType.SUB_MESH_WEIGHT_P2P: submesh_weight_recv_op,
    CommunicationType.INTER_MESH_P2P: inter_mesh_recv_op,
    CommunicationType.INTER_MESH_WEIGHT_P2P: inter_mesh_weight_recv_op,
}


@dataclass
class BaseCommunicationChannel:
    """
    Represents a base communication channel for data transfer between executors.

    Attributes:
        name (str): The name of the communication channel.
        outbound_executor_name (str): The name of the executor sending data.
        inbound_executor_name (str): The name of the executor receiving data.
        communication_type (CommunicationType): The type of communication being used.
        executor_context (ExecutorContext): The context in which the executors operate.
        handles (Optional[List[Tensor]]): Optional list of tensor handles for communication.
    """

    name: str
    outbound_executor_name: str
    inbound_executor_name: str
    communication_type: CommunicationType
    executor_context: ExecutorContext
    handles: Optional[List[Tensor]] = None

    def step(self, step) -> Dict[str, Any]:
        """
        The actual logic of a single communication step.
        Args:
            step: The current global step.
        Returns:
            A dictionary containing the data to be sent to the inbound executor.
        """
        pass

    def wait_for_complete(self):
        """
        Wait for the communication to complete.
        """
        if self.handles is not None:
            for handle in self.handles:
                fetch_shard(handle).result()


class CommunicationChannel(BaseCommunicationChannel):
    """
    Represents a communication channel between two executors for data transfer.

    This class facilitates the sending and receiving of data between an outbound executor and an inbound executor
    using a specified communication type. It manages the communication process by utilizing the appropriate send
    and receive operations based on the communication type.

    Attributes:
        name (str): The name of the communication channel.
        outbound_executor_name (str): The name of the executor sending data.
        inbound_executor_name (str): The name of the executor receiving data.
        communication_type (CommunicationType): The type of communication being used.
        executor_context (ExecutorContext): The context in which the executors operate.
    """

    def __init__(
        self,
        name: str,
        outbound_executor_name: str,
        inbound_executor_name: str,
        communication_type: CommunicationType,
        executor_context: ExecutorContext,
    ):
        super().__init__(
            name=name,
            outbound_executor_name=outbound_executor_name,
            inbound_executor_name=inbound_executor_name,
            communication_type=communication_type,
            executor_context=executor_context,
        )
        assert (
            outbound_executor_name in executor_context.name_to_executor
            and inbound_executor_name in executor_context.name_to_executor
        ), f"outbound_executor_name: {outbound_executor_name} and inbound_executor_name: {inbound_executor_name} must be in {executor_context.name_to_executor.keys()}"

        self.outbound_executor = self.executor_context.name_to_executor.get(
            self.outbound_executor_name, None
        )
        self.inbound_executor = self.executor_context.name_to_executor.get(
            self.inbound_executor_name, None
        )

        self.outbound_mesh = self.executor_context.name_to_mesh.get(
            self.outbound_executor_name, None
        )
        self.inbound_mesh = self.executor_context.name_to_mesh.get(
            self.inbound_executor_name, None
        )

        self.outbound_executor_pipe = self.executor_context.name_to_pipe.get(
            self.outbound_executor_name, None
        )
        self.inbound_executor_pipe = self.executor_context.name_to_pipe.get(
            self.inbound_executor_name, None
        )
        self.send_op = SEND_OPS[self.communication_type]
        self.recv_op = RECV_OPS[self.communication_type]

    def step(self, step, blocking=True) -> Dict[str, Any]:
        """
        Executes a communication step between executors using the specified communication channel.

        This method ensures that all send and receive operations are completed before moving to the next step.
        It iterates over the handles returned by these operations and calls `fetch_shard(handle).result()` on
        each handle, blocking execution until the operation associated with the handle is finished.

        Args:
            step (int): The current global step.
            blocking (bool): If True, waits for all operations to complete before returning.

        Returns:
            Dict[str, Any]: The input dictionary received from the inbound executor.
        """
        logger.info(f"step:{step} communication_channel {self.name}")

        output_dict, handles = self.send_op(
            channel_name=self.name,
            outbound_mesh=self.outbound_mesh,
            inbound_mesh=self.inbound_mesh,
            outbound_executor=self.outbound_executor,
            inbound_executor=self.inbound_executor,
            outbound_executor_pipe=self.outbound_executor_pipe,
            inbound_executor_pipe=self.inbound_executor_pipe,
        )
        for handle in handles:
            fetch_shard(handle).result()
        handles = []

        input_dict, handles = self.recv_op(
            channel_name=self.name,
            outbound_mesh=self.outbound_mesh,
            inbound_mesh=self.inbound_mesh,
            outbound_executor=self.outbound_executor,
            inbound_executor=self.inbound_executor,
            outbound_executor_pipe=self.outbound_executor_pipe,
            inbound_executor_pipe=self.inbound_executor_pipe,
            output_dict=output_dict,
        )
        if blocking:
            self.wait_for_complete()
        return input_dict


class WeightsCommunicationChannel(CommunicationChannel):
    """
    This class is dedicated to handling weights communication between executors.

    It extends the CommunicationChannel class to specifically manage the transfer of model weights
    between an outbound executor and an inbound executor. The class utilizes the appropriate send
    and receive operations for weight-based communication, ensuring that all operations are completed
    before proceeding to the next step.
    """

    def __init__(
        self,
        name: str,
        outbound_executor_name: str,
        inbound_executor_name: str,
        communication_type: CommunicationType,
        executor_context: ExecutorContext,
        _model_state_dict_key_to_shape: Dict[str, torch.Size],
        _keys_not_supported: Optional[List[str]] = None,
    ):
        """
        Initializes a WeightsCommunicationChannel instance.

        Args:
            name (str): The name of the communication channel.
            outbound_executor_name (str): The name of the executor sending data.
            inbound_executor_name (str): The name of the executor receiving data.
            communication_type (CommunicationType): The type of communication being used.
            executor_context (ExecutorContext): The context in which the executors operate.
            _model_state_dict_key_to_shape (Dict[str, torch.Size]): A dictionary mapping model state dict keys to their shapes.
            _keys_not_supported (Optional[List[str]]): A list of keys that are not supported for communication.
        """
        super().__init__(
            name=name,
            outbound_executor_name=outbound_executor_name,
            inbound_executor_name=inbound_executor_name,
            communication_type=communication_type,
            executor_context=executor_context,
        )
        self._model_state_dict_key_to_shape = _model_state_dict_key_to_shape or {}
        self._keys_not_supported = _keys_not_supported or {}

    def step(self, step, blocking=True) -> Dict[str, Any]:
        """
        Handles the communication between executors through the specified communication channel.

        This function ensures that all send operations and receive operations
        are completed before proceeding to the next step. It does this by iterating
        over the handles returned by the send and receive operations and calling
        `fetch_shard(handle).result()` on each handle. This blocks the execution
        until the operation associated with the handle is finished, ensuring that
        all communication is completed before moving on.

        Args:
            step (int): The current global step.
            blocking (bool): If True, waits for all operations to complete before returning.

        Returns:
            Dict[str, Any]: The input dictionary received from the inbound executor.
        """
        logger.info(f"step:{step} communication_channel {self.name}")

        output_dict, handles = self.send_op(
            channel_name=self.name,
            outbound_mesh=self.outbound_mesh,
            inbound_mesh=self.inbound_mesh,
            outbound_executor=self.outbound_executor,
            inbound_executor=self.inbound_executor,
            outbound_executor_pipe=self.outbound_executor_pipe,
            inbound_executor_pipe=self.inbound_executor_pipe,
            model_state_dict_key_to_shape=self._model_state_dict_key_to_shape,
        )
        for handle in handles:
            fetch_shard(handle).result()
        handles = []

        input_dict, handles = self.recv_op(
            channel_name=self.name,
            outbound_mesh=self.outbound_mesh,
            inbound_mesh=self.inbound_mesh,
            outbound_executor=self.outbound_executor,
            inbound_executor=self.inbound_executor,
            outbound_executor_pipe=self.outbound_executor_pipe,
            inbound_executor_pipe=self.inbound_executor_pipe,
            output_dict=output_dict,
        )
        if blocking:
            self.wait_for_complete()
        return input_dict
