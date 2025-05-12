import logging

from post_training.lib.communication.communication_channel import (
    CommunicationChannel,
    CommunicationType,
    WeightsCommunicationChannel,
)

from post_training.lib.executor_context import ExecutorContext
from post_training.lib.executor_controller import ExecutorController

from .config import GRPOConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(grpo_params=None) -> None:
    grpo_params = grpo_params or GRPOConfig()
    grpo_trainer_configs = grpo_params.trainer_config
    grpo_generator_configs = grpo_params.generator_config
    logger.info(f"GRPO parameters: {grpo_params=}")

    executor_context = ExecutorContext(
        executor_configs_group=[grpo_trainer_configs, grpo_generator_configs],
        params=grpo_params,
    )
    grpo_trainer = executor_context.name_to_executor["grpo_trainer"]
    grpo_generator = executor_context.name_to_executor["grpo_generator"]

    if grpo_params.multi_mesh:
        weights_update_channel_commu_type = CommunicationType.INTER_MESH_WEIGHT_P2P
        generation_channel_commu_type = CommunicationType.INTER_MESH_P2P
    else:
        weights_update_channel_commu_type = CommunicationType.SUB_MESH_WEIGHT_P2P
        generation_channel_commu_type = CommunicationType.SUB_MESH_P2P

    executor_controller = ExecutorController(
        executor_context=executor_context,
        checkpoint_executor=grpo_trainer,
        communication_channels=[
            WeightsCommunicationChannel(
                name="weights_update_channel",
                outbound_executor_name=grpo_trainer_configs.name,
                inbound_executor_name=grpo_generator_configs.name,
                communication_type=weights_update_channel_commu_type,
                executor_context=executor_context,
                _model_state_dict_key_to_shape=executor_context.name_to_state_dict_key_to_shape[
                    grpo_trainer_configs.name
                ],
            ),
            CommunicationChannel(
                name="generation_channel",
                outbound_executor_name=grpo_generator_configs.name,
                inbound_executor_name=grpo_trainer_configs.name,
                communication_type=generation_channel_commu_type,
                executor_context=executor_context,
            ),
        ],
        max_steps=grpo_params.steps,
        nsteps_per_checkpoint=grpo_params.save_freq,
    )
    logger.info(f"Starting GRPO training with {grpo_params}.")
    executor_controller.run()
    executor_context.shutdown()


if __name__ == "__main__":
    main()
