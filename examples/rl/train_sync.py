# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-unsafe
"""A simple semi-sync RL workflow using Monarch.

Implements "off-by-1" sync, where the learner is trained on a batch of data
one step behind the actors.

To run on a single GPU host:

$ buck2 run @//mode/opt fbcode//monarch/examples/rl:train_sync

"""

import logging
import pprint
from typing import List

import torch

import tyro
from monarch import fetch_shard, inspect
from monarch.common.device_mesh import DeviceMesh
from monarch_supervisor.logging import initialize_logging
from rl.common_remote import (
    create_mesh,
    get_latest_time,
    RemoteGenerator,
    RemoteLearner,
    set_global_config,
    timer_start,
    timer_stop,
)
from rl.config import Config
from rl.data import Trajectory

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

_TIMER_KEY_GENERATE = "generate"
_TIMER_KEY_GENERATE_UPDATE = "generator_weight_update"
_TIMER_KEY_LEARNER_STEP = "learner_step"


def run(
    learner_mesh: DeviceMesh, generator_meshes: List[DeviceMesh], config: Config
) -> None:
    logging.info("Creating learner")
    with learner_mesh.activate():
        learner = RemoteLearner(config)

    logging.info("Creating generators")

    generators = []
    for mesh in generator_meshes:
        with mesh.activate():
            generator = RemoteGenerator(config)
        generators.append(generator)

    def update_generators() -> None:
        logger.debug("Updating generators.")
        # TODO - replace w/ Actor<>Actor API
        with learner_mesh.activate():
            timer_start(_TIMER_KEY_GENERATE_UPDATE)
            state_dict = learner.get_state_dict()
            for generator, mesh in zip(generators, generator_meshes):
                transferred_state_dict = {
                    k: v.to("cuda").to_mesh(mesh) for k, v in state_dict.items()
                }
                with mesh.activate():
                    generator.load_state_dict(transferred_state_dict)
            timer_stop(_TIMER_KEY_GENERATE_UPDATE)

    def queue_n_rollouts(n: int, response_length: int) -> List[Trajectory]:
        """Queues N trajectories.

        TODOS:
            - Proper load balancing
            - Add an inference batch size, this only assumes = 1

        Returns:
            A list of trajectories, each of length n.
        """
        logger.debug("Queuing rollouts.")
        trajectories = []

        for i in range(n):
            g_idx = i % config.num_generators
            mesh = generator_meshes[g_idx]
            generator = generators[g_idx]

            with mesh.activate():
                # TODO - replace this with a data loader
                prompt = torch.randn(
                    (config.prompt_length, config.input_shape),
                    device="cuda",
                    dtype=torch.float32,
                )
                timer_start(_TIMER_KEY_GENERATE)
                trajectory = generator.generate(prompt)
                timer_stop(_TIMER_KEY_GENERATE)
            trajectories.append(trajectory)

        logger.debug("Done queuing rollouts.")
        return trajectories

    update_generators()

    # Seed the replay buffer with initial data from policy 0.
    trajectories = queue_n_rollouts(
        n=config.batch_size, response_length=config.response_length
    )

    for step in range(config.num_steps):
        # Queue the next N rollouts before the trainer starts
        next_trajectories = queue_n_rollouts(
            n=config.batch_size, response_length=config.response_length
        )

        logger.debug("Transferring trajectories to learner")

        t = []
        for trajectory in trajectories:
            query_and_response, response, reward = trajectory
            t.append(
                (
                    # pyre-fixme[16]: `Tensor` has no attribute `to_mesh`.
                    query_and_response.to("cuda").to_mesh(learner_mesh),
                    response.to("cuda").to_mesh(learner_mesh),
                    reward.to("cuda").to_mesh(learner_mesh),
                )
            )

        trajectories = t

        # Train step
        with learner_mesh.activate():
            timer_start(_TIMER_KEY_LEARNER_STEP)
            loss, rewards = learner.step(trajectories)
            timer_stop(_TIMER_KEY_LEARNER_STEP)

        if step != 0 and step % config.log_n_steps == 0:
            with learner_mesh.activate():
                local_reward = fetch_shard(rewards.mean().to("cuda")).result()
                local_loss = fetch_shard(loss).result()
                local_learner_step_time = inspect(
                    get_latest_time(_TIMER_KEY_LEARNER_STEP)
                )
                local_generator_update_time = inspect(
                    get_latest_time(_TIMER_KEY_GENERATE_UPDATE)
                )
            with generator_meshes[0].activate():
                local_generate_rollout_time = inspect(
                    get_latest_time(_TIMER_KEY_GENERATE)
                )

            logging.info(
                "[step %d], policy_loss: %.2f, avg_reward: %.2f, learner_step_ms: %.2f, generator_rollout_ms: %.2f, generator_update_ms: %.2f",
                step,
                local_loss,
                local_reward,
                local_learner_step_time,
                local_generate_rollout_time,
                local_generator_update_time,
            )

        update_generators()
        trajectories = next_trajectories


# --- Main setup logic
def main(config: Config) -> None:
    pp = pprint.PrettyPrinter(indent=4)

    torch.manual_seed(config.seed)
    torch.set_default_device("cuda")

    logging.info("Initializing with args: %s", pp.pformat(vars(config)))
    set_global_config(config)

    # 1 GPU per generator, 1 GPU for learner
    num_gpus = config.num_generators + 1
    assert num_gpus <= 8, "Assumes only single host for now."

    # Mesh creation
    logging.info("Creating mesh with 1 host and %d GPUs per host", num_gpus)

    with create_mesh(
        mesh_type=config.mesh_type, num_hosts=1, gpus_per_host=num_gpus
    ) as world_mesh:
        world_mesh = world_mesh.flatten("gpu")

        logging.info("Dividing meshes...")
        learner_mesh = world_mesh(gpu=0)
        generator_meshes = [
            world_mesh(gpu=i) for i in range(1, config.num_generators + 1)
        ]

        try:
            run(
                learner_mesh=learner_mesh,
                generator_meshes=generator_meshes,
                config=config,
            )
        finally:
            logging.info("Completed training.")
            world_mesh.exit()


if __name__ == "__main__":
    initialize_logging()
    logger.setLevel(logging.INFO)

    config = tyro.cli(Config)
    main(config)
