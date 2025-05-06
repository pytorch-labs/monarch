# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-unsafe
"""A simple async RL workflow using Monarch.

To run on a single GPU host:

~/fbsource/fbcode $ buck run @mode/opt monarch/examples/rl:train_async

"""

import contextlib
import itertools
import logging
import os
import pprint
import queue
import random

import signal
import string

import threading
import time
import traceback

from datetime import datetime

import aix
import aix.logging
import monarch

import torch

import tyro
from monarch.common.device_mesh import DeviceMesh
from monarch.timer.execution_timer import ExecutionTimer
from monarch_supervisor.logging import initialize_logging
from rl.common_remote import (
    create_mesh,
    get_latest_time,
    RemoteGenerator,
    RemoteLearner,
    set_global_config,
    set_worker_random_seed,
    timer_start,
    timer_stop,
)
from rl.config import Config

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

_TIMER_KEY_GENERATE = "generate"
_TIMER_KEY_LEARNER_STEP = "learner_step"
_TIMER_KEY_L_PS = "l_ps_weight_transfer"
_TIMER_KEY_PS_G = "ps_g_weight_transfer"


# ---- data structures
class SafeThread(threading.Thread):
    """Thread that kills the program if it fails."""

    def run(self):
        try:
            super().run()
        except Exception as e:
            logger.error(f"Thread {self.name} failed with {e}")
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGKILL)


class ParameterServer:
    # TODO - ParameterServer requires the dispatch lock to be held by the Driver.
    # Of course, this can be prone to error.
    # Once we have threadsafe dispatches, we can remove this requirement.
    def __init__(self, mesh: DeviceMesh, dispatch_lock: threading.Lock):
        self.mesh = mesh
        self.version = -1
        self.model_state = None
        self.dispatch_lock = dispatch_lock
        logging.info("[parameter_server] Initialized with mesh %s.", mesh)

    def push_weights(
        self,
        learner: RemoteLearner,
        learner_mesh: DeviceMesh,
    ):
        """Updates the parameter server with the latest weights from the learner."""
        logger.debug(
            f"[parameter_server] Updating internal weights to version {self.version + 1}"
        )
        with self.dispatch_lock:
            with learner_mesh.activate():
                state_dict = learner.get_state_dict()
                with self.mesh.activate():
                    timer_start(_TIMER_KEY_L_PS)
                    state_dict = {
                        # pyre-fixme[16]: `Tensor` has no attribute `to_mesh`.
                        k: v.to("cuda").to_mesh(self.mesh)
                        for k, v in state_dict.items()
                    }
                    self.model_state = state_dict
                    self.version += 1
                    logger.debug(
                        "[parameter_server] Successfully updated weights to version %d.",
                        self.version,
                    )
                    timer_stop(_TIMER_KEY_L_PS)
        return self.version

    def get_current_version(self):
        """Returns the current policy version."""
        return self.version

    def request_weights(self, target_mesh: DeviceMesh) -> tuple[dict, int]:
        """Requests the latest weights from the parameter server."""
        while self.model_state is None:
            logger.debug("[parameter_server] No weights available yet, waiting...")
            time.sleep(5)

        logger.debug("[parameter_server] Transferring weights to mesh %s", target_mesh)
        with self.dispatch_lock:
            with self.mesh.activate():
                timer_start(_TIMER_KEY_PS_G, use_cpu=False)
                state_dict = {
                    k: v.clone().to_mesh(target_mesh)
                    for k, v in self.model_state.items()
                }
                logger.debug(
                    "[parameter_server] Successfully transmitted weights (version %d)",
                    self.version,
                )
                timer_stop(_TIMER_KEY_PS_G, use_cpu=False)
        return state_dict, self.version


# ---- main training logic


class Driver:
    def __init__(self, world_mesh: DeviceMesh, config: Config):
        self.world_mesh = world_mesh
        self.config = config
        self.experience_buffer = queue.LifoQueue(maxsize=config.replay_buffer_size)

        # TODO - dispatch lock is necessary because the Monarch front-end is not threadsafe.
        # Remove once resolved.
        self._dispatch_lock = threading.Lock()
        self.stop_event = threading.Event()

        logger.info("Creating learner mesh")
        learner_mesh = world_mesh(gpu=0)
        self.learner_thread = SafeThread(
            target=self._learner, name="learner", args=(learner_mesh,)
        )

        logger.info("Creating parameter server mesh")
        pserver_mesh = world_mesh(gpu=1)
        self.parameter_server = ParameterServer(
            mesh=pserver_mesh, dispatch_lock=self._dispatch_lock
        )
        self.pserver_mesh = pserver_mesh

        self.generator_threads = []
        logger.info("Creating %d generator thread(s)", config.num_generators)

        for i in range(config.num_generators):
            logger.info("Creating generator mesh %d", i + 2)
            gmesh = world_mesh(gpu=i + 2)
            self.generator_threads.append(
                SafeThread(
                    target=self._generator,
                    name=f"generator_{i}",
                    args=(gmesh, i),
                )
            )

        self.all_threads = list(
            itertools.chain(self.generator_threads, [self.learner_thread])
        )
        for t in self.all_threads:
            t.start()

        logger.info("Started all threads.")

    def join(self):
        for t in self.all_threads:
            t.join()

    def _generator(self, mesh: DeviceMesh, idx: int):
        """Runs the generator loop."""
        logger.info("Starting generator thread %d with mesh %s...", idx, mesh)
        torch.set_default_device("cuda")

        version = None

        with self._dispatch_lock:
            logger.info("[generator %d] Creating policy...", idx)
            with mesh.activate():
                process_idx = mesh.process_idx()
                set_worker_random_seed(self.config.seed, process_idx)
                generator = RemoteGenerator(self.config)

        while True:
            if self.stop_event.is_set():
                logger.info("[generator %d] stopping...", idx)
                break

            if self.parameter_server.get_current_version() != version:
                logger.debug(
                    "[generator %d] Version mismatch, requesting weights from parameter server...",
                    idx,
                )
                state_dict, version = self.parameter_server.request_weights(mesh)
                with self._dispatch_lock:
                    with mesh.activate():
                        generator.load_state_dict(state_dict)
                logger.debug(
                    "[generator %d] Successfully loaded weights from version %d",
                    idx,
                    version,
                )

            with self._dispatch_lock:
                with mesh.activate():
                    with torch.no_grad():
                        prompt = torch.randn(
                            (self.config.prompt_length, self.config.input_shape),
                            device="cuda",
                            dtype=torch.float32,
                        )
                        timer_start(_TIMER_KEY_GENERATE)
                        trajectory = generator.generate(prompt)
                        timer_stop(_TIMER_KEY_GENERATE)
            try:
                self.experience_buffer.put((trajectory, version), timeout=3)
            except queue.Full:
                if self.stop_event.is_set():
                    break
            # compute reward, store experience
            if idx == 1:
                logger.debug("[generator 1] pushed experience")

    def _learner(self, mesh: DeviceMesh):
        """Runs the learner loop."""
        logger.info("Starting learner thread with mesh %s...", mesh)
        torch.set_default_device("cuda")
        learner_version = 0

        with self._dispatch_lock:
            logger.info("[learner] Creating policy...")
            with mesh.activate():
                process_idx = mesh.process_idx()
                set_worker_random_seed(self.config.seed, process_idx)
                learner = RemoteLearner(self.config)

        logger.info(
            "[learner] Created policy and optimizer. Pushing weights to parameter server..."
        )
        self.parameter_server.push_weights(learner, mesh)

        for step in range(self.config.num_steps):
            if self.config.enable_aix and step % 10 == 0:
                aix.set_run_progress(step, total_steps=self.config.num_steps)
            if self.stop_event.is_set():
                logger.info("[learner] stopping...")
                break
            trajectories, versions = [], []

            with ExecutionTimer.time("build_batch", use_cpu=True):
                while len(versions) < self.config.batch_size:
                    trajectory, version = self.experience_buffer.get(block=True)
                    logger.debug(
                        "[learner] Received experience from generator, transferring to mesh..."
                    )
                    with self._dispatch_lock:
                        query_and_response, response, reward = trajectory
                        trajectories.append(
                            (
                                query_and_response.to("cuda").to_mesh(mesh),
                                response.to("cuda").to_mesh(mesh),
                                reward.to("cuda").to_mesh(mesh),
                            )
                        )
                    versions.append(version)

            logger.debug("[learner] Built batch of size %d", len(versions))
            with self._dispatch_lock:
                with mesh.activate():
                    timer_start(_TIMER_KEY_LEARNER_STEP)
                    loss, rewards = learner.step(trajectories)
                    timer_stop(_TIMER_KEY_LEARNER_STEP)
                    avg_reward = rewards.mean().to("cuda")
                    local_avg_reward = monarch.fetch_shard(avg_reward).result()
                    local_loss = monarch.fetch_shard(loss).result()

                with self.pserver_mesh.activate():
                    local_l_ps_weight_transfer_ms = monarch.inspect(
                        get_latest_time(_TIMER_KEY_L_PS)
                    )
                    local_ps_g_weight_transfer_ms = monarch.inspect(
                        get_latest_time(_TIMER_KEY_PS_G)
                    )

            build_batch_ms = ExecutionTimer.get_latest_measurement("build_batch")
            average_staleness = learner_version - sum(versions) / len(versions)

            if step != 0 and step % self.config.log_n_steps == 0:
                logger.info(
                    "[learner] Completed step %d. loss: %.2f, "
                    "avg_reward: %.2f, build_batch_ms: %.2f, avg_staleness: %.2f, "
                    "learner_server_transfer_ms: %.4f, server_generator_transfer_ms: %.4f",
                    step,
                    local_loss,
                    local_avg_reward,
                    build_batch_ms,
                    average_staleness,
                    local_l_ps_weight_transfer_ms.item(),
                    local_ps_g_weight_transfer_ms.item(),
                )
            logger.debug("[learner] Pushing weights to parameter server...")
            if self.config.enable_aix:
                aix.logging.log(
                    {
                        "loss": local_loss,
                        "avg_reward": local_avg_reward,
                        "avg_staleness": average_staleness,
                        "build_batch_ms": build_batch_ms,
                        "learner_server_transfer_ms": local_l_ps_weight_transfer_ms.item(),
                        "server_generator_transfer_ms": local_ps_g_weight_transfer_ms.item(),
                    },
                    step=step,
                )
            self.parameter_server.push_weights(learner, mesh)
            learner_version += 1

        logging.info("Learner completed all steps. Stopping...")
        self.stop_event.set()


# --- Main setup logic
def main(config: Config) -> None:
    pp = pprint.PrettyPrinter(indent=4)
    set_global_config(config)

    torch.manual_seed(config.seed)
    torch.set_default_device("cuda")

    if config.enable_aix:
        exp_name = f"async_rl_{datetime.now().strftime('%Y%m%d%H%M%S')}_{''.join(random.choices(string.ascii_lowercase, k=5))}"
        aix.init_local_run(
            name=exp_name,
            force_recreate_run=True,
        )
        aix.set_run_metadata(vars(config))
        aix.add_stages(["Training"])

    logger.info("Initializing with args: %s", pp.pformat(vars(config)))

    # 1 GPU per generator, 1 GPU for learner, 1 GPU for parameter server
    num_gpus = config.num_generators + 2
    assert num_gpus <= 8, "Assumes only single host for now."

    # Mesh creation
    logger.info("Creating mesh with 1 host and %d GPUs per host", num_gpus)

    with create_mesh(
        mesh_type=config.mesh_type, num_hosts=1, gpus_per_host=num_gpus
    ) as world_mesh:
        world_mesh = world_mesh.flatten("gpu")

        logger.info("Setting the logging level for all workers to info...")
        logger.info("Creating driver...")

        try:
            driver = Driver(world_mesh=world_mesh, config=config)
            driver.join()
        finally:
            logger.info("All done, exiting...")
            world_mesh.exit()


if __name__ == "__main__":
    initialize_logging()
    logger.setLevel(logging.INFO)

    config = tyro.cli(Config)
    main(config)
