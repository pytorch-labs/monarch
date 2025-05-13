"""A skeleton to demonstrate the to-be-named, async distributed RL infra.

The key ideas of the RL infrastructure are shared at:
1) https://docs.google.com/document/d/1wBzYSZwx1UHZBVpXrDwCpAgQrx32g_bKovPUNMOvVYc/edit?tab=t.0
2) P1786253875 - daju@'s paste demonstrating a few of the ideas.

suo@ created https://github.com/fairinternal/xlformers/pull/20165, which
implements the same thing but is actually tied into XLFormers.

This is a minimal example to enable faster iterate and communicate the
core ideas clearly.

Run this with:

buck2 run @//mode/opt //monarch/examples/rl:async_pipelined

"""

import asyncio

import logging
import os
import traceback
from collections import defaultdict

import monarch
from monarch.proc_mesh import proc_mesh
from monarch.service import endpoint, MonarchContext

# TODO - figure out logging between actors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO - make this configurable..
GPUS_PER_TRAINER = 1
GPUS_PER_GENERATOR = 1
NUM_GENERATORS = 1


class GlobalQueue(monarch.service.Actor):
    """A global K/V queue that can be used by multiple actors to communicate.

    This implements a simple pub/sub pattern where:
    - Each topic is a separate asyncio.Queue
    - Actors can publish messages to specific topics
    - Actors can subscribe to and consume messages from specific topics
    """

    def __init__(self):
        self.queues = defaultdict(asyncio.Queue)

    @endpoint
    async def put(self, topic, obj):
        await self.queues[topic].put(obj)

    @endpoint
    async def get(self, topic):
        return await self.queues[topic].get()


def pipeline(*, input=None, output=None, yield_output=None, max_queue_length=None):
    """Decorator that marks a function as part of the data pipeline.

    This decorator configures how data flows through the system by specifying:
    - input: Topic to read input data from
    - output: Topic to send output data to (single result)
    - yield_output: Topic to send output data to (streaming results)
    - max_queue_length: Maximum number of items in the queue

    The decorated function becomes an endpoint that can be called remotely,
    and is automatically connected to the specified input/output topics.
    """
    # TODO - actually satisfy max_queue_length.

    def decorator(func):
        endpoint_wrapped = endpoint(func)
        endpoint_wrapped._input_pipe = input  # pyre-ignore
        endpoint_wrapped._output_pipe = output  # pyre-ignore
        endpoint_wrapped._yield_output_pipe = yield_output  # pyre-ignore
        endpoint_wrapped._is_pipeline = True  # pyre-ignore
        return endpoint_wrapped

    return decorator


class Server(monarch.service.Actor):
    """Handles the orchestration of actors and messages within the distributed system.

    The Server is responsible for:
    1. Managing actors that have already been deployed to process meshes
    2. Setting up communication channels between actors using the global queue
    3. Running the pipeline loops that process data through the system
    4. Identifying and executing functions marked with the @pipeline decorator
    5. Routing messages between input and output topics based on pipeline configurations
    """

    _queue = None
    _pipeline_loops = None
    _actor = None

    @endpoint
    async def _deploy(self, queue: GlobalQueue, actor):
        """Deploy a server with access to the global queue."""
        self._queue = queue
        self._pipeline_loops = []
        self._actor = actor
        print(f"Deploying server for {self._actor._class.__name__}")

    def __repr__(self) -> str:
        return f"Server::{self._actor._class.__name__}"

    @endpoint
    async def _run(self):
        """Start all pipeline loops for this actor.

        This method:
        1. Finds all methods marked with @pipeline decorator
        2. Creates async tasks that continuously process data through these methods
        3. Handles input/output routing based on the pipeline configuration
        """
        print(f"Running {self}")
        for attr_name in dir(self._actor._class):
            attr_value = getattr(self._actor._class, attr_name, None)
            if attr_value is None:
                continue

            # Check if the attribute is a function marked with the @pipeline decorator.
            if not getattr(attr_value, "_is_pipeline", False):
                continue

            method_name = attr_value._method.__name__

            func = attr_value
            input = getattr(func, "_input_pipe", None)
            output = getattr(func, "_output_pipe", None)
            yield_output = getattr(func, "_yield_output_pipe", None)

            castable = getattr(self._actor, method_name)
            assert castable is not None

            # capture these explicitly to avoid late binding in closures.
            async def runner(
                input=input,
                output=output,
                yield_output=yield_output,
                castable=castable,
                method_name=method_name,
            ):
                """Inner function that runs a continuous pipeline loop for a specific method.

                This continuously:
                1. Gets input from the specified queue topic (if any)
                2. Calls the method with that input
                3. Routes the output to the specified output topic(s)
                """
                print(
                    f"starting pipeline for {self}::{method_name} (input:{input}, output:{output}, yield_output:{yield_output})"
                )
                try:
                    if input is not None:
                        while True:
                            msg = await self._queue.get.call(input)
                            stream = castable.stream(msg)
                            result = [res async for res in stream]
                            if yield_output:
                                if result is not None:
                                    for item in result:
                                        await self._queue.put.call(yield_output, item)
                            elif output and result is not None:
                                await self._queue.put.call(output, result)
                    else:
                        while True:
                            stream = castable.stream()
                            result = [res async for res in stream]
                            if yield_output:
                                if result is not None:
                                    for item in result:
                                        await self._queue.put.call(yield_output, item)
                            elif output and result is not None:
                                await self._queue.put.call(output, result)
                            await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Exception in pipeline loop: {e}")
                    traceback.print_exc()

            self._pipeline_loops.append(asyncio.create_task(runner()))


async def deploy(*mesh_and_groups):
    """Deploy the entire actor system across process meshes.

    This function:
    1. Creates a global queue for inter-actor communication
    2. Spawns servers for each actor group on their respective process meshes
    3. Deploys the queue to each actor
    4. Starts all pipeline loops

    Args:
        *mesh_and_groups: Tuples of (process_mesh, actor_group)

    Returns:
        The global queue instance
    """
    queue_proc_mesh = await proc_mesh(gpus=1, env={})
    queue = await queue_proc_mesh.spawn("queue", GlobalQueue)

    groups = []
    servers = []
    for mesh, group in mesh_and_groups:
        server = await mesh.spawn(f"{group._class.__name__}", Server)
        servers.append(server)
        groups.append(group)

    tasks = []
    for group, server in zip(groups, servers):
        tasks.append(server._deploy.call(queue=queue, actor=group))

    await asyncio.gather(*tasks)
    tasks.clear()

    # Deploy queue to the servers which use it.
    for group in groups:
        if not hasattr(group, "deploy_queue"):
            print(
                f"Group {group._class.__name__} does not have a deploy_queue method, skipping. If this is not expected, ensure that the group extends BaseActor."
            )
        else:
            print("Deploying queue to group {}".format(group._class.__name__))
            tasks.append(group.deploy_queue.call(queue=queue))

    await asyncio.gather(*tasks)
    tasks.clear()

    for group, server in zip(groups, servers):
        print(f"Starting pipeline loops for {group._class.__name__}")
        tasks.append(server._run.call())

    await asyncio.gather(*tasks)
    return queue


# -------------
# Dummy business-logic functions and classes
# -------------
def compute_reward(prompts):
    """A dummy reward function: simply wrap the prompt text."""
    if isinstance(prompts, list):
        return [f"reward({p})" for p in prompts]
    else:
        return f"reward({prompts})"


class BaseActor(monarch.service.Actor):
    """Base class for all actors in the system.

    All actor types (Trainer, Generator, etc.) managed by Server should
    inherit from this class to ensure they have the necessary methods for the Server
    to work with them.
    """

    @endpoint
    async def deploy_queue(self, queue):
        self._queue = queue


class DummyModel:
    def __call__(self, batch):
        print(f"Trainer's model processing batch: {batch}")

        class DummyLoss:
            def back(self):
                print("Trainer's loss backpropagation")

        return DummyLoss()

    def generate_next_token(self, seqs):
        """
        Modified to always return "<end_of_generation>" so that sequences finish.
        """
        tokens = ["<end_of_generation>" for _ in seqs]

        return tokens

    def update(self, weights):
        print(f"Model updated with weights: {weights}")


class DummyOptimizer:
    def zero_grad(self):
        print("Trainer's optimizer zero grad")

    def step(self):
        print("Trainer's optimizer step")


class Trainer(BaseActor):
    def __init__(self):
        ctx = MonarchContext.get()
        self.rank = ctx.rank
        os.environ["RANK"] = str(ctx.rank)
        os.environ["LOCAL_RANK"] = str(ctx.rank % GPUS_PER_TRAINER)
        os.environ["WORLD_SIZE"] = str(ctx.shape.len)
        self.model = DummyModel()
        self.optimizer = DummyOptimizer()
        self.step = 0

    @pipeline(input="replaybuffer2trainer")
    async def run(self, batch):
        print(f"[{self}::step-{self.step}] Received batch: {batch}")
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss.back()
        self.optimizer.step()
        # Simulate sending updated weights and new prompts back to the generator.
        if self.rank == 0:
            await self._queue.put.call(
                "trainer2generator_weights", f"new_weights({self.step})"
            )

        num_prompts_to_insert = NUM_GENERATORS
        new_prompts = [
            f"step_{self.step}_prompt_{i}" for i in range(num_prompts_to_insert)
        ]
        if self.rank == 0:
            await self._queue.put.call("trainer2generator", new_prompts)
        self.step += 1

    def __repr__(self) -> str:
        return f"Trainer(shard{self.rank})"


class Generator(BaseActor):
    def __init__(self, global_rank: int):
        self.prompts = []
        self.seqs = []
        self.batchsize = 3
        ctx = MonarchContext.get()
        # local_rank here refers to a parallelism rank within a distributed group
        self.local_rank = ctx.rank
        # global rank refers to the index out of the total number of generators
        self.global_rank = global_rank
        os.environ["RANK"] = str(ctx.rank)
        os.environ["LOCAL_RANK"] = str(ctx.rank % GPUS_PER_GENERATOR)
        os.environ["WORLD_SIZE"] = str(ctx.shape.len)
        self.model = DummyModel()

    @pipeline(input="trainer2generator")
    async def get_prompts(self, prompts):
        print(f"[{self}] Received prompts: {prompts}")
        self.prompts.extend(prompts)

    @pipeline(input="trainer2generator_weights")
    async def get_weights(self, received_weights):
        print(f"[{self}] Received weights: {received_weights}")
        self.model.update(received_weights)

    @pipeline(yield_output="generator2reward")
    async def gen(self):
        # Add new prompts to the active sequences if needed.
        if len(self.seqs) < self.batchsize and self.prompts:
            self.seqs.extend(self.prompts)
            self.prompts = []
        if not self.seqs:
            await asyncio.sleep(1)
            return
        # Always produce an end token for debugging.
        tokens = self.model.generate_next_token(self.seqs)
        new_seqs = []
        outputs = []
        for seq, token in zip(self.seqs, tokens):
            new_seq = seq + " " + token
            if token == "<end_of_generation>":
                outputs.append(new_seq)
            else:
                new_seqs.append(new_seq)
        self.seqs = new_seqs
        return [f"({self.global_rank}.{self.local_rank})"] + outputs

    def __repr__(self) -> str:
        return f"Generator-{self.global_rank}(shard-{self.local_rank})"


class ReplayBuffer(BaseActor):
    """A simple replay buffer that stores and samples prompts with rewards.

    The ReplayBuffer is responsible for:
    1. Receiving prompts with associated rewards from the Reward actor
    2. Storing these prompts in an internal buffer
    3. Sampling batches of prompts to send to the Trainer actor
    4. Managing the size of the buffer to ensure efficient sampling

    Attributes:
        replaybuffer (list): A list to store prompts with rewards.
        sampler (function): A function to sample a batch from the replay buffer.
    """

    def __init__(self):
        self.replaybuffer = []
        self.sampler = lambda rb: rb[:2] if len(rb) >= 2 else None

    @pipeline(input="reward2replaybuffer")
    async def get(self, prompts_with_rewards):
        print(f"[{self}] received:", prompts_with_rewards)
        self.replaybuffer.append(prompts_with_rewards)

    @pipeline(output="replaybuffer2trainer", max_queue_length=3)
    async def prefetch(self):
        if len(self.replaybuffer) >= 2:
            batch = self.sampler(self.replaybuffer)
            self.replaybuffer = self.replaybuffer[2:]
            print(f"[{self}] prefetching batch:", batch)
            return batch
        else:
            await asyncio.sleep(1)
            return None

    def __repr__(self) -> str:
        return "ReplayBuffer"


class Reward(BaseActor):
    # Today we do not support standalone functions. But we will soon; @zdevito
    # is working on it
    @pipeline(input="generator2reward", output="reward2replaybuffer")
    async def reward_func(self, prompts):
        result = compute_reward(prompts)
        print(f"[{self}] Reward function computed:", result)
        return result

    def __repr__(self) -> str:
        return "Reward"


async def main(max_gpus: int = 8) -> None:
    assert (
        GPUS_PER_GENERATOR * NUM_GENERATORS + GPUS_PER_TRAINER + 2 <= max_gpus
    )  # 1 GPU for replay buffer, 1 GPU for queue

    # Env needed for torch distributed
    master_port = int(os.environ.get("BASE_MASTER_PORT", "12345"))
    master_addr = "localhost"
    # Spawn a process for the trainer
    trainer_proc_mesh = await proc_mesh(
        gpus=GPUS_PER_TRAINER,
        env={
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "DUMP_DIR": "/tmp/monarch_xlformers_dump",
        },
    )
    # Spawn the trainer class into that process
    trainer = await trainer_proc_mesh.spawn("trainer", Trainer)

    replay_buffer_proc_mesh = await proc_mesh(gpus=1, env={})
    replay_buffer = await replay_buffer_proc_mesh.spawn("replay_buffer", ReplayBuffer)

    # as an example: here we re-use the replay buffer process, so the reward function
    # will be in the same process as the replay buffer
    reward_server = await replay_buffer_proc_mesh.spawn("reward", Reward)

    generators = []

    for i in range(NUM_GENERATORS):
        generator_proc_mesh = await proc_mesh(
            gpus=GPUS_PER_GENERATOR,
            env={
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port + i + 1),
                "DUMP_DIR": "/tmp/monarch_xlformers_dump",
            },
        )
        generator = await generator_proc_mesh.spawn(
            "generator", Generator, global_rank=i
        )
        generators.append((generator_proc_mesh, generator))

    queue = await deploy(
        (trainer_proc_mesh, trainer),
        *generators,
        (replay_buffer_proc_mesh, replay_buffer),
        (replay_buffer_proc_mesh, reward_server),
    )

    # the demo code put some global channels that were declared as pipeline inputs.
    # Make them available to the trainer directly.
    await queue.put.call("trainer2generator", ["Hello", "World"])

    # This sleep controls how long the workload runs for.
    # TODO - wire in mechanisms to stop the workload based on a number of steps completed.
    await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
