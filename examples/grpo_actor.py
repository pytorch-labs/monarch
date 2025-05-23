# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from monarch.proc_mesh import local_proc_mesh
from monarch.rdma import RDMABuffer
from monarch.service import Actor, endpoint, Service

"""
This example demonstrates an online reinforcement learning (RL) training loop
using the Monarch actor framework. It defines three actor roles:
1. RewardModel: A small MLP that scores state-action pairs to produce rewards.
2. Generator: Copies policy weights from the Learner via RDMA buffers, generates action
   sequences using a categorical distribution, and sends them to the Learner asynchronously.
3. Learner: Implements a PPO-like update, consuming generated actions from Generators,
   computing standardized advantages, updating its policy network, and then broadcasting
   the new weights back to Generators.

The example uses local_proc_mesh to spawn actors on separate GPU meshes, asyncio queues
for inter-actor communication, and RDMABuffer for efficient weight synchronization.
It illustrates key concepts: distributed actors, asynchronous endpoints, RDMA-based
weight sharing, and a simplified PPO training loop across multiple processes/GPUs.
"""


@dataclass
class GeneratorOutput:
    actions: torch.Tensor  # [G] (LongTensor)
    logps: torch.Tensor  # [G] (FloatTensor)


@dataclass
class TrainingBatch:
    state: torch.Tensor  # [STATE_DIM]
    actions: torch.Tensor  # [G]
    old_logps: torch.Tensor  # [G]
    rewards: torch.Tensor  # [G]


G = 8  # group size
STATE_DIM = 4
ACTION_DIM = 4  # vocab size


@dataclass
class GenerationData:
    state: torch.Tensor
    generator_output: GeneratorOutput


class Learner(Actor):
    def __init__(self, reward_model):
        # main policy network
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        ).to("cuda")
        # frozen reference policy
        self.ref_model = nn.Sequential(
            nn.Linear(STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        ).to("cuda")
        self.update_reference_policy()
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-5)
        self.eps = 0.2  # PPO clipping epsilon
        self.generation_queue: asyncio.Queue[GenerationData] = asyncio.Queue()
        self.reward_model = reward_model

    @endpoint
    async def init_generators(self, generators) -> None:
        """Store generator service reference for updates."""
        self.generators = generators
        self.num_generators = len(generators._ndslice)

    def update_reference_policy(self) -> None:
        """Copy current policy weights into the frozen reference model."""
        self.ref_model.load_state_dict(self.model.state_dict())
        for p in self.ref_model.parameters():
            p.requires_grad = False

    @endpoint
    async def weights_handle(self) -> Dict[str, RDMABuffer]:
        """Expose RDMA buffers for the model's flattened weights."""
        return {
            k: RDMABuffer(v.view(torch.uint8).flatten())
            for k, v in self.model.state_dict().items()
        }

    @endpoint
    async def put_generation(
        self, state: torch.Tensor, gen_output: GeneratorOutput
    ) -> None:
        """Receive generated samples and enqueue them for training."""
        await self.generation_queue.put(GenerationData(state, gen_output))

    async def _collect_generations(self) -> List[GenerationData]:
        """Gather one batch of generation data from all generators."""
        return [await self.generation_queue.get() for _ in range(self.num_generators)]

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards to compute advantages."""
        mean, std = rewards.mean(), rewards.std(unbiased=False) + 1e-8
        return (rewards - mean) / std

    def _apply_policy_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logps: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clipped surrogate loss and update policy via gradient step."""
        dist = torch.distributions.Categorical(logits=self.model(states))
        new_logps = dist.log_prob(actions)
        ratio = (new_logps - old_logps).exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
        loss = -torch.min(unclipped, clipped).mean()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        return loss.detach()

    @endpoint
    async def step(self, step_num: int, state: torch.Tensor) -> torch.Tensor:
        """
        Perform a training iteration:
        1) Refresh reference policy
        2) Collect generations
        3) Score with reward model
        4) Normalize and apply PPO update
        5) Notify generators of new weights
        """
        # 1) refresh reference
        self.update_reference_policy()
        # 2) collect
        gens = await self._collect_generations()
        # 3) predict rewards
        reward_preds = await asyncio.gather(
            *[
                self.reward_model.predict.call_one(g.state, g.generator_output.actions)
                for g in gens
            ]
        )
        # batch tensors
        raw_states = torch.stack([g.state for g in gens])  # [N, STATE_DIM]
        actions = torch.cat([g.generator_output.actions for g in gens])  # [N * G]
        old_logps = torch.cat([g.generator_output.logps for g in gens])
        rewards = torch.cat(reward_preds)
        # expand states to match actions
        states = raw_states.repeat_interleave(G, dim=0).to("cuda")  # [N * G, STATE_DIM]
        actions, old_logps, rewards = [
            t.to("cuda") for t in (actions, old_logps, rewards)
        ]
        # 4) update
        advantages = self._compute_advantages(rewards)
        loss = self._apply_policy_update(states, actions, old_logps, advantages)
        # 5) notify generators
        await self.generators.update.call()
        return loss


class GeneratorState:
    READY_TO_GENERATE = "READY_TO_GENERATE"
    READY_TO_UPDATE = "READY_TO_UPDATE"


class Generator(Actor):
    def __init__(self, weight_buffers: Dict[str, RDMABuffer], learner):
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        ).to("cuda")
        self.weight_buffers = weight_buffers
        self.learner = learner
        self.state = GeneratorState.READY_TO_GENERATE
        self.condition = asyncio.Condition()

    @endpoint
    async def update(self) -> None:
        """Pull latest weights from learner via RDMA buffers."""
        async with self.condition:
            if self.state != GeneratorState.READY_TO_UPDATE:
                raise RuntimeError("Invalid state; expected READY_TO_UPDATE")
            sd = self.model.state_dict()
            for name, buf in self.weight_buffers.items():
                await buf.read_into(sd[name].view(torch.uint8).flatten())
            self.model.load_state_dict(sd)
            self.state = GeneratorState.READY_TO_GENERATE
            self.condition.notify_all()

    @endpoint
    async def generate(self, step: int, state: torch.Tensor) -> None:
        """Generate actions and send them to the Learner."""
        async with self.condition:
            if self.state != GeneratorState.READY_TO_GENERATE:
                raise RuntimeError("Invalid state; expected READY_TO_GENERATE")
            x = state.to("cuda").unsqueeze(0).repeat(G, 1)
            dist = torch.distributions.Categorical(logits=self.model(x))
            acts = dist.sample()
            logps = dist.log_prob(acts)
            gen_out = GeneratorOutput(actions=acts.cpu(), logps=logps.cpu())
            await self.learner.put_generation.call(state, gen_out)
            self.state = GeneratorState.READY_TO_UPDATE
            self.condition.notify_all()


class RewardModel(Actor):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        ).to("cuda")
        self.optim = optim.Adam(self.net.parameters(), lr=1e-3)

    @endpoint
    async def predict(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute reward predictions for state-action pairs."""
        s = state.to("cuda").unsqueeze(0).repeat(G, 1)
        a = actions.to("cuda").float().unsqueeze(-1)
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


async def main():
    learner_mesh = await local_proc_mesh(gpus=1)
    gen_mesh = await local_proc_mesh(gpus=2)

    reward_model = await learner_mesh.spawn("reward", RewardModel)
    learner = await learner_mesh.spawn("learner", Learner, reward_model)

    wb = await learner.weights_handle.call_one()
    generators = await gen_mesh.spawn("generator", Generator, wb, learner)

    await learner.init_generators.call(generators)
    for step in range(5):
        state = torch.randn(STATE_DIM)
        _, loss = await asyncio.gather(
            generators.generate.call(step, state),
            learner.step.call_one(step, state),
        )
        print(f"[Step {step:02d}] loss={loss:.3f}")
    print("✅ done")


if __name__ == "__main__":
    asyncio.run(main())
