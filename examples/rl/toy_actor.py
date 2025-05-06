import asyncio

from typing import Dict, List, Tuple

import torch
from monarch.proc_mesh import proc_mesh
from monarch.rdma import RDMABuffer
from monarch.service import Actor, endpoint


class Learner(Actor):
    def __init__(self):
        self.model = torch.nn.Linear(4, 4, bias=False, device="cuda")
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            eps=1e-5,
        )

    @endpoint
    async def weights_handle(self) -> Dict[str, RDMABuffer]:
        ret = {}
        for k, v in self.model.state_dict().items():
            ret[k] = RDMABuffer(v.view(torch.uint8).flatten())
        return ret

    @endpoint
    async def step(
        self,
        inputs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # list(tensor, tensor) => list(tensor), list(tensor)
        inputs, rewards = zip(*inputs)

        # list(tensor) => tensor
        tensor = torch.stack(inputs)
        rewards = torch.stack(rewards)

        logits = self.model(tensor)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.mean(rewards.detach().squeeze() * log_probs.sum(dim=[1, 2]))
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        print("[learner] weights: ", self.model.state_dict())
        return loss, rewards.sum()


class Generator(Actor):
    def __init__(self, weight_buffer: RDMABuffer):
        self.model = torch.nn.Linear(4, 4, bias=False, device="cuda")
        self.weight_buffer = weight_buffer

    @endpoint
    async def update(self):
        print("original weights: ", self.model.state_dict())
        state_dict = self.model.state_dict()
        for k, _ in state_dict.items():
            byte_tensor = state_dict[k].view(torch.uint8).flatten()
            await self.weight_buffer[k].read_into(byte_tensor)
        print("new weights: ", self.model.state_dict())

    @endpoint
    async def generate(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(inputs)
        reward = torch.sum(logits)
        return logits, reward


async def main():
    num_generators = 2
    learner_mesh = await proc_mesh(gpus=1, env={})
    gen_mesh = await proc_mesh(gpus=num_generators, env={})

    learner = await learner_mesh.spawn("learner", Learner)
    weight_buffer = await learner.weights_handle().call()
    generators = await gen_mesh.spawn("generator", Generator, weight_buffer)

    generation_stream = generators.generate(torch.randn(4, 4, device="cuda")).stream()
    for step in range(3):
        generations = [gen async for gen in generation_stream]
        loss, rewards = await learner.step(generations).call()
        print(f"step: {step}, loss: {loss}, rewards: {rewards}")
        generation_stream = generators.generate(
            torch.randn(4, 4, device="cuda")
        ).stream()
        await generators.update().broadcast_and_wait()

    print("done")


asyncio.run(main())
