import asyncio

import torch
from monarch.service import endpoint, local_proc_mesh, RDMABuffer


class Counter:
    def __init__(self, v: int):
        self.v = v

    @endpoint
    async def incr(self):
        self.v += 1

    @endpoint
    async def value(self) -> int:
        return self.v


class Indirect:
    @endpoint
    async def call_value(self, c: Counter) -> int:
        return await c.value.call()


class ParameterServer:
    def __init__(self):
        self.params = torch.rand(10, 10)
        self.grad_buffer = torch.rand(10, 10)

    @endpoint
    async def grad_handle(self) -> RDMABuffer:
        return RDMABuffer(self.grad_buffer)

    async def update(self):
        self.params += 0.01 * self.grad_buffer

    async def log(self):
        print(self.params)


async def main():
    proc = await local_proc_mesh(hosts=1, gpus=2)
    v = await proc.spawn("counter", Counter, 3)
    result = await v.value.call()

    # RuntimeError: an actor with name 'anon' has already been spawned
    i = await proc.spawn("indirect", Indirect)
    result2 = await i.call_value.call(v)
    assert result == result2

    x = await proc.spawn("params", ParameterServer)
    gh = await x.grad_handle.call()

    await gh.write(torch.zeros(10, 10))

    dst = torch.ones(10, 10)

    await gh.read_into(dst)

    assert torch.sum(dst) == 0

    async for x in v.value.stream():
        print(x)


asyncio.run(main())
