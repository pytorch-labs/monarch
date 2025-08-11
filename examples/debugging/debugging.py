# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from monarch.actor import Actor, current_rank, debug_client, endpoint, proc_mesh


def _bad_rank():
    raise ValueError("bad rank")


def _debugee_actor_internal(rank):
    if rank % 4 == 0:
        breakpoint()  # noqa
        rank += 1
        return rank
    elif rank % 4 == 1:
        breakpoint()  # noqa
        rank += 2
        return rank
    elif rank % 4 == 2:
        breakpoint()  # noqa
        rank += 3
        _bad_rank()
    elif rank % 4 == 3:
        breakpoint()  # noqa
        rank += 4
        return rank


class DebugeeActor(Actor):
    @endpoint
    async def to_debug(self):
        rank = current_rank().rank
        return _debugee_actor_internal(rank)


async def main() -> None:
    process_mesh = proc_mesh(hosts=4, gpus=4)
    debugee_mesh = process_mesh.spawn("debugee", DebugeeActor).get()
    res = debugee_mesh.to_debug.call()
    debug_client().enter.call_one().get()
    print(res.get())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
