# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


# pyre-strict

import asyncio
import pickle

import monarch._monarch.hyperactor as hyperactor
from monarch._monarch.hyperactor import Actor


class MyActor(Actor):
    async def handle(
        self, mailbox: hyperactor.Mailbox, message: hyperactor.PythonMessage
    ) -> None:
        pass

    async def hello(self, message: str) -> None:
        print(f"Hello, {message}!")


async def main() -> None:
    hyperactor.init_asyncio_loop()
    proc = hyperactor.Proc()
    print(proc.world_name)
    actor_handle = await proc.spawn(MyActor)
    # @lint-ignore PYTHONPICKLEISBAD
    actor_handle.send(hyperactor.PythonMessage("hello", pickle.dumps(("world",))))

    client = proc.attach("test")
    handle, receiver = client.open_port()
    # @lint-ignore PYTHONPICKLEISBAD
    handle.send(hyperactor.PythonMessage("hello", pickle.dumps(("world",))))
    message = await receiver.recv()
    assert message.method == "hello"
    # @lint-ignore PYTHONPICKLEISBAD
    assert pickle.loads(message.message) == ("world",)

    port_id = handle.bind()
    print("port id", port_id)
    client.post(port_id, hyperactor.PythonMessage("hello", pickle.dumps(("world",))))
    message = await receiver.recv()
    assert message.method == "hello"
    assert pickle.loads(message.message) == ("world",)

    actor_id = actor_handle.bind()
    print("actor_id", actor_id)
    client.post(
        actor_id, hyperactor.PythonMessage("hello", pickle.dumps(("world again",)))
    )

    handle, receiver = client.open_once_port()
    # @lint-ignore PYTHONPICKLEISBAD
    handle.send(hyperactor.PythonMessage("hello", pickle.dumps(("world",))))
    message = await receiver.recv()
    assert message.method == "hello"
    # @lint-ignore PYTHONPICKLEISBAD
    assert pickle.loads(message.message) == ("world",)
    try:
        await receiver.recv()
        raise AssertionError("expected receive to fail")
    except ValueError:
        pass

    port_id = hyperactor.PortId.from_string("foo[0].bar[1][2]")
    print("parsed port id", port_id)

    await asyncio.sleep(5)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
