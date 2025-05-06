import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from monarch import fetch_shard, local_mesh, remote, Simulator, Stream, Tensor
from monarch.common.borrows import Borrow
from monarch.common.pipe import FakePipe, remote_generator
from monarch.gradient_generator import grad_generator
from monarch.worker.worker import ProcessPipe

logging.basicConfig(level=logging.INFO)


reducer_stream = Stream("reducer_stream")

logger = logging.getLogger(__name__)


DIM = 10000


class Net(nn.Module):
    def __init__(self, nlayers: int):
        super().__init__()
        self.nlayers = nlayers
        self.layers = nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(DIM, DIM))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for i in range(self.nlayers):
            x = self.layers[i * 2](x)
            x = self.layers[i * 2 + 1](x)
        return x


def checkpoint_worker_main(pipe: ProcessPipe, split: str, config: Any) -> None:
    """
    This runs a background process on trainer hosts and communicates with trainer process using monarch.Pipe API
    """
    logger.info("init  checkpoint_worker_main")
    while True:
        msg = pipe.recv()
        logger.info(f"received {msg=}")
        if msg["op"] == "save":
            # spend a few secs in background process to simulate checkpointing
            time.sleep(5)
            path = Path(msg["path"])
            path.mkdir(parents=True, exist_ok=True)
            torch.save(msg["state_dict"], path / f"model_{pipe.ranks['gpu']}.pt")
            t = torch.Tensor(1)
            t[0] = 1
            pipe.send(t)
        elif msg["op"] == "exit":
            t = torch.Tensor(1)
            t[0] = 2
            pipe.send(t)
            break
        else:
            raise ValueError(f"unknown op {msg['op']}")
    logger.info("reached end of process")


@remote_generator("checkpointing.checkpoint_worker_main", max_messages=50)
def create_checkpoint_pipe(pipe: FakePipe, split: str, config: Any) -> []:
    while True:
        yield torch.empty(1)


def log_remote(msg: str) -> None:
    logger.info(f"{threading.current_thread().ident}::{time.ctime()}::{msg}")


remote_logger = remote("checkpointing.log_remote", propagate="inspect")


def borrow_map(state_dict, stream):
    result_state_dict = {}
    borrow_map = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            result_state_dict[key], borrow_map[key] = stream.borrow(value)
        elif isinstance(value, dict):
            result_state_dict[key], borrow_map[key] = borrow_map(value, stream)
        elif isinstance(value, list):
            result_state_dict[key] = []
            borrow_map[key] = []
            for v in value:
                result, borrow = borrow_map(v, stream)
                result_state_dict[key].append(result)
                borrow_map[key].append(borrow)

    return result_state_dict, borrow_map


def drop_borrow_map(state_dict):
    if state_dict is None:
        return
    for _, value in state_dict.items():
        if isinstance(value, dict):
            drop_borrow_map(value)
        if isinstance(value, list):
            for v in value:
                drop_borrow_map(v)
        else:
            value.drop()


def stage(state_dict):
    result_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            result_state_dict[key] = value.cpu()
        elif isinstance(value, dict):
            result_state_dict[key] = stage(value)

    return result_state_dict


def main():
    parser = argparse.ArgumentParser(description="DDP argparser")
    parser.add_argument("--simulate", action="store_true", help="Use SimulatorBackend")
    args = parser.parse_args()

    if args.simulate:
        simulator = Simulator(hosts=1, gpus=2, upload_trace=True)
        device_mesh = simulator.mesh
    else:
        device_mesh = local_mesh(hosts=1, gpus=2)

    torch.set_default_device("cuda")
    with device_mesh.activate():
        model = Net(nlayers=5)
        # Non-fused Adam doesn't work because `step` is a CPU tensor that
        # SC will raise an exception about using a local tensor.
        # So we always have to do fused optimizer
        optimizer = torch.optim.Adam(model.parameters(), fused=True)
        remote_logger("init in worker")

        checkpoint_stream = Stream("checkpoint")
        checkpoint_pipe = create_checkpoint_pipe("test", {"a": 1, "b": 2})
        borrowed_cp_state = {}
        test_tensor = torch.randn((2, 2))

        for step in range(6):
            batch = torch.randn((8, DIM))
            loss = model(batch)
            test_tensor = test_tensor * 2

            # Backward pass and optimization
            loss.sum().backward()

            # drop borrow before optim step.
            drop_borrow_map(borrowed_cp_state)

            optimizer.step()
            optimizer.zero_grad()

            if step > 0 and step % 2 == 0:
                # use a dummy tensor to test checkpointing as borrow does not
                # work on state_dict.
                remote_logger(f"taking a checkpoint on step:{step}")
                model_state_dict, borrowed_cp_state = borrow_map(
                    model.state_dict(), checkpoint_stream
                )
                remote_logger(f" borrowed tensor on {step}")

                # blocking for worker main stream if an earlier checkpoint is still
                # being saved
                with checkpoint_stream.activate():
                    staged_state_dict = stage(model_state_dict)

                    # this ensures the borrow.drop on main stream is not blocked
                    del model_state_dict

                    remote_logger(f"sending tensor to process for step {step}")
                    checkpoint_pipe.send(
                        {
                            "op": "save",
                            "state_dict": staged_state_dict,
                            "path": "checkpoints",
                        }
                    )
                    remote_logger(f"after sending tensor to process for step {step}")
                    # this ensures we block until process saved checkpoint
                    checkpoint_pipe.recv()
                    remote_logger(f"after receive response to process for step {step}")

        # without this the process will hang as it seems to shutdown the pipe process
        # but wait for a message from the process
        fetch_shard(test_tensor).result()
        drop_borrow_map(borrowed_cp_state)

    device_mesh.exit()


if __name__ == "__main__":
    main()
