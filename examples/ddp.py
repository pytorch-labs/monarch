# pyre-unsafe
import argparse
import logging
import sys
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable, cast, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from monarch import fetch_shard, Simulator, Stream, Tensor
from monarch.common.borrows import Borrow
from monarch.gradient_generator import grad_generator
from monarch.memory import dump_memory_snapshot, record_memory_history
from monarch.profiler import profile, record_function, Schedule
from monarch.rust_local_mesh import local_mesh, LoggingLocation, SocketType
from monarch.tensorboard import Tensorboard

logging.basicConfig(level=logging.INFO)


reducer_stream = Stream("reducer_stream")


@dataclass
class Bucket:
    params: List[Tensor]
    grads: List[Tensor]
    bucket_tensor: Optional[Tensor] = None
    borrow: Optional[Borrow] = None


def replicate(
    model: nn.Module,
    bucket_size: int,
    max_unsync_buckets: int = 1,
) -> Callable:
    def reduce_bucket(bucket: Bucket) -> Bucket:
        bucket_tensor = cast(
            Tensor, torch.cat(tuple(g.flatten() for g in bucket.grads))
        )
        borrow_tensor, borrow = reducer_stream.borrow(bucket_tensor)

        with reducer_stream.activate():
            borrow_tensor.reduce_("host", reduction="avg")

        bucket.bucket_tensor, bucket.borrow = bucket_tensor, borrow
        return bucket

    def unflatten_bucket(bucket: Bucket) -> Bucket:
        split_sizes = [grad.numel() for grad in bucket.grads]
        assert bucket.borrow is not None
        bucket.borrow.drop()
        bucket_tensor = cast(Tensor, bucket.bucket_tensor)
        for param, grad in zip(bucket.params, bucket_tensor.split(split_sizes)):
            param.grad = grad.reshape(param.shape)
        return bucket

    def bucket_generator(generator) -> Generator[Bucket, None, None]:
        curr_bucket_size = 0
        bucket = Bucket([], [])
        for param, grad in generator:
            assert param.shape == grad.shape, (param.shape, grad.shape)
            curr_bucket_size += param.numel() * param.element_size() / 1_000_000.0
            bucket.params.append(param)
            bucket.grads.append(grad)
            if curr_bucket_size < bucket_size:
                continue

            curr_bucket_size = 0
            yield bucket
            bucket = Bucket([], [])

        if curr_bucket_size:
            yield bucket

    def backward(
        loss, params, max_unsync_buckets
    ) -> Generator[Tuple[Tensor], None, None]:
        # Design doc of gradient generator: https://fburl.com/gdoc/s5he4kyo
        generator = grad_generator(loss, params)
        unsync_buckets = deque()

        for bucket in bucket_generator(zip(params, generator, strict=True)):
            unsync_buckets.append(reduce_bucket(bucket))
            if len(unsync_buckets) > max_unsync_buckets:
                reduced_bucket = unflatten_bucket(unsync_buckets.pop())
                yield tuple(reduced_bucket.params)
        while unsync_buckets:
            reduced_bucket = unflatten_bucket(unsync_buckets.pop())
            yield tuple(reduced_bucket.params)

    params = list(reversed(list((model.parameters()))))
    return partial(backward, params=params, max_unsync_buckets=max_unsync_buckets)


DIM = 100


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


def maybe_profile(should_profile: bool):
    if should_profile:
        return profile(
            activities=[
                # pyre-ignore[16]: Module `profiler` has no attribute
                torch.profiler.ProfilerActivity.CPU,
                # pyre-ignore[16]: Module `profiler` has no attribute
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready="./traces/",
            schedule=Schedule(wait=1, warmup=1, active=2, repeat=1),
            record_shapes=True,
        )
    else:
        return nullcontext()


def maybe_record_memory_history(should_record_memory_history: bool):
    if should_record_memory_history:
        return record_memory_history()


def maybe_dump_snapshot(should_dump_snapshot: bool):
    if should_dump_snapshot:
        return dump_memory_snapshot(dir_snapshots="./snapshots/")


def main(running_as_unittest=False):
    parser = argparse.ArgumentParser(description="DDP argparser")
    parser.add_argument(
        "--memory-snapshot",
        action="store_true",
        help="Records memory history and dumps snapshots.",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling of the model."
    )
    parser.add_argument("--simulate", action="store_true", help="Use SimulatorBackend")
    parser.add_argument(
        "--tensorboard", type=str, default="", help="TensorBoard log directory"
    )

    if running_as_unittest:
        sys.argv = [sys.argv[0]]
    args = parser.parse_args()

    if args.simulate:
        simulator = Simulator(hosts=1, gpus=2, upload_trace=True)
        device_mesh = nullcontext(simulator.mesh)
    else:
        device_mesh = local_mesh(
            hosts=1,
            gpus_per_host=2,
            socket_type=SocketType.UNIX,
            logging_location=LoggingLocation.DEFAULT,
        )

    with device_mesh as device_mesh:
        rank0_mesh = device_mesh(host=0, gpu=0)
        if args.tensorboard:
            tensorboard = Tensorboard(rank0_mesh, args.tensorboard)
        else:
            tensorboard = None

        optimizer_in_backward = True

        with device_mesh.activate():
            torch.set_default_device("cuda")
            net = Net(nlayers=5)

            optimizers = []
            max_unsync_buckets = 1 if optimizer_in_backward else 10000
            backward = replicate(
                net, bucket_size=25, max_unsync_buckets=max_unsync_buckets
            )

            maybe_record_memory_history(args.memory_snapshot)
            with maybe_profile(args.profile) as prof:
                for step in range(2):
                    batch = torch.randn((8, DIM))
                    with record_function("forward"):
                        loss = net(batch)

                        if step == 0:
                            with record_function("backward_optimizer_1st"):
                                # Non-fused Adam doesn't work because `step` is a CPU tensor that
                                # SC will raise an exception about using a local tensor.
                                # So we always have to do fused optimizer
                                for params in backward(loss.sum()):
                                    optimizers.append(
                                        torch.optim.Adam(params, fused=True)
                                    )
                                    optimizers[-1].step()
                        else:
                            with record_function("backward_optimizer"):
                                for optim, _ in zip(
                                    optimizers, backward(loss.sum()), strict=True
                                ):
                                    optim.step()

                    with record_function("zero_grad"):
                        for optim in optimizers:
                            optim.zero_grad()

                    loss = loss.sum().detach()
                    # TODO: do multi-dimensional reduce once reduce() supports
                    # the feature.
                    loss = loss.reduce(("gpu", "host"), reduction="avg")
                    loss = loss.slice_mesh(gpu=0, host=0).to_mesh(rank0_mesh)
                    loss = fetch_shard(loss).result()
                    if tensorboard:
                        tensorboard.log("loss", loss, step)

                    if args.profile:
                        prof.step()

                    logging.info(f"Step {step} done, loss {loss}")

            maybe_dump_snapshot(args.memory_snapshot)
            if tensorboard:
                tensorboard.close()
        device_mesh.exit()


if __name__ == "__main__":
    main()
