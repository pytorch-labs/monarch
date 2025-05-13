# pyre-unsafe
import asyncio
import itertools
import pickle
from dataclasses import dataclass, replace
from typing import Any, Callable, List

import torch

from morpho.config import DebugConfig, MachineConfig, TrainingConfig
from morpho.debug.observer import plot_runs

from morpho.subprocess import call
from morpho.titan import train as train_titan
from morpho.train import train


# the different things we can configure to sweep over with this functionality.
@dataclass(frozen=True)
class SweepConfig:
    training: TrainingConfig
    machine: MachineConfig
    debug: DebugConfig
    experimental_train: Any


async def local_sweep(configs: List[SweepConfig]):
    free_gpus = asyncio.Queue()
    for i in range(torch.cuda.device_count()):
        free_gpus.put_nowait(i)

    async def run_one(config: SweepConfig, gpus: List[int]):
        visible = ",".join(str(i) for i in gpus)
        machine = replace(config.machine, visible_gpus=visible)
        train_fn = (
            config.experimental_train
            if config.debug.variant == "experiment"
            else train_titan
        )
        try:
            result = await call(
                train_fn, machine=machine, training=config.training, debug=config.debug
            )
        finally:
            for g in gpus:
                free_gpus.put_nowait(g)
        return result

    tasks = []
    for config in sorted(configs, key=lambda x: x.machine.ngpu, reverse=True):
        gpus = [await free_gpus.get() for _ in range(config.machine.ngpu)]
        tasks.append(asyncio.create_task(run_one(config, gpus)))

    return [await t for t in tasks]


debug_training_config = replace(TrainingConfig.default, steps=3)
debug_machine_config = replace(MachineConfig.default, ngpu=1)


async def local_debug_sweep(
    filename: str,
    run: str = "debug_sweep",
    noise: bool = True,
    reassociated: bool = True,
    samples: int = 2,
    training: TrainingConfig = debug_training_config,
    machine: MachineConfig = debug_machine_config,
    experimental_train: Callable = train,
):
    sweep = []
    variants = ["reference", "experiment"]
    if noise:
        variants.append("noise")
    if reassociated:
        variants.append("reassociated")
    for sample, variant in itertools.product(range(samples), variants):
        debug = DebugConfig(sample=sample, variant=variant, observe=True, run=run)  # type: ignore

        sweep.append(SweepConfig(training, machine, debug, experimental_train))

    result_filenames = await local_sweep(sweep)
    observations = [pickle.load(open(filename, "rb")) for filename in result_filenames]
    plot_runs(observations, output_filename=filename)
