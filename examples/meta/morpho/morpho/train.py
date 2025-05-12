# this file currently contains basically the same code as titan.py
# but will be modified step-by-step to run the monarch training loop.

# pyre-unsafe
import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import monarch
import monarch.random

import torch

from monarch_supervisor.logging import initialize_logging
from morpho.checkpointer import Checkpointer
from morpho.config import (
    DebugConfig,
    JobConfig,
    MachineConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)

from morpho.dataloader import create_dataloader, Cursor, Dataloader
from morpho.debug.inject import inject_debug_behavior
from morpho.debug.observer import Observer
from morpho.lr_schedules import linear_warmup_linear_decay
from morpho.optimizer import AdamW, HyperParam, Optimizer
from morpho.random_context import RandomContext
from morpho.report import Report
from morpho.trainer import Trainer
from torch import Generator
from torchtitan import utils
from torchtitan.models import model_name_to_cls, models_config

# have to copy-pasta train.py because torch titan doesn't include it as
# part of its package. It also doesn't contain its default config, so we
# need to put it here.

# this is an example of how not using Python APIs to configure things
# just magnifies glue code.

logger = logging.getLogger(__name__)


def lr_schedule(lr: float, warmup_steps: int, steps: int) -> HyperParam:
    decay_steps = max(1, steps - warmup_steps)
    logger.info(
        f"learning_rate: {lr}, {warmup_steps} warmup steps, {decay_steps} decay steps"
    )
    return functools.partial(linear_warmup_linear_decay, lr, warmup_steps, decay_steps)


def make_deterministic():
    logger.info("Deterministic algorithm enabled (expect perf degradation).")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # env var for deterministic CuBLAS
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@contextmanager
def _initialize_training_state(
    machine: MachineConfig, debug: DebugConfig, training: TrainingConfig
):
    initialize_logging()
    # !! Machine acquisition stuff

    if machine.visible_gpus is not None:
        # TODO: just make local_mesh take a list of GPUs to use.
        os.environ["CUDA_VISIBLE_DEVICES"] = machine.visible_gpus

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    global_mesh = monarch.python_local_mesh(gpus=machine.ngpu)
    mesh = global_mesh.flatten("dp")
    with mesh.activate():
        if training.deterministic:
            monarch.random.make_deterministic()

        def random_context_from_base_seed(seed):
            return new_random_context(torch.tensor(seed))

        randomness = random_context_from_base_seed(training.seed)

        with randomness, inject_debug_behavior(
            debug, random_context_from_base_seed
        ) as (
            observer,
            debug_filename,
        ):
            yield mesh, observer, debug_filename
    global_mesh.exit()


def construct_trainer(
    mesh,
    observer: Optional[Observer],
    job: JobConfig,
    training: TrainingConfig,
    model_config: ModelConfig,
    parallelism: ParallelismConfig,
    should_checkpoint,
) -> Tuple[Trainer, Dataloader, Any, int]:
    dataloader = create_dataloader(training, model_config.tokenizer_path)

    model, num_flop_per_token = load_model(
        model_config, dataloader.n_words, training.seq_len
    )
    report = Report(job.metrics.log_freq, num_flop_per_token)

    optimizer = Optimizer()
    optimizer.add_model(
        model,
        AdamW(
            lr=lr_schedule(training.lr, training.warmup_steps, training.steps),
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
        ),
    )

    checkpointer = Checkpointer(job.checkpoint.folder, should_checkpoint)
    if observer:
        report.add_metrics_logger(observer)

    cursor = None
    step = 0
    starting_checkpoint = checkpointer.load()
    if starting_checkpoint is not None:
        cursor = starting_checkpoint.dataloader_cursor
        step = starting_checkpoint.step + 1

    trainer = Trainer(
        mesh,
        model,
        optimizer,
        checkpointer,
        report,
        parallelism,
        training.max_norm,
        starting_checkpoint,
    )
    return trainer, dataloader, cursor, step


def train(
    job: JobConfig = JobConfig.default,
    training: TrainingConfig = TrainingConfig.default,
    machine: MachineConfig = MachineConfig.default,
    debug: DebugConfig = DebugConfig.default,
    model_config: ModelConfig = ModelConfig.default,
    parallelism: ParallelismConfig = ParallelismConfig.default,
):
    def should_checkpoint(step, last):
        return job.checkpoint.enable_checkpoint and (
            (step > 0 and step + 1 % job.checkpoint.interval == 0) or last
        )

    with _initialize_training_state(machine, debug, training) as (
        mesh,
        observer,
        debug_filename,
    ):
        trainer, dataloader, cursor, step = construct_trainer(
            mesh, observer, job, training, model_config, parallelism, should_checkpoint
        )
        trainer.train(dataloader, cursor, step, training.steps)
    return debug_filename


def train_checkpoint_testing(
    job: JobConfig = JobConfig.default,
    training: TrainingConfig = TrainingConfig.default,
    machine: MachineConfig = MachineConfig.default,
    debug: DebugConfig = DebugConfig.default,
    model_config: ModelConfig = ModelConfig.default,
    parallelism: ParallelismConfig = ParallelismConfig.default,
):
    with _initialize_training_state(machine, debug, training) as (
        mesh,
        observer,
        debug_filename,
    ):
        for _i in range(training.steps):
            trainer, dataloader, cursor, step = construct_trainer(
                mesh,
                observer,
                job,
                training,
                model_config,
                parallelism,
                lambda step, last: True,
            )
            trainer.train(dataloader, cursor, step, step + 1)
    return debug_filename


def load_model(model_config: ModelConfig, n_words: int, seq_len: int):
    # build model (using meta init)
    model_name = model_config.name
    model_cls = model_name_to_cls[model_name]
    titan_model_config = models_config[model_name][model_config.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    titan_model_config.norm_type = model_config.norm_type
    titan_model_config.vocab_size = n_words
    titan_model_config.max_seq_len = seq_len

    logger.info(
        f"Building {model_name} {model_config.flavor} with {titan_model_config}"
    )
    with torch.device("meta"):
        model = model_cls.from_model_args(titan_model_config)

    # a no-op hander if float8 is not enabled
    # log model size
    model_param_count = utils.get_num_params(model)

    logger.info(
        f"Model {model_name} {model_config.flavor} "
        f"size: {model_param_count:,} total parameters"
    )

    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        titan_model_config,
        seq_len,
    )

    return model, num_flop_per_token


def new_random_context(seed: torch.Tensor) -> RandomContext:
    return RandomContext(
        monarch.random.new_state(seed),
        monarch.random.get_state,
        monarch.random.set_state,
    )
