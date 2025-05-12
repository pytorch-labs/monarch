# pyre-unsafe
"""
This training script is like train.py, but instead of training with Monarch, trains with SPMD.
Mainly intended to be a baseline to measure perf relative to Monarch.

To run (Monarch):
python -m llama3.train ./llama3/configs/llama8b_local.py --tp=1 --dp=8 --pp=1  --n_gpus=8 --n_layer=8 --max_iters=30  --dataset=shakespeare --use_te=False --mesh_type=rust_local

To run (SPMD):
torchrun --standalone --nproc_per_node=8 -m llama3.train_spmd ./llama3/configs/llama8b_local.py --tp=1 --dp=8 --pp=1 --n_gpus=8  --n_layer=8 --max_iters=30  --dataset=shakespeare

"""

import logging
import os
import pprint
import sys
from typing import List, Tuple

import torch
from llama3.config import TrainConfig

from llama3.data_loader import DataLoaderConfig
from llama3.model import loss_fn, Transformer
from llama3.util import (
    estimate_mfu,
    get_num_flop_per_token,
    get_num_params,
    write_perf_stats_to_file,
)
from monarch.timer import ExecutionTimer

from monarch_supervisor.logging import initialize_logging
from nanoGPT.data_loader import get_batch_local_no_pipe
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe


logger = logging.getLogger(__name__)


def train(world_size) -> Tuple[float, int]:
    assert (
        TrainConfig.n_gpus * TrainConfig.n_hosts
        == TrainConfig.pp * TrainConfig.tp * TrainConfig.dp
    ), f"n_gpus ({TrainConfig.n_gpus}) * n_hosts ({TrainConfig.n_hosts})  must be divisible by pp ({TrainConfig.pp}) * tp ({TrainConfig.tp}) * dp ({TrainConfig.dp})"

    world_mesh = init_device_mesh(
        TrainConfig.device,
        (TrainConfig.dp, TrainConfig.tp, TrainConfig.pp),
        mesh_dim_names=("dp", "tp", "pp"),
    )

    torch.set_default_device("cuda")

    seed_offset = 0
    tokens_per_iter = world_size * TrainConfig.batch_size * TrainConfig.block_size
    logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")
    torch.manual_seed(1337 + seed_offset)

    iter_num = 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    default_rank = TrainConfig.local_rank if TrainConfig.local_rank is not None else 0
    ddp_local_rank = int(os.getenv("LOCAL_RANK", default_rank))
    TrainConfig.device = f"cuda:{ddp_local_rank}"
    device = torch.device(TrainConfig.device)
    torch.cuda.set_device(device)

    tp_group = world_mesh["tp"].get_group()
    model = Transformer(TrainConfig, tp_group)

    model_param_count = get_num_params(model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        TrainConfig,
        TrainConfig.block_size,
    )
    logging.info(
        "Model param count: %d, num_flop_per_token: %d",
        model_param_count,
        num_flop_per_token,
    )

    # pipeline setup
    pp_enabled = TrainConfig.pp > 1
    pp_mesh = world_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    if TrainConfig.batch_size < pp_size:
        raise ValueError(
            f"batch size {TrainConfig.batch_size} must be >= pp size {pp_size}"
        )
    is_logger_process = (
        pp_enabled and rank == pp_size - 1 or not pp_enabled and rank == 0
    )
    if pp_enabled:
        with torch.device("meta"):
            stages_per_layer = (
                len(model.layers) + TrainConfig.pp - 1
            ) // TrainConfig.pp

            # Retain only the layers that should be present for this pipeline stage
            start_layer = pp_rank * stages_per_layer
            end_layer = min(start_layer + stages_per_layer, len(model.layers))
            model.layers = model.layers[start_layer:end_layer]

            # example_input/output is needed in older PyTorch versions where
            # shape inference is not supported.
            if pp_rank != 0:
                model.tok_embeddings = None
            if pp_rank != pp_size - 1:
                model.norm = None
                model.output = None

            if pp_rank == 0:
                example_input = torch.zeros(
                    (1, TrainConfig.block_size), dtype=torch.int64
                )
                example_output = torch.zeros(
                    (1, TrainConfig.block_size, TrainConfig.dim), dtype=torch.bfloat16
                )
            elif pp_rank != pp_size - 1:
                example_input = torch.zeros(
                    (1, TrainConfig.block_size, TrainConfig.dim), dtype=torch.bfloat16
                )
                example_output = torch.zeros(
                    (1, TrainConfig.block_size, TrainConfig.dim), dtype=torch.bfloat16
                )
            else:  # pp_rank == pp_size - 1
                example_input = torch.zeros(
                    (1, TrainConfig.block_size, TrainConfig.dim), dtype=torch.bfloat16
                )
                example_output = torch.zeros(
                    (1, TrainConfig.block_size, TrainConfig.vocab_size),
                    dtype=torch.bfloat16,
                )

            stage = PipelineStage(
                model,
                stage_index=pp_rank,
                num_stages=pp_size,
                device=device,
                group=pp_mesh.get_group(),
                input_args=example_input,
                output_args=example_output,
            )
        model.to(device=device)

        # TODO - make num_microbatches configurable. Is this equal to monarch?
        n_microbatches = pp_size
        train_schedule = ScheduleGPipe(
            stage=stage, n_microbatches=n_microbatches, loss_fn=loss_fn
        )
        eval_schedule = ScheduleGPipe(stage=stage, n_microbatches=n_microbatches)

    optimizer = model.configure_optimizers(
        TrainConfig.weight_decay,
        TrainConfig.learning_rate,
        (TrainConfig.beta1, TrainConfig.beta2),
        device,
        TrainConfig.optimizer,
        model.parameters(),
    )

    # TODO: after DDP w/ overlap on train.py, add real DDP back
    # model = DDP(model, device_ids=[ddp_local_rank])

    if (
        TrainConfig.init_from == "resume_training_ckpt"
        or TrainConfig.init_from == "resume"
    ):
        raise NotImplementedError("resume not implemented yet")

    model.to(device)

    data_loader_config = DataLoaderConfig.from_config(TrainConfig)
    # helps estimate an arbitrarily accurate loss over either split using many batches

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            if pp_enabled:
                losses = []
                for _ in range(TrainConfig.eval_iters):
                    X, Y = get_batch_local_no_pipe(split, data_loader_config)

                    if pp_rank == 0:
                        eval_schedule.step(X)
                    elif pp_rank == pp_size - 1:
                        logits = eval_schedule.step(target=Y)
                        losses.append(loss_fn(logits, Y))
                    else:
                        eval_schedule.step()
                out[split] = (
                    torch.mean(torch.stack(losses))
                    if pp_rank == pp_size - 1
                    else torch.tensor([-1.0], device=device)
                )
            else:
                losses = torch.zeros(TrainConfig.eval_iters)
                for k in range(TrainConfig.eval_iters):
                    X, Y = get_batch_local_no_pipe(split, data_loader_config)
                    logits = model(X)
                    loss = loss_fn(logits, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        model.train()
        return out

    # training loop
    X, Y = get_batch_local_no_pipe(
        "train", data_loader_config
    )  # fetch the very first batch

    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    dp_group = world_mesh["dp"].get_group()

    while True:
        # evaluate the loss on train/val sets and write checkpoints
        if (
            iter_num > 0
            and iter_num % TrainConfig.eval_interval == 0
            or TrainConfig.eval_only
        ):
            losses = estimate_loss()
            if is_logger_process:
                logger.info(
                    f"step {iter_num}: train loss {losses['train']}, val loss {losses['val']}"
                )
        if iter_num == 0 and TrainConfig.eval_only:
            break

        X, Y = get_batch_local_no_pipe("train", data_loader_config)
        with ExecutionTimer.time():
            if pp_enabled:
                losses = []
                targets, losses = (Y, []) if pp_rank == pp_size - 1 else (None, None)
                if pp_rank == 0:
                    train_schedule.step(X, target=targets, losses=losses)
                else:
                    train_schedule.step(target=Y, losses=losses)
                loss = (
                    torch.mean(torch.stack(losses))
                    if pp_rank == pp_size - 1
                    else torch.tensor([-1.0])
                )
            else:
                logits = model(X)
                loss = loss_fn(logits, Y)
                loss.backward()

            # TODO: after DDP w/ overlap on train.py, remove manual all_reduce
            for p in model.parameters():
                assert p.grad is not None
                dp_group.allreduce(p.grad, op=dist.ReduceOp.SUM)
                assert p.grad is not None
                p.grad /= world_size

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        dt = ExecutionTimer.get_latest_measurement() / 1000
        if iter_num % TrainConfig.log_interval == 0 and is_logger_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            # as with the reference implementation we only fetch the loss from the lgoger_process
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = estimate_mfu(
                    num_flop_per_token=num_flop_per_token,
                    batch_size=TrainConfig.batch_size,
                    seq_len=TrainConfig.block_size,
                    dt=dt,
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            logger.info(
                f"iter {iter_num}: time {dt*1000:.2f}ms, loss {loss}, mfu {running_mfu*100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > TrainConfig.max_iters:
            break

    if is_logger_process:
        logging.info(
            "Training step time results: %s", pprint.pformat(ExecutionTimer.summary())
        )
    ms_per_iter = ExecutionTimer.summary()["default"]["mean_ms"]
    dist.destroy_process_group()
    return ms_per_iter / 1000, 0


def main(args: List[str]) -> Tuple[float, int]:
    initialize_logging()

    # # -----------------------------------------------------------------------------
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # # -----------------------------------------------------------------------------
    TrainConfig.configure(args)
    TrainConfig.use_monarch = False

    world_size = TrainConfig.n_gpus * TrainConfig.n_hosts
    return train(world_size)


if __name__ == "__main__":
    # run the training loop
    s_per_iter, ret = main(sys.argv[1:])
    write_perf_stats_to_file(s_per_iter)
    sys.exit(ret)
