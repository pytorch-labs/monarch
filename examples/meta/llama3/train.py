# pyre-unsafe
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Example command to run (with monarch conda set up):
python -m llama3.train ./llama3/configs/llama8b_local.py --tp=1 --dp=8 --pp=1  --n_gpus=8 --n_layer=8 --max_iters=30  --dataset=shakespeare --use_te=False --mesh_type=rust_local

"""

import contextlib
import logging
import os
import sys
from typing import List, Tuple

import torch
from llama3.checkpointing import checkpoint, transform_checkpoint_for_te

from llama3.config import reconfigure_worker, TrainConfig

from llama3.data_loader import data_loader_pipe, DataLoaderConfig
from llama3.model import loss_fn, Transformer
from llama3.pp import PipelineParallelism
from llama3.util import (
    estimate_mfu,
    get_num_flop_per_token,
    get_num_params,
    write_perf_stats_to_file,
)

from monarch import (
    fetch_shard,
    inspect,
    python_local_mesh,
    remote,
    rust_backend_mesh,
    Simulator,
)
from monarch._monarch.hyperactor import ActorId
from monarch.memory import dump_memory_snapshot, record_memory_history
from monarch.rust_local_mesh import local_mesh, LoggingLocation, SocketType
from monarch.tensorboard import Tensorboard
from monarch_supervisor.logging import initialize_logging

logger = logging.getLogger(__name__)


# user-defined remote functions
set_worker_logging_level = remote(
    "monarch.worker.worker.set_worker_logging_level", propagate="inspect"
)

set_worker_random_seed = remote(
    "monarch.worker.worker.set_random_seed_impl", propagate="inspect"
)

timer_start = remote(
    "monarch.timer.execution_timer.execution_timer_start",
    propagate=lambda: torch.tensor(0.0, dtype=torch.float64),
)

timer_stop = remote(
    "monarch.timer.execution_timer.execution_timer_stop",
    propagate=lambda: torch.tensor(0.0, dtype=torch.float64),
)

get_latest_time = remote(
    "monarch.timer.execution_timer.get_latest_timer_measurement",
    propagate=lambda: torch.tensor(0.0, dtype=torch.float64),
)

get_ms_per_iter = remote(
    "monarch.timer.execution_timer.get_execution_timer_average_ms",
    propagate=lambda: torch.tensor(0.0, dtype=torch.float64),
)

log = remote("monarch.worker.worker.log", propagate="inspect")


load_weights = remote("llama3.remote_functions.worker_load_weights")


def train(mesh, world_size) -> Tuple[float, int]:
    assert (
        TrainConfig.n_gpus * TrainConfig.n_hosts
        == TrainConfig.pp * TrainConfig.tp * TrainConfig.dp
    ), f"n_gpus ({TrainConfig.n_gpus}) * n_hosts ({TrainConfig.n_hosts})  must be divisible by pp ({TrainConfig.pp}) * tp ({TrainConfig.tp}) * dp ({TrainConfig.dp})"
    o_mesh = mesh
    # for MAST training split host into dp and pp
    if TrainConfig.tp == TrainConfig.n_gpus:
        mesh = mesh.split(host=("dp", "pp"), gpu=("tp",), pp=TrainConfig.pp)
    # for local training split gpu into pp, dp and tp
    else:
        mesh = mesh.flatten("gpu").split(
            gpu=("dp", "pp", "tp"), pp=TrainConfig.pp, tp=TrainConfig.tp
        )
    # Have to set default device instead of initialize on cpu and cast to CUDA
    # because T194391401
    torch.set_default_device("cuda")

    pp_meshes = [mesh.slice(pp=i) for i in range(TrainConfig.pp)]

    with mesh.activate():
        reconfigure_worker(sys.argv[1:])
        set_worker_logging_level(logging.WARNING)

    if TrainConfig.tensorboard:
        tensorboard = Tensorboard(pp_meshes[-1](dp=0, tp=0), TrainConfig.tensorboard)
    else:
        tensorboard = None

    master_process = True
    seed_offset = 0
    tokens_per_iter = world_size * TrainConfig.batch_size * TrainConfig.block_size
    logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")
    torch.manual_seed(1337 + seed_offset)

    iter_num = 0
    num_flop_per_token = None

    # model init
    with mesh.activate():
        tp_group = mesh.process_group(("tp",))
        # all workers should initialize model with the same values
        set_worker_random_seed(TrainConfig.seed, 0)
        with pp_meshes[0].activate():
            model = Transformer(TrainConfig, tp_group)
            num_flop_per_token = get_num_flop_per_token(
                get_num_params(model, exclude_embedding=True),
                TrainConfig,
                TrainConfig.block_size,
            )
            for n, m in model.named_modules():
                m.name = n

        pp_stages = [[] for _ in range(TrainConfig.pp)]
        pp_stages[0] = [model.tok_embeddings]

        STAGES_PER_LAYER = (len(model.layers) + TrainConfig.pp - 1) // TrainConfig.pp
        for idx, layer in enumerate(model.layers):
            pp_stages[idx // STAGES_PER_LAYER].append(layer)
        pp_stages[-1] += [model.norm, model.output]

        pp_class = PipelineParallelism(pp_meshes, pp_stages)

        if TrainConfig.init_from == "resume_training_ckpt":
            raise NotImplementedError("resume not implemented yet")
            # assert (
            #     TrainConfig.checkpoint_dir is not None
            # ), "checkpoint_dir is required if training is resumed"
            # logger.info(
            #     f"Resuming training of {TrainConfig.model_name} from {TrainConfig.checkpoint_dir}"
            # )

        elif TrainConfig.init_from == "resume":
            logger.info(
                f"Loading {TrainConfig.model_name} from {TrainConfig.init_from}"
            )
            model_checkpoint = load_weights(model.state_dict(), TrainConfig.init_from)
            if TrainConfig.use_te:
                model_checkpoint = transform_checkpoint_for_te(
                    model_checkpoint, model, mesh.rank("tp"), TrainConfig.tp
                )

            model.load_state_dict(model_checkpoint, strict=True)
            logger.info("Loaded model successfully!")

        pp_class.initialize()

        logger.info(
            f"Initialized PP model ({TrainConfig.model_name}) num_params {model.get_num_params()} model_size: {model.get_readable_total_size()}"
        )

        model.to(TrainConfig.device)

        # optimizer
        optimizers = pp_class.configure_optimizers(
            TrainConfig, model.configure_optimizers
        )
        data_loader_config = DataLoaderConfig.from_config(TrainConfig)

    with pp_meshes[0].activate():
        train_data_loader = data_loader_pipe("train", data_loader_config)
        eval_data_loader = data_loader_pipe("eval", data_loader_config)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    # TODO log to tensorboard rather than fetch_shard on controller
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, data_loader in [
            ("train", train_data_loader),
            ("val", eval_data_loader),
        ]:
            losses = []
            for _i in range(TrainConfig.eval_iters):
                X, Y = data_loader.recv()
                X, Y = X.to(TrainConfig.device), Y.to(TrainConfig.device)
                Y = Y.to_mesh(pp_meshes[-1])
                with pp_meshes[0].activate():
                    logits = model(X)
                    loss = loss_fn(logits, Y)
                losses.append(torch.reshape(loss, (1, 1)))
            out[split] = torch.cat(losses).mean()
        model.train()
        return out

    # training loop
    with pp_meshes[0].activate():
        # set per-rank random seed before the start of the training loop
        process_idx = mesh.process_idx()
        set_worker_random_seed(TrainConfig.seed, process_idx)

        X, Y = train_data_loader.recv()
        # TODO:  a custom CPU allocator that gets this stuff nicely pinned.
        X, Y = X.to(TrainConfig.device), Y.to(TrainConfig.device)

    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    if TrainConfig.memory_profile:
        for m in pp_meshes:
            with m.activate():
                record_memory_history()

    while True:
        # evaluate the loss on train/val sets and write checkpoints
        if (
            iter_num > 0
            and iter_num % TrainConfig.eval_interval == 0
            and master_process
        ):
            with mesh.activate():
                losses = estimate_loss()
                try:
                    local_losses = fetch_shard(losses).result()
                except Exception as e:
                    logger.error(f"Error happened when fetching shard for losses: {e}")
                    mesh.exit(e)
                    raise e
            logger.info(
                f"step {iter_num}: train loss {local_losses['train']}, val loss {local_losses['val']}"
            )

        if iter_num == 0 and TrainConfig.eval_only:
            break

        with mesh.activate():
            timer_start()

        with pp_meshes[0].activate():
            Y = Y.to_mesh(pp_meshes[-1])
            logits = model(X)
            loss = loss_fn(logits, Y)
            del logits
            X, Y = train_data_loader.recv()
            X, Y = X.to(TrainConfig.device), Y.to(TrainConfig.device)
        with pp_meshes[-1].activate():
            loss.backward()

        with mesh.activate():
            for p in model.parameters():
                # pyre-ignore[16]
                p.grad.reduce_("dp", reduction="avg")
        for o, m in zip(optimizers, pp_class.meshes):
            with m.activate():
                o.step()
                # flush the gradients as soon as we can, no need for this memory anymore
                o.zero_grad(set_to_none=True)

        with mesh.activate():
            # timing and logging is done on worker and passed to controller
            timer_stop()

        if iter_num % TrainConfig.log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            # as with the reference implementation we only fetch the loss from the master_process which is rank 0
            with pp_meshes[-1].activate():
                loss = loss.detach().reduce(("dp",), reduction="avg")
                loss = loss.slice_mesh(dp=0, tp=0).to_mesh(
                    pp_meshes[-1].slice(dp=0, tp=0)
                )
            if local_iter_num >= 5:  # let the training loop settle a bit
                with mesh.activate():
                    dt = get_latest_time()
                    mfu = estimate_mfu(
                        num_flop_per_token=num_flop_per_token,
                        batch_size=TrainConfig.batch_size,
                        seq_len=TrainConfig.block_size,
                        dt=dt / 1000,
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                    log(f"iter {iter_num}")
                if tensorboard:
                    tensorboard.log("step_time (ms)", dt, iter_num)
                    tensorboard.log("mfu", running_mfu, iter_num)

            if tensorboard:
                tensorboard.log("train loss", loss, iter_num)

        if (
            TrainConfig.checkpoint_dir is not None
            and iter_num > 0
            and iter_num % TrainConfig.checkpoint_interval == 0
        ):
            logger.info(f"Saving checkpoint to {TrainConfig.checkpoint_dir}")
            checkpoint(
                TrainConfig.checkpoint_dir,
                iter_num,
                pp_stages,
                pp_meshes,
                optimizers,
            )

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > TrainConfig.max_iters:
            break
    with mesh.activate():
        ms_per_iter = inspect(get_ms_per_iter())

    logger.info("Average step time (ms): %.4f", ms_per_iter)
    if tensorboard:
        tensorboard.close()
    if TrainConfig.memory_profile:
        for m in pp_meshes:
            with m.activate():
                dump_memory_snapshot(dir_snapshots=TrainConfig.memory_profile)
    o_mesh.exit()
    return ms_per_iter / 1000, 0


def main(args: List[str]):
    initialize_logging()

    # # -----------------------------------------------------------------------------
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # # -----------------------------------------------------------------------------
    TrainConfig.configure(args)

    # TODO: factor this logic out into a separate file.
    match TrainConfig.mesh_type:
        case "python":
            mesh = contextlib.nullcontext(
                python_local_mesh(hosts=TrainConfig.n_hosts, gpus=TrainConfig.n_gpus),
            )
        case "rust_local":
            mesh = local_mesh(
                hosts=TrainConfig.n_hosts,
                gpus_per_host=TrainConfig.n_gpus,
            )
        case "rust_mast":
            # All of these environment variables are set inside
            # hyperactor_meta/src/mast_main.rs:spawn_main_script.
            hyperactor_system_addr = os.environ["HYPERACTOR_SYSTEM_ADDR"]
            logger.info(
                f"Using hyperactor, connecting to system at {hyperactor_system_addr} as controller"
            )
            controller_actor_id = ActorId.from_string(
                os.environ["MONARCH_CONTROLLER_ACTOR_ID"]
            )
            worker_world = os.environ["HYPERACTOR_WORKER_WORLD"]

            mesh = contextlib.nullcontext(
                rust_backend_mesh(
                    system_addr=hyperactor_system_addr,
                    hosts=TrainConfig.n_hosts,
                    gpus=TrainConfig.n_gpus,
                    client_proc_id="client[0]",
                    worker_world=worker_world,
                    controller_id=controller_actor_id,
                ),
            )
        case "rust_test":
            mesh = local_mesh(
                hosts=TrainConfig.n_hosts,
                gpus_per_host=TrainConfig.n_gpus,
                socket_type=SocketType.UNIX,
                logging_location=LoggingLocation.DEFAULT,
            )
        case "simulator":
            mesh = contextlib.nullcontext(
                Simulator(
                    hosts=TrainConfig.n_hosts,
                    gpus=TrainConfig.n_gpus,
                    upload_trace=True,
                ).mesh
            )
            # Limit the iterations as we get nothing except for more duplicated
            # traces when training more iterations.
            TrainConfig.max_iters = min(TrainConfig.max_iters, 1)
        case _:
            raise ValueError(f"Unknown mesh type: {TrainConfig.mesh_type}")

    world_size = TrainConfig.n_gpus * TrainConfig.n_hosts
    with mesh as mesh:
        return train(mesh, world_size)


if __name__ == "__main__":
    # run the training loop
    s_per_iter, ret = main(sys.argv[1:])
    write_perf_stats_to_file(s_per_iter)
    sys.exit(ret)
