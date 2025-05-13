# pyre-unsafe
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To Run:

1. Download the dataset:
   $ python ~/fbsource/fbcode/monarch/examples/nanoGPT/data/shakespeare_char/prepare.py

2. Run the trainer (local + single GPU):
   $ cd ~/fbsource/fbcode
   $ buck run @mode/opt monarch/examples/nanoGPT:train -- \
        ./monarch/examples/nanoGPT/config/train_shakespeare_char_small.py \
        --data_root_dir=./monarch/examples/nanoGPT/data/ \
        --eval_only=False \
        --n_gpus=1

NOTE: //monarch/examples/tests/nanoGPT:test_nanogpt
      runs this example with mocked Pipe (but NOT MockMesh)
      hence the dataset need not be present for the test
"""

import gc
import logging

import math
import os
import pickle
import sys

import time
import unittest.mock
from contextlib import nullcontext

import torch
from monarch import DeviceMesh, fetch_shard, local_mesh, remote

# this function helps get a local device mesh for testing
from monarch._testing import mock_mesh
from monarch.common._coalescing import compile

from nanoGPT.config import NanoGPTConfig, reconfigure_worker
from nanoGPT.data_loader import data_loader_pipe, DataLoaderConfig
from nanoGPT.model import GPT, GPTConfig


log = remote("monarch.worker._testing_function.log", propagate="inspect")

set_worker_logging_level = remote(
    "monarch.worker.worker.set_worker_logging_level", propagate="inspect"
)

set_worker_random_seed = remote(
    "monarch.worker.worker.set_random_seed_impl", propagate="inspect"
)


def main(args=None):
    # buck main_module bypasses the __main__ block, so we have to manually set the config
    args = sys.argv[1:] if not args else args
    # # -----------------------------------------------------------------------------
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # # -----------------------------------------------------------------------------
    NanoGPTConfig.configure(args)

    if NanoGPTConfig.mocked:
        device_mesh = nullcontext(
            mock_mesh(hosts=NanoGPTConfig.n_hosts, gpus=NanoGPTConfig.n_gpus)
        )
    else:
        device_mesh = local_mesh(
            hosts=NanoGPTConfig.n_hosts,
            gpus_per_host=NanoGPTConfig.n_gpus,
        )

    with device_mesh as device_mesh:
        run(device_mesh, args)


def run(device_mesh: DeviceMesh, args=None) -> None:
    """Runs training. Assumes that the caller has:

    1. Configured configs from CLI arguments by calling `NanoGPTConfig.configure(args)`
    2. Caller has created a device mesh (but NOT activated it)
    3. Data files exists at `NanoGPTConfig.data_root_dir`
    4. NB: the original `args` used in #1 should be passed to reconfigure the workers' configs
    """

    # Have to set default device instead of initialize on cpu and cast to CUDA
    # because T194391401
    torch.set_default_device("cuda")

    # Currently gc is expensive bc we have lots of objects.
    # For now, disable in training loop and manually request_status (clearing idents) and gc.collect()
    # TODO: Long term, we want to request status once the # of invocations is above a threshold.
    # TODO: We’d want to pipeline actually waiting for that status until the next status request so that we aren’t constantly pausing.
    if not NanoGPTConfig.monarch_compile:
        gc.disable()

    maybe_compile = (
        compile(verify=True) if NanoGPTConfig.monarch_compile else lambda fn: fn
    )

    use_local_sgd = NanoGPTConfig.local_steps > 0
    if use_local_sgd:
        n_hosts = NanoGPTConfig.n_hosts
        n_gpus = NanoGPTConfig.n_gpus
        n_local = NanoGPTConfig.local_group_size
        assert (
            n_local % n_gpus == 0
        ), "local SGD group size should be multiple of n_gpus"
        assert (
            n_gpus * n_hosts
        ) % n_local == 0, "local SGD group size should divide n_gpus * n_hosts"
        outer_hosts = n_hosts * n_gpus // n_local
        split_mesh = device_mesh.split(host=("oh", "ih"), oh=outer_hosts)
        default_mesh = split_mesh
    else:
        default_mesh = device_mesh

    world_size = NanoGPTConfig.n_gpus * NanoGPTConfig.n_hosts
    # in local mesh the outer dimension is outer groups, inner dims combined are inner group

    with default_mesh.activate():
        reconfigure_worker(args)
        set_worker_logging_level(logging.WARNING)

    master_process = True
    seed_offset = 0
    tokens_per_iter = (
        NanoGPTConfig.gradient_accumulation_steps
        * world_size
        * NanoGPTConfig.batch_size
        * NanoGPTConfig.block_size
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(NanoGPTConfig.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[NanoGPTConfig.dtype]
    ctx = (
        nullcontext()
        if NanoGPTConfig.device_type == "cpu"
        else torch.amp.autocast(device_type=NanoGPTConfig.device_type, dtype=ptdtype)
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(NanoGPTConfig.data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    elif NanoGPTConfig.data_vocab_size > 0:
        # Manually set vocab size to config falue
        meta_vocab_size = NanoGPTConfig.data_vocab_size
        print(f"Manual vocab size = {NanoGPTConfig.data_vocab_size} ")

    # model init
    with default_mesh.activate():
        torch.set_default_device("cuda")

        checkpoint = None
        model_args = {
            "n_layer": NanoGPTConfig.n_layer,
            "n_head": NanoGPTConfig.n_head,
            "n_embd": NanoGPTConfig.n_embd,
            "block_size": NanoGPTConfig.block_size,
            "bias": NanoGPTConfig.bias,
            "vocab_size": None,
            "dropout": NanoGPTConfig.dropout,
        }  # start with model_args from command line
        if NanoGPTConfig.init_from == "scratch":
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print(
                    "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
                )
            model_args["vocab_size"] = (
                meta_vocab_size if meta_vocab_size is not None else 50304
            )
            # pyre-fixme[6]: For 1st argument expected `bool` but got `Optional[float]`.
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            if use_local_sgd:
                outer_model = GPT(gptconf)
        elif NanoGPTConfig.init_from == "resume":
            print(f"Resuming training from {NanoGPTConfig.out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(NanoGPTConfig.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=NanoGPTConfig.device)  # noqa: TOR102
            checkpoint_model_args = checkpoint["model_args"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            # pyre-fixme[6]: For 1st argument expected `bool` but got `Optional[float]`.
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k in state_dict:
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)  # noqa: E203
            model.load_state_dict(state_dict)
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint["best_val_loss"]  # noqa: F841
        elif NanoGPTConfig.init_from.startswith("gpt2"):
            print(f"Initializing from OpenAI GPT-2 weights: {NanoGPTConfig.init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = {"dropout": NanoGPTConfig.dropout}  # noqa: F841
            model = GPT.from_pretrained(
                NanoGPTConfig.init_from, NanoGPTConfig.override_args
            )
            # read off the created config params, so we can store them into checkpoint correctly
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                model_args[k] = getattr(model.config, k)
        # crop down the model block size if desired, using model surgery
        if NanoGPTConfig.block_size < model.config.block_size:
            model.crop_block_size(NanoGPTConfig.block_size)
            model_args["block_size"] = (
                NanoGPTConfig.block_size
            )  # so that the checkpoint will have the right value
        model.to(NanoGPTConfig.device)
        if use_local_sgd:
            outer_model.to(NanoGPTConfig.device)
            outer_model.load_state_dict(model.state_dict())
        # set worker random seeds to be unique and reproducible per worker
        process_idx = default_mesh.process_idx()
        set_worker_random_seed(NanoGPTConfig.seed, process_idx)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(NanoGPTConfig.dtype == "float16")
        )

        # optimizer
        optimizer = model.configure_optimizers(
            NanoGPTConfig.weight_decay,
            NanoGPTConfig.learning_rate,
            (NanoGPTConfig.beta1, NanoGPTConfig.beta2),
            NanoGPTConfig.device_type,
        )
        if use_local_sgd > 0:
            outer_optimizer = outer_model.configure_optimizers(
                NanoGPTConfig.outer_optim_weight_decay,
                NanoGPTConfig.outer_optim_lr,
                (),
                NanoGPTConfig.device_type,
                NanoGPTConfig.outer_optim_type,
                NanoGPTConfig.outer_optim_momentum,
            )
        if NanoGPTConfig.init_from == "resume":
            assert checkpoint is not None
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            # we want to initialize the optimizer state before taking a
            # timestep so that the first iteraiton does not do additional
            # initialization work, making it not capturable as a trace.

            # unfortunately, the optimizer API makes this pretty difficult.
            # We need to actually have the optimizer step with some grad
            # tensors to get things initialized. However, we also do not
            # want the supplemental state such as momemntums to get updated
            # during this step. So we pretend to do a step but mock out
            # the actual optimizer update step, which invokes the initialization
            # code but not the update code.

            all_params = [p for p in model.parameters() if p.requires_grad]
            for p in all_params:
                p.grad = torch.zeros_like(p)
            orig = optimizer.__class__.__base__.__init__.__globals__["adam"]
            try:
                m = optimizer.__class__.__base__.__init__.__globals__["adam"] = (
                    unittest.mock.MagicMock()
                )
                optimizer.step()
                assert m.called, "didn't call adam?"
                optimizer.zero_grad(set_to_none=True)
            finally:
                optimizer.__class__.__base__.__init__.__globals__["adam"] = orig

        checkpoint = None  # free up memory

        # compile the model
        if NanoGPTConfig.compile:
            print("compiling the model... (takes a ~minute)")
            #    unoptimized_model = model
            model = torch.compile(model)  # requires PyTorch 2.0

        data_loader_config = DataLoaderConfig.from_config(NanoGPTConfig)
        train_data_loader = data_loader_pipe("train", data_loader_config)
        eval_data_loader = data_loader_pipe("eval", data_loader_config)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @maybe_compile
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, data_loader in [
            ("train", train_data_loader),
            ("val", eval_data_loader),
        ]:
            losses = []
            for k in range(NanoGPTConfig.eval_iters):
                X, Y = data_loader.recv()
                X, Y = X.to(NanoGPTConfig.device), Y.to(NanoGPTConfig.device)
                with ctx:
                    logits, loss = model(X, Y)
                losses.append(torch.reshape(loss, (1, 1)))
            out[split] = torch.cat(losses).mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < NanoGPTConfig.warmup_iters:
            return NanoGPTConfig.learning_rate * it / NanoGPTConfig.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > NanoGPTConfig.lr_decay_iters:
            return NanoGPTConfig.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - NanoGPTConfig.warmup_iters) / (
            NanoGPTConfig.lr_decay_iters - NanoGPTConfig.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return NanoGPTConfig.min_lr + coeff * (
            NanoGPTConfig.learning_rate - NanoGPTConfig.min_lr
        )

    # logging
    if NanoGPTConfig.wandb_log and NanoGPTConfig.master_process:
        import wandb  # @manual=fbsource//third-party/pypi/wandb:wandb

        wandb.init(
            project=NanoGPTConfig.wandb_project,
            name=NanoGPTConfig.wandb_run_name,
            config=NanoGPTConfig.getConfigDict(),
        )

    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0

    with default_mesh.activate():
        lr_tensor = torch.zeros(())
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_tensor

    @maybe_compile
    def step():
        # forward backward update, with optional radient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        loss = None
        with default_mesh.activate():
            for _ in range(NanoGPTConfig.gradient_accumulation_steps):
                X, Y = train_data_loader.recv()
                X, Y = X.to(NanoGPTConfig.device), Y.to(NanoGPTConfig.device)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = (
                        loss / NanoGPTConfig.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # reduce gradients across all GPUs and scale by world size
            reduction_dims = ("ih", "gpu") if use_local_sgd else ()
            for p in model.parameters():
                reduced = p.grad.reduce(reduction_dims, reduction="avg")
                p.grad = reduced

            # clip the gradient after reduction
            if NanoGPTConfig.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), NanoGPTConfig.grad_clip
                )

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if use_local_sgd and (iter_num + 1) % NanoGPTConfig.local_steps == 0:
                # sync pseudo gradients to outer model
                for p_outer, p_local in zip(
                    outer_model.parameters(), model.parameters()
                ):
                    with torch.no_grad():
                        delta = p_outer - p_local
                        outer_grad = delta.reduce((), reduction="avg")
                    p_outer.grad = outer_grad
                # step the optimizer
                outer_optimizer.step()
                # flush the gradients as soon as we can, no need for this memory anymore
                outer_optimizer.zero_grad(set_to_none=True)
                # sync parameters from outer model
                model.load_state_dict(outer_model.state_dict())
        return loss

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if NanoGPTConfig.decay_lr else NanoGPTConfig.learning_rate
        with default_mesh.activate():
            lr_tensor.fill_(lr)

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % NanoGPTConfig.eval_interval == 0 and master_process:
            with default_mesh.activate():
                losses = estimate_loss()
                local_losses = fetch_shard(losses).result()
            print(
                f"step {iter_num}: train loss {local_losses['train']}, val loss {local_losses['val']}"
            )
            # TODO(rajeshn) get wandb logging & checkpointing working
            # if NanoGPTConfig.wandb_log:
            #     wandb.log({
            #         "iter": iter_num,
            #         "train/loss": losses['train'],
            #         "val/loss": losses['val'],
            #         "lr": lr,
            #         "mfu": running_mfu*100, # convert to percentage
            #     })
            # if losses['val'] < best_val_loss or NanoGPTConfig.always_save_checkpoint:
            #     best_val_loss = losses['val']
            #     if iter_num > 0:
            #         checkpoint = {
            #             'model': raw_model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'model_args': model_args,
            #             'iter_num': iter_num,
            #             'best_val_loss': best_val_loss,
            #             'config': {},
            #         }
            #         print(f"saving checkpoint to {NanoGPTConfig.out_dir}")
            #         torch.save(checkpoint, os.path.join(NanoGPTConfig.out_dir, 'ckpt.pt'))
        if iter_num == 0 and NanoGPTConfig.eval_only:
            break

        loss = step()
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % NanoGPTConfig.log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            # as with the reference implementation we only fetch the loss from the master_process which is rank 0
            with default_mesh.activate():
                local_loss = fetch_shard(loss).result()
                if not NanoGPTConfig.monarch_compile:
                    # collecting every iteration when compiled code
                    # sends very few commands adds overhead to
                    # walk all objects.
                    gc.collect()
            lossf = local_loss.item() * NanoGPTConfig.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(
                    NanoGPTConfig.batch_size
                    * NanoGPTConfig.gradient_accumulation_steps,
                    dt,
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > NanoGPTConfig.max_iters:
            break

    default_mesh.exit()

    return


if __name__ == "__main__":
    # run the training loop
    ret = main(sys.argv[1:])
    sys.exit(ret)
