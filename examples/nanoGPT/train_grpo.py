# pyre-unsafe
"""
1. This code demonstrates the use of the GRPO (Generative Reward Policy Optimization) algorithm to
    perform reinforcement learning on a policy model, specifically a nanoGPT model. The reward
    function is defined to assign a random value to each sample.

2. Some of the nanoGPT model creation and training loop code is derived from
    https://github.com/karpathy/nanoGPT/blob/master/train.py
To run on a single GPU, example:
$ cd fbsource/fbcode/monarch/examples
python3 -m nanoGPT.train_grpo --model_config=./nanoGPT/config/train_shakespeare_char_small_grpo.py

"""

import argparse
import logging
import math
import os

import pickle
import time
from contextlib import nullcontext

import torch

from monarch import fetch_shard, local_mesh, no_mesh
from monarch_supervisor.logging import initialize_logging

from nanoGPT.config import NanoGPTConfig
from nanoGPT.data_loader import DataLoaderConfig, get_batch_local_no_pipe
from nanoGPT.grpo import GRPO, GRPOConfig
from nanoGPT.model import GPT, GPTConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def reward_func_1(prompts, completions):
    """
    Generates a random float tensor with values between 0.0 and 1.0 as the reward for each completion.

    Args:
        prompts: A list of input prompts.
        completions: A list of generated completions for which random rewards are generated.


    Returns:
        A tensor containing random float values between 0.0 and 1.0 for each completion.
    """
    # Generate a random float tensor with values between 0.0 and 1.0
    result = []
    for _ in completions:
        random_tensor = torch.rand(1)
        result.append(random_tensor)
    return torch.stack(result, dim=0)


def prepare_encoder_decoder(meta_path):
    """
    Prepares the encoder and decoder functions based on the metadata file.

    Args:
        meta_path (str): The path to the metadata file containing the encoder and decoder mappings.

    Returns:
        encode: A function for converting input prompt strings to token index lists.
        decode: A functionfor converting model output token index lists to strings.

    Note:
        To obtain the metadata file, run the prepare script:
        e.g., python data/shakespeare_char/prepare.py
        The meta file can then be found under the same data directory.
    """
    logger.info(f"Looking for meta.pkl in {meta_path}")
    if not os.path.exists(meta_path):
        raise Exception("meta.pkl not found")

    logger.info(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(s):
        return [stoi[c] for c in s]

    def decode(lst):
        return "".join([itos[i] for i in lst])

    return encode, decode


def main():
    parser = argparse.ArgumentParser(description="train GRPO rgparser")
    parser.add_argument(
        "--model_config",
        type=str,
        default="./nanoGPT/config/train_shakespeare_char_small_grpo.py",
        help="configurate the policy model",
    )
    temperature = 0.7
    top_k = 200
    max_prompt_length = 64
    max_completion_length = 16
    num_generations = 4
    args = parser.parse_args()

    initialize_logging()
    world_size = 2
    orig_mesh = local_mesh(hosts=1, gpus=world_size)
    mesh = orig_mesh.flatten("gpu").split(
        gpu=(
            "dp",
            "n_models",
        ),
        n_models=world_size,
    )
    policy_mesh = mesh(n_models=0)
    ref_mesh = mesh(n_models=1)

    torch.manual_seed(23456)

    torch.set_default_device("cuda")

    with policy_mesh.activate():
        (
            model,
            ref_model,
            optimizer,
            model_args,
            ctx,
            scaler,
            iter_num,
            best_val_loss,
            data_loader_config,
        ) = create_nano_gpt(
            [args.model_config],
            policy_mesh=policy_mesh,
            ref_mesh=ref_mesh,
        )

        assert max_prompt_length + max_completion_length <= model.config.block_size

        meta_path = os.path.join(NanoGPTConfig.out_dir, "meta.pkl")
        encode, decode = prepare_encoder_decoder(meta_path)
        master_process = True

        grpo_confg = GRPOConfig(
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            beta=0.01,
            temperature=temperature,
            top_k=top_k,
        )

        grpo_trainer = GRPO(
            model=model,
            ref_model=ref_model,
            args=grpo_confg,
            encoder=encode,
            decoder=decode,
            reward_funcs=[reward_func_1],
            device=NanoGPTConfig.device,
            mesh=mesh,
            policy_mesh=policy_mesh,
            ref_mesh=ref_mesh,
        )

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

    with policy_mesh.activate():
        # training loop
        data_loader_config.block_size = max_prompt_length
        with no_mesh.activate():
            x_local, _ = get_batch_local_no_pipe(
                "train", data_loader_config
            )  # fetch the very first batch
            x_value = x_local.detach().cpu().numpy()
        with policy_mesh.activate():
            X = torch.tensor(x_value)
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model  # unwrap DDP container if needed
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = (
                get_lr(iter_num)
                if NanoGPTConfig.decay_lr
                else NanoGPTConfig.learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with ctx:
                (
                    loss,
                    _,
                    _,
                ) = grpo_trainer.prediction_step(X)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # X, _ = get_batch_local_no_pipe("train", data_loader_config)
            with no_mesh.activate():
                x_local, _ = get_batch_local_no_pipe(
                    "train", data_loader_config
                )  # fetch the very first batch
                x_value = x_local.detach().cpu().numpy()
            with policy_mesh.activate():
                X = torch.tensor(x_value)

            # backward pass, with gradient scaling if training in fp16. If cuda is mocked, we need to disable
            # multithreading for backward. Otherwise, the backward ops will execute on a different thread that
            # doesn't have cuda mocking enabled, leading to an invalid memory access and a crash.
            with torch.autograd.set_multithreading_enabled(True):
                scaler.scale(loss).backward()
            # clip the gradient
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
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            grpo_trainer.update_reference_model()

            if iter_num % NanoGPTConfig.log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                # lossf = loss.item() * NanoGPTConfig.gradient_accumulation_steps
                loss_local = fetch_shard(loss).result()
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(
                        NanoGPTConfig.batch_size
                        * NanoGPTConfig.gradient_accumulation_steps,
                        dt,
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {iter_num}: loss {loss_local.item():.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > NanoGPTConfig.max_iters:
                break
    orig_mesh.exit()
    orig_mesh.deactivate()


def create_nano_gpt(config_file, policy_mesh=None, ref_mesh=None):
    NanoGPTConfig.configure(config_file)
    # various inits, derived attributes, I/O setup
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = (
        NanoGPTConfig.gradient_accumulation_steps
        * ddp_world_size
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

    # model init
    model_args = dict(
        n_layer=NanoGPTConfig.n_layer,
        n_head=NanoGPTConfig.n_head,
        n_embd=NanoGPTConfig.n_embd,
        block_size=NanoGPTConfig.block_size,
        bias=NanoGPTConfig.bias,
        vocab_size=None,
        dropout=NanoGPTConfig.dropout,
    )  # start with model_args from command line
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
        gptconf = GPTConfig(**model_args)
        with policy_mesh.activate():
            model = GPT(gptconf)
        with ref_mesh.activate():
            ref_model = GPT(gptconf)
    elif NanoGPTConfig.init_from == "resume":
        print(f"Resuming training from {NanoGPTConfig.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(NanoGPTConfig.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=NanoGPTConfig.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif NanoGPTConfig.init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {NanoGPTConfig.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=NanoGPTConfig.dropout)
        model = GPT.from_pretrained(
            NanoGPTConfig.init_from, NanoGPTConfig.override_args
        )
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if NanoGPTConfig.block_size < model.config.block_size:
        model.crop_block_size(NanoGPTConfig.block_size)
        model_args["block_size"] = (
            NanoGPTConfig.block_size
        )  # so that the checkpoint will have the right value
    model.to(NanoGPTConfig.device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(NanoGPTConfig.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        NanoGPTConfig.weight_decay,
        NanoGPTConfig.learning_rate,
        (NanoGPTConfig.beta1, NanoGPTConfig.beta2),
        NanoGPTConfig.device_type,
    )
    if NanoGPTConfig.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    data_loader_config = DataLoaderConfig.from_config(NanoGPTConfig)
    return (
        model,
        ref_model,
        optimizer,
        model_args,
        ctx,
        scaler,
        iter_num,
        best_val_loss,
        data_loader_config,
    )


if __name__ == "__main__":
    # run the training loop
    main()
