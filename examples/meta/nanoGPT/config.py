# pyre-unsafe
import os
from ast import literal_eval

import torch

from monarch.common.remote import remote


# the orignal configuration for nanogpt is a mess. this tries to put that mess into a box but i still
# don't like it.
class NanoGPTConfig:
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir = "out"
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False  # disabled by default
    wandb_project = "owt"
    wandb_run_name = "gpt2"  # 'run' + str(time.time())
    # data
    # default the root dir assuming you are going to run from fbcode root
    random_data: bool = False
    data_root_dir = "./monarch/examples/nanoGPT/data"
    dataset = "openwebtext"
    data_vocab_size = (
        -1
    )  # -1 means default, either get from pkl file or use gpt2 vocab. Can set manually
    gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
    batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 1024
    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False  # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    max_iters = 600000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # outer optimizer
    local_steps = 0
    local_group_size = 2
    outer_optim_type = "nesterov"
    outer_optim_weight_decay = 0.1
    outer_optim_lr = 1
    out_optim_grad_clip = 1.0
    outer_optim_momentum = 0.9

    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    n_gpus = 1
    n_hosts = 1
    seed = 1337

    mocked = False
    monarch_compile = False

    device_type = "cpu"
    data_dir = ""
    override_args = None
    master_process = None

    @classmethod
    def _configureCLIArgs(cls, args):
        for arg in args:
            if "=" not in arg:
                # assume it's the name of a config file
                assert not arg.startswith("--")
                config_file = arg
                print(f"Overriding config with {config_file}:")
                locals_before = locals().copy()
                with open(config_file) as f:
                    exec(f.read())
                new_locals = locals()
                for key, val in new_locals.items():
                    if (
                        key not in locals_before
                        and key != "locals_before"
                        and not key.startswith("__")
                    ):
                        print(f"Adding: {key} = {val}")
                        setattr(cls, key, val)
            else:
                # assume it's a --key=value argument
                assert arg.startswith("--")
                key, val = arg.split("=")
                key = key[2:]
                if key in cls.__dict__:
                    try:
                        # attempt to eval it it (e.g. if bool, number, or etc)
                        attempt = literal_eval(val)
                    except (SyntaxError, ValueError):
                        # if that goes wrong, just use the string
                        attempt = val
                    # ensure the types match ok
                    assert (
                        type(attempt) == type(cls.__dict__[key])
                    ), f"Type attempt {type(attempt)} != type {type(cls.__dict__[key])} for {key} = {val}"
                    # cross fingers
                    print(f"Overriding: {key} = {attempt}")
                    setattr(cls, key, attempt)
                else:
                    raise ValueError(f"Unknown config key: {key}")

    @classmethod
    def configure(cls, args):
        cls._configureCLIArgs(args)

        # configure the derived variables
        cls.device_type = (
            "cuda" if "cuda" in cls.device else "cpu"
        )  # for later use in torch.autocast
        cls.data_dir = (
            os.path.join(cls.data_root_dir, cls.dataset)
            if len(cls.data_root_dir) > 0
            else ""
        )
        torch.manual_seed(cls.seed)
        world_size = cls.n_gpus * cls.n_hosts
        assert cls.gradient_accumulation_steps % world_size == 0
        cls.gradient_accumulation_steps //= world_size

        if cls.local_steps > 0:
            assert (
                cls.init_from == "scratch"
            ), "only from scratch is supported for local_sgd"
            assert cls.local_group_size % cls.n_gpus == 0
            assert world_size % cls.local_group_size == 0

    @classmethod
    def getConfigDict(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    @classmethod
    def __str__(cls):
        return str(cls.getConfigDict())


reconfigure_worker = remote("nanoGPT.config._reconfigure_impl", propagate="inspect")


def _reconfigure_impl(args):
    print(f"reconfiguring with {args}")
    NanoGPTConfig.configure(args)
