# pyre-unsafe
import logging
import os
import typing
from ast import literal_eval
from typing import Literal

import torch

from llama3.te_import import is_te_available

from monarch.common.remote import remote

logger = logging.getLogger(__name__)


class TrainConfig:
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    eval_interval = 2000
    log_interval = 1
    checkpoint_interval = 500
    eval_iters = 200
    eval_only = False  # if True, script exits right after the first eval

    init_from = "scratch"  # 'scratch' or 'resume' or filepath to checkpoint
    model_name = "toy"

    # Model args
    n_layer: int = 32
    n_head: int = 8
    n_local_heads: int = 1
    dim: int = 512
    intermediate_size: int = 2048
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    norm_eps: float = 1e-5
    rope_base: float = 500000
    use_te = is_te_available()

    # local_rank for perf baseline test set through args in torch.distributed.launch as opposed to LOCAL_RANK env var
    local_rank: int | None = None

    # data
    data_root_dir: str | None = "./llama3/data/"
    dataset: str | None = None
    data_dir: str | None = None
    checkpoint_dir: str | None = None
    vocab_size: int = 128256
    batch_size: int = (
        12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )
    block_size: int = 8192
    # dataloading
    xlformers_data = None
    xlformers_tokenizer = None
    random_data: bool = False

    # adamw optimizer
    optimizer = "adamw"
    learning_rate = 6e-4  # max learning rate
    max_iters = 200  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: torch.dtype = torch.bfloat16

    # distributed training
    n_gpus = 1
    n_hosts = 1
    pp = 1
    tp = 1
    dp = 1

    # mesh type
    mesh_type: Literal[
        "rust_local", "python", "simulator", "rust_test", "rust_mast"
    ] = "rust_local"
    use_monarch = True

    seed = 1337

    # utilities
    dump_dir = os.environ.get("DUMP_DIR", None)
    tensorboard = "" if dump_dir is None else f"{dump_dir}/tb"
    memory_profile = ""

    @classmethod
    def _configureCLIArgs(cls, args):
        for arg in args:
            if "=" not in arg:
                # assume it's the name of a config file
                assert not arg.startswith("--")
                config_file = arg
                logging.info(f"Overriding config with {config_file}:")
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
                        logging.info(f"Adding: {key} = {val}")
                        setattr(cls, key, val)
            else:
                # assume it's a --key=value argument
                assert arg.startswith("--")
                key, val = arg.split("=")
                key = key[2:].replace("-", "_")
                if key in cls.__dict__:
                    try:
                        # attempt to eval it it (e.g. if bool, number, or etc)
                        attempt = literal_eval(val)
                    except (SyntaxError, ValueError):
                        # if that goes wrong, just use the string
                        attempt = val
                    annotations = TrainConfig.__annotations__
                    if annotations.get(key, None) is not None:
                        annotation_type = annotations[key]
                        if typing.get_origin(annotation_type) == typing.Literal:
                            literal_args = typing.get_args(annotation_type)
                            assert (
                                attempt in literal_args
                            ), f"Value {attempt} not in allowed values {literal_args} for {key} = {val}"
                        else:
                            assert isinstance(
                                attempt, annotation_type
                            ), f"Type attempt {type(attempt)} != type {annotation_type} for {key} = {val}"
                    else:
                        # ensure the types match ok
                        assert (
                            type(attempt) == type(cls.__dict__[key])
                        ), f"Type attempt {type(attempt)} != type {type(cls.__dict__[key])} for {key} = {val}"
                    # cross fingers
                    logging.info(f"Overriding: {key} = {attempt}")
                    setattr(cls, key, attempt)
                else:
                    raise ValueError(f"Unknown config key: {key}")

        if getattr(cls, "use_te") and not is_te_available():  # noqa
            raise Exception(
                "Please install transformer_engine to run llama8b or set config locally!"
            )

    @classmethod
    def configure(cls, args):
        cls._configureCLIArgs(args)

        # configure the derived variables
        cls.device_type = (
            "cuda" if "cuda" in cls.device else "cpu"
        )  # for later use in torch.autocast

        if cls.data_root_dir is not None and cls.dataset is not None:
            cls.data_dir = os.path.join(cls.data_root_dir, cls.dataset)

        # Provide a way to specify None for the xlformer tokenizer and data
        if cls.xlformers_data is not None and len(cls.xlformers_data) == 0:
            cls.xlformers_data = None
        if cls.xlformers_tokenizer is not None and len(cls.xlformers_tokenizer) == 0:
            cls.xlformers_tokenizer = None

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


reconfigure_worker = remote("llama3.config._reconfigure_impl", propagate="inspect")


def _reconfigure_impl(args):
    logging.info(f"reconfiguring with {args}")
    TrainConfig.configure(args)
