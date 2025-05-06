# pyre-unsafe
from dataclasses import dataclass
from typing import Literal, Optional

from morpho.debug.data import test_dataset, test_tokenizer


class DefaultProperty:
    def __get__(self, obj, klass):
        return klass()


@dataclass(frozen=True)
class _Default:
    default = DefaultProperty()


@dataclass(frozen=True)
class ProfilingConfig(_Default):
    enable_profiling: bool = False
    save_traces_folder: str = "profile_traces"
    profile_freq: int = 10
    enable_memory_snapshot: bool = False
    save_memory_snapshot_folder: str = "memory_snapshot"


@dataclass(frozen=True)
class MetricsConfig(_Default):
    log_freq: int = 1
    enable_tensorboard: bool = False
    disable_color_printing: bool = False
    save_tb_folder: str = "tb"
    rank_0_only: bool = True
    enable_wandb: bool = False


# note: imported directly from titan
# most options are not implemented
@dataclass(frozen=True)
class CheckpointConfig(_Default):
    enable_checkpoint: bool = False
    folder: str = "checkpoint"
    interval_type: str = "steps"
    interval: int = 500
    model_weights_only: bool = False
    export_dtype: str = "float32"
    create_seed_checkpoint: bool = False
    async_mode: str = "disabled"
    keep_latest_k: int = 0
    load_step: int = -1


@dataclass(frozen=True)
class JobConfig(_Default):
    """
    Options related to how the job displays information, logs metrics,
    and where it loads and stores outputs like checkpoints.

    These options should not affect the way the model is trained or parallelized.
    """

    profiling: ProfilingConfig = ProfilingConfig()
    metrics: MetricsConfig = MetricsConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()


@dataclass(frozen=True)
class TrainingConfig(_Default):
    batch_size: int = 8
    seq_len: int = 2048
    steps: int = 10
    warmup_steps: int = 2  # lr scheduler warm up, normally 20% of the train steps
    max_norm: float = 1.0  # grad norm clipping
    # supported datasets: c4_test (2K), c4 (177M)
    dataset: Literal["c4_test", "c4"] = "c4_test"
    dataset_path: str = test_dataset
    seed: int = 4  # https://xkcd.com/221/
    deterministic: bool = True
    lr: float = 8e-4


@dataclass(frozen=True)
class MachineConfig(_Default):
    ngpu: int = 8
    visible_gpus: Optional[str] = None


Variant = Literal["reference", "reassociated", "noise", "experiment"]


@dataclass(frozen=True)
class DebugConfig(_Default):
    run: str = "unnamed"
    variant: Variant = "reference"
    sample: int = 0
    observe: bool = False


@dataclass(frozen=True)
class ModelConfig(_Default):
    name: Literal["llama3"] = "llama3"
    flavor: Literal["debugmodel", "8B", "70B", "405B"] = "debugmodel"
    norm_type: Literal["layernorm", "np_layernorm", "rmsnorm", "fused_rmsnorm"] = (
        "rmsnorm"
    )
    tokenizer_path: str = test_tokenizer


@dataclass(frozen=True)
class ParallelismConfig(_Default):
    ac_freq: int = 2
