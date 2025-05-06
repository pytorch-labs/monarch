# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from dataclasses import dataclass


@dataclass
class GRPOTrainerConfig:
    path: str = "post_training.grpo.grpo_trainer.GRPOTrainer"
    name: str = "grpo_trainer"
    num_processes: int = 1

    # model configs
    model_type: str = "test"
    init_param_value: float = 0.0
    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 4

    # optimizer configs
    lr = 1.0
    delta = 0.1

    # parallelism configs
    pipeline_parallel_size: int = 1
    model_parallel_size: int = 1


@dataclass
class GRPOGeneratorConfig:
    path: str = "post_training.grpo.grpo_generator.GRPOGenerator"
    name: str = "grpo_generator"
    num_processes: int = 1

    # model config
    model_type: str = "test"
    init_param_value: float = 0.0
    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 4
    max_output_size: int = 2048
    # algorithm config
    max_prompt_length: int = 64
    max_completion_length: int = 128
    num_generations: int = 4
    beta: float = 0.01
    temperature: float = 0.7
    top_k: int = 100

    # algorithm config
    max_prompt_length: int = 64
    max_completion_length: int = 128
    num_generations: int = 4
    beta: float = 0.01
    temperature: float = 0.7
    top_k: int = 100


@dataclass
class GRPOConfig:
    # Monarch configs
    multi_mesh: bool = True

    steps: int = 5
    save_freq: int = 0
    trainer_config: GRPOTrainerConfig = GRPOTrainerConfig()
    generator_config: GRPOGeneratorConfig = GRPOGeneratorConfig()
