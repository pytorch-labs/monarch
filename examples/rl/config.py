from dataclasses import dataclass


@dataclass
class Config:
    # ---- Experiment level configs
    # the name of this experiment
    exp_name: str = "toy"
    log_n_steps: int = 5
    replay_buffer_size: int = 20

    # ---- Monarch Mesh configs
    # the Monarch mesh type to use
    mesh_type: str = "local"  # "local", "rust_mast", "simulator"

    # TODO - add more flexibility for multiple DeviceMeshes
    num_generators: int = 4

    # ---- Hyperparameters
    num_steps: int = 500
    seed: int = 42
    learning_rate: float = 1e-3
    epsilon: float = 1e-5

    # Assume input is a prompt, output is a response
    prompt_length: int = 8
    response_length: int = 8
    input_shape: int = 128
    output_shape: int = 128
    model_dim: int = 1024
    model_depth: int = 4
    batch_size: int = 4

    enable_aix: bool = False
