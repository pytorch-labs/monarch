Monarch Examples
================

Welcome to Monarch's examples! This section contains various examples demonstrating how to use Monarch for distributed execution in PyTorch.

Jupyter Notebooks
----------------

These interactive tutorials demonstrate key features and use cases of Monarch, helping you understand how to leverage distributed execution for your PyTorch workloads.

- **Ping Pong**: A simple demonstration of basic communication between processes in a distributed setting. This example shows how to send and receive tensors between different ranks, illustrating the fundamental building blocks of distributed computing.

- **SPMD DDP**: An implementation of Single Program Multiple Data (SPMD) and Distributed Data Parallel (DDP) training with Monarch. This notebook shows how to scale your PyTorch models across multiple GPUs and nodes for faster training.

Each notebook contains detailed explanations, code snippets, and comments to guide you through the implementation.

.. toctree::
    :maxdepth: 1
    :caption: Examples Notebooks

    examples/notebooks/ping_pong
    examples/notebooks/spmd_ddp

Python Examples
--------------

These Python scripts demonstrate how to use Monarch's APIs directly in Python code:

- **Grpo Actor**: :doc:`examples/grpo_actor`
