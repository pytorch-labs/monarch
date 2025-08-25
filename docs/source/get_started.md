# Get Started

Welcome to Monarch! This guide will help you get up and running with Monarch, a distributed execution engine for PyTorch that delivers high-quality user experience at cluster scale.

## What is Monarch?

Monarch is designed to extend PyTorch's capabilities to efficiently run on distributed systems. It maintains the familiar PyTorch API while handling the complexities of distributed execution, making it easier to scale your deep learning workloads across multiple GPUs and nodes.

## Prerequisites

Before installing Monarch, ensure you have:

- A Linux system (Monarch is currently only supported on Linux)
- Python 3.10 or later
- CUDA-compatible GPU(s)
- Basic familiarity with PyTorch

## Installation

### Quick Installation

The simplest way to install Monarch is via pip:

```bash
pip install torchmonarch-nightly
```

### Manual Installation

For more control or development purposes, you can install Monarch manually:

```bash
# Create and activate the conda environment
conda create -n monarchenv python=3.10 -y
conda activate monarchenv

# Install nightly rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
rustup default nightly

# Install non-python dependencies
conda install libunwind -y

# Install the correct cuda and cuda-toolkit versions for your machine
sudo dnf install cuda-toolkit-12-0 cuda-12-0

# Install clang-dev and nccl-dev
sudo dnf install clang-devel libnccl-devel
# Or, in some environments, the following may be necessary instead
conda install -c conda-forge clangdev nccl
conda update -n monarchenv --all -c conda-forge -y

# Install build dependencies
pip install -r build-requirements.txt
# Install test dependencies
pip install -r python/tests/requirements.txt

# Build and install Monarch
pip install --no-build-isolation .
# or setup for development
pip install --no-build-isolation -e .
```

## Verifying Your Installation

After installation, you can verify that Monarch is working correctly by running the unit tests:

```bash
pytest python/tests/ -v -m "not oss_skip"
```

## Basic Usage

Here's a simple example to get you started with Monarch:

```python
import torch
import monarch as mon

# Initialize Monarch
mon.init()

# Create a simple model
model = torch.nn.Linear(10, 5)

# Distribute the model using Monarch
distributed_model = mon.distribute(model)

# Create some input data
input_data = torch.randn(8, 10)

# Run a forward pass
output = distributed_model(input_data)

# Clean up
mon.shutdown()
```

## Example: Ping Pong

One of the simplest examples of using Monarch is the "ping pong" example, which demonstrates basic communication between processes:

```python
import monarch as mon
import torch

# Initialize Monarch
mon.init()

# Get the current process rank and world size
rank = mon.get_rank()
world_size = mon.get_world_size()

# Create a tensor to send
send_tensor = torch.tensor([rank], dtype=torch.float32)

# Determine the destination rank
dst_rank = (rank + 1) % world_size

# Send the tensor to the destination rank
mon.send(send_tensor, dst_rank)

# Receive a tensor from the source rank
src_rank = (rank - 1) % world_size
recv_tensor = torch.zeros(1, dtype=torch.float32)
mon.recv(recv_tensor, src_rank)

print(f"Rank {rank} received {recv_tensor.item()} from rank {src_rank}")

# Clean up
mon.shutdown()
```

## Distributed Data Parallel Training

Monarch makes it easy to implement distributed data parallel training:

```python
import monarch as mon
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize Monarch
mon.init()

# Create a simple model
model = nn.Linear(10, 5)
model = mon.distribute(model)

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create loss function
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    # Assume data_loader is your distributed data loader
    for data, target in data_loader:
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Clean up
mon.shutdown()
```

## Next Steps

Now that you've got the basics, you can:

1. Check out the [Examples](./generated/examples/index) directory for more detailed demonstrations
2. Explore the [API documentation](rust-api) for a complete reference


## Troubleshooting

If you encounter issues:

- Make sure your CUDA environment is properly set up
- Check that you're using a compatible version of PyTorch
- Verify that all dependencies are installed correctly
- Consult the [GitHub repository](https://github.com/meta-pytorch/monarch) for known issues

Remember that Monarch is currently in an experimental stage, so you may encounter bugs or incomplete features. Contributions and bug reports are welcome!
