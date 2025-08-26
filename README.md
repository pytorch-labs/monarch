# Monarch 🦋

**Monarch** is a distributed execution engine for PyTorch. Our overall goal is
to deliver the high-quality user experience that people get from single-GPU
PyTorch, but at cluster scale.

> ⚠️ **Early Development Warning** Monarch is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

Note: Monarch is currently only supported on Linux systems

## 📖 Documentation

View Monarch's hosted documentation [at this link](https://meta-pytorch.org/monarch/).

## Installation

### On Fedora distributions

`pip install torchmonarch-nightly`

or manually

```sh

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
# Or, in some envrionments, the following may be necessary instead
conda install -c conda-forge clangdev nccl
conda update -n monarchenv --all -c conda-forge -y

# If you are building with RDMA support, build monarch with `USE_TENSOR_ENGINE=1 pip install --no-build-isolation .` and dnf install the following packages
sudo dnf install -y libibverbs rdma-core libmlx5 libibverbs-devel rdma-core-devel

# Install build dependencies
pip install -r build-requirements.txt
# Install test dependencies
pip install -r python/tests/requirements.txt

# Build and install Monarch
pip install --no-build-isolation .
# or setup for development
pip install --no-build-isolation -e .

# Run unit tests. consider -s for more verbose output
pytest python/tests/ -v -m "not oss_skip"
```

### On Ubuntu distributions

```sh
# Clone the repository and navigate to it
git clone https://github.com/pytorch-labs/monarch.git
cd monarch

# Install nightly rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup toolchain install nightly
rustup default nightly

# Install Ubuntu-specific system dependencies
sudo apt install -y ninja-build
sudo apt install -y libunwind-dev
sudo apt install -y clang

# Set clang as the default C/C++ compiler
export CC=clang
export CXX=clang++

# Install build dependencies
pip install -r build-requirements.txt
# Install test dependencies
pip install -r python/tests/requirements.txt

# Build and install Monarch (with tensor engine support)
pip install --no-build-isolation .

# or
# Build and install Monarch (without tensor engine support)
USE_TENSOR_ENGINE=0 pip install --no-build-isolation .

# or setup for development
pip install --no-build-isolation -e .

# Verify installation
pip list | grep monarch
```

### On MacOS

You can also build Monarch to run locally on a MacOS system.

Note that this does not support tensor engine, which is tied to CUDA and RDMA (via ibverbs).


```sh

# Create and activate the conda environment
conda create -n monarchenv python=3.10 -y
conda activate monarchenv

# Install nightly rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
rustup default nightly

# Install build dependencies
pip install -r build-requirements.txt
# Install test dependencies
pip install -r python/tests/requirements.txt

# Build and install Monarch
USE_TENSOR_ENGINE=0 pip install --no-build-isolation .
# or setup for development
USE_TENSOR_ENGINE=0 pip install --no-build-isolation -e .

```


## Running examples

Check out the `examples/` directory for demonstrations of how to use Monarch's APIs.

We'll be adding more examples as we stabilize and polish functionality!

## License

Monarch is BSD-3 licensed, as found in the [LICENSE](LICENSE) file.
