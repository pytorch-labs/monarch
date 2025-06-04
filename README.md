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

## Installation

`pip install torchmonarch`

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

# Install the correct cuda and cuda-toolkit versions for your machine, as well as NCCL-dev
sudo dnf install cuda-toolkit-12-0 cuda-12-0 libnccl-devel

# Install clang dev
sudo dnf install clang-devel
# In some envrionments, the following may be necessary instead
conda install conda-forge::clangdev
conda update -n monarchenv --all -c conda-forge -y

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

## Running examples

Check out the `examples/` directory for demonstrations of how to use Monarch's APIs.

We'll be adding more examples as we stabilize and polish functionality!

## License

Monarch is BSD-3 licensed, as found in the [LICENSE](LICENSE) file.
