# Documentation

See the source documentation or `bunnylol rustdoc torch-sys` to see docs.

# Cargo build

The cargo build requires that you have a version of PyTorch installed in your
Python environment. To get set up, run the following on your devgpu:

```sh
# get conda on devserver
sudo feature install genai_conda

# Set up conda env
conda create -n monarch
conda activate monarch

# install pytorch
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia

# install cuda toolkit on devserver (requires devgpu)
sudo dnf install cuda-12-0

# install nccl on devserver (requires devgpu)
sudo dnf install libnccl-devel

# install libclang on devserver (needed for rust-bindgen)
sudo dnf install clang-devel

# in monarch/torch-sys
cargo test
```
