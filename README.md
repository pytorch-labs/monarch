# Monarch

**Monarch** is a distributed execution engine for PyTorch.

> ⚠️ **Early Development Warning** Monarch is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

## Installation

pip install torchmonarch

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
conda install python=3.10
conda install libunwind

# needs cuda-toolkit-12-0 as that is the version that matches the /usr/local/cuda/ on devservers
sudo dnf install cuda-toolkit-12-0 cuda-12-0 libnccl-devel clang-devel
# install build dependencies
pip install setuptools-rust
# install torch, can use conda or build it yourself or whatever
pip install torch
# install core deps, see pyproject.toml for latest
pip install pyzmq requests numpy pyre-extensions cloudpickle
# Install test dependencies
pip install pytest pytest-timeout pytest-asyncio

# install the package
python setup.py install
# or setup for development
python setup.py develop

# run unit tests. consider -s for more verbose output
pytest python/tests/ -v -m "not oss_skip"
```

## Running examples

grpo_actor.py showcases using Monarch's asynchronous actor model for PPO
`python examples/grpo_actor.py`

## Debugging

If everything is hanging, set the environment
`CONTROLLER_PYSPY_REPORT_INTERVAL=10` to get a py-spy dump of the controller and
its subprocesses every 10 seconds.

Calling `pdb.set_trace()` inside a worker remote function will cause pdb to
attach to the controller process to debug the worker. Keep in mind that if there
are multiple workers, this will create multiple sequential debug sessions for
each worker.

For the rust based setup you can adjust the log level with
`RUST_LOG=<log level>` (eg. `RUST_LOG=debug`).

## Profiling

The `monarch.profiler` module provides functionality similar to
[PyTorch's Profiler](https://pytorch.org/docs/stable/profiler.html) for model
profiling. It includes `profile` and `record_function` methods. The usage is
generally the same as `torch.profiler.profile` and
`torch.profiler.record_function`, with a few modifications specific to
`monarch.profiler.profile`:

1. `monarch.profiler.profile` exclusively accepts `monarch.profiler.Schedule`, a
   dataclass that mimics `torch.profiler.schedule`.
2. The `on_trace_ready` argument in `monarch.profiler.profile` must be a string
   that specifies the directory where the worker should save the trace files.

Below is an example demonstrating how to use `monarch.profiler`:

```py
    from monarch.profiler import profile, record_function
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready="./traces/",
        schedule=monarch.profilerSchedule(wait=1, warmup=1, active=2, repeat=1),
        record_shapes=True,
    ) as prof:
        with record_function("forward"):
            loss = model(batch)

        prof.step()
```

## Memory Viewer

The `monarch.memory` module provides functionality similar to
[PyTorch's Memory Snapshot and Viewer](https://pytorch.org/docs/stable/torch_cuda_memory.html)
for visualizing and analyzing memory usage in PyTorch models. It includes
`monarch.memory.dump_memory_snapshot` and `monarch.memory.record_memory_history`
methods:

1. `monarch.memory.dump_memory_snapshot`: This function wraps
   `torch.cuda.memory._dump_snapshot()` to dump memory snapshot remotely. It can
   be used to save a snapshot of the current memory usage to a file.
2. `monarch.memory.record_memory_history`: This function wraps
   `torch.cuda.memory_record_memory_history()` to allow recording memory history
   remotely. It can be used to track memory allocation and deallocation over
   time.

Both functions use `remote` to execute the corresponding remote functions
`_memory_controller_record` and `_memory_controller_dump` on the specified
device mesh.

Below is an example demonstrating how to use `monarch.memory`:

```py
    ...
    monarch.memory.record_memory_history()
    for step in range(2):
        batch = torch.randn((8, DIM))
        loss = net(batch)
        ...
    monarch.memory.dump_memory_snapshot(dir_snapshots="./snapshots/")
```

## License

Monarch is BSD-3 licensed, as found in the [LICENSE](LICENSE) file.
