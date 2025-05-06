# Getting started

## Installation

Follow
https://www.internalfb.com/wiki/Monarch/Monarch_xlformers_integration/Running_Monarch_on_Conda#how-to-run-monarch

## Development

If you are developing against a standard conda environment, we suggest
https://www.internalfb.com/wiki/Monarch/Monarch_xlformers_integration/Running_Monarch_on_Conda#install-editable-monarch

If you are developing against a different conda env (e.g. one you made
yourself), you can set your env up for a monarch build with the following
instructions:

```sh
# These instructions assume you have a devserver (and need to use fwdproxy to access the internet)

# install conda and set it up
feature install genai_conda && conda-setup

# Install nightly rust toolchain
curl $(fwdproxy-config curl) --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sed 's#curl $_retry#curl $(fwdproxy-config curl) $_retry#g' | env $(fwdproxy-config --format=sh curl) sh

with-proxy rustup toolchain install nightly
rustup default nightly

# Install non-python dependencies
with-proxy conda install python=3.10
# needs cuda-toolkit-12-0 as that is the version that matches the /usr/local/cuda/ on devservers
sudo dnf install cuda-toolkit-12-0 libnccl-devel clang-devel
# install build dependencies
with-proxy pip install setuptools-rust
# install torch, can use conda or build it yourself or whatever
with-proxy pip install torch
# install other deps, see pyproject.toml for latest
with-proxy pip install pyzmq requests numpy pyre-extensions
```

## Running examples

### Basic Monarch features

In the `examples` folder under `fbcode/monarch`, there are handful of Monarch
examples. `controller/example.py` is a good starting point to understand Monarch
basic features. Run the following command to launch the example. It is also
recommended to run the following file in Bento by selecting the `monarch` Bento
kernel. In both cases, cli or Bento, it will launch Monarch processes locally.

```sh
MESH_TYPE=RUST_LOCAL python examples/controller/example.py
```

### Train nanoGPT

`examples/nanoGPT` folder contains the nanoGPT example. To train nanoGPT, we
need to prepare dataset first with the following command. Run with ssh proxy if
on devserver.

```sh
cd examples/nanoGPT && python data/shakespeare_char/prepare.py
```

Train nanoGPT with buck

```sh
cd examples/nanoGPT
buck run @//mode/opt :train -- ./config/train_shakespeare_char_small.py --eval_only=False --data_root_dir=./data/ --n_gpus=1
```

Train nanoGPT with conda

```sh
cd examples
python3 -m nanoGPT.train ./nanoGPT/config/train_shakespeare_char_small.py --eval_only=False --data_root_dir=./nanoGPT/data/ --n_gpus=1
```

### Train Llama3 locally

Train llama3 locally (more options are in the llama3 README
https://fburl.com/code/rwqzfill). Need to populate the shakespeare dataset
similar to nanoGPT.

```sh
cd examples
python llama3/data/shakespeare/prepare.py
python -m llama3.train ./llama3/configs/llama8b_local.py --mesh_type=rust_local
```

### Train Llama4 locally or on MAST

We support training Llama4 on Monarch both locally and on MAST. Due to the
complex dependencies from Llama4, we launch Llama4 training jobs only through
xlformers branches. Following instruction on xlformers branch:
https://www.internalfb.com/wiki/Monarch/Monarch_xlformers_integration/Running_Monarch_on_Conda/#launching-on-mast

## Connect MAST job from Bento

Connect to Rust backend on MAST from non-MAST environment (devgpu, Bento, etc.).
To launch a MAST job, it is always recommended to start with Monarch's conda
environment. Follow
https://www.internalfb.com/wiki/Monarch/Monarch_xlformers_integration/Running_Monarch_on_Conda/#launching-on-mast
to install your conda environment and check out `llama4_monarch` branch as your
starting point. Launch an Monarch job without any script

```sh
# Launch MAST jobs with Rust backend
cd examples
torchx run --scheduler_args "enableLegacy=True,localityConstraints=region;pci,rmAttribution=gen_ai_rf_nextgen_infra,conda_fbpkg_id='',conda_path_in_fbpkg=." core/monarch/mast.py:train --nodes=2 --env="NUM_GPUS=8" --enable_ttls=True --sweep=projects/pretrain/sweeps/monarch/reduced_monarch.yaml --monarch=stable --script=""

# Connect to the Monarch cluster in your Python script (e.g., Bento) using `rust_mast_mesh`.
# For example, `mesh = rust_mast_mesh(job_name='monarch-btcdxz2q', hosts=2, gpus=8)`
# `hosts` and `gpus` should match `--nodes` and `--nproc_per_node` in the launch script above.
```

## Debugging

If everything is hanging, set the environment
`CONTROLLER_PYSPY_REPORT_INTERVAL=10` to get a py-spy dump of the controller and
its subprocesses every 10 seconds.

Calling `pdb.set_trace()` inside a worker remote function will cause pdb to
attach to the controller process to debug the worker. Keep in mind that if there
are multiple workers, this will create multiple sequential debug sessions for
each worker.

[GILWatcher](https://www.internalfb.com/intern/wiki/Python/Helpful_Libraries/GIL_Watcher/)
is a tool that will snapshot thread and python process stack traces if the GIL
is not released for too long. It can be helpful debugging the python-based
controller. See D60989783 for how to use. Recommend using aiplatform API.

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

## Simulator

To use the simulator, simply import `SimulatorBackend` and use it as the backend
when initializing the device mesh. A trace.json will be created after `Exit` is
called. The trace.json can be viewed via
[Perfetto](https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html)
or chrome trace viewer. Two import differences need to be done:

1. `Exit` must be called so that the simulator knows when to dump the trace.
2. The simulator will only record 10,000 commands. So when using
   SimulatorBackend, it is recommended to train only 1-2 iterations.

## Creating a Device Mesh with MAST Workers

### Reserving the Workers

First, make sure monarch is installed in your conda environment as a
non-editable package, e.g.
https://www.internalfb.com/wiki/Monarch/Monarch_xlformers_integration/Running
//monarhch/oss_ci:

Then, to reserve `N` mast nodes that can be used as a remote device mesh
communicating with your devgpu, activate your conda environment with monarch and
related dependencies installed, then run:

```sh
python -m monarch.notebook reserve --hosts N
```

Use `python -m monarch.notebook reserve --help` to see the full list of flags.

### Enabling OILFS for Syncing from DevGPU to MAST (Optional)

To enable OILFS on your devgpu so that files can be synced in real time to your
mast workers, follow the instructions in
[this wiki](https://fburl.com/wiki/dctvirdh) in the sections "Ensure you have an
oilfs package with the rmdir fix" and "Setup mount on your devgpu / devvm". For
the final step when you run the `oilfs` CLI, surround the value passed to the
`--extra-flags` flag with quotes.

Upon success, you will have an OILFS-mouned directory on your devgpu at
`~/fuse-aidev` corresponding to the warm storage directory
`ws://ws.ai.{ws-region}/checkpoint/infra/aidev/$USER`. When you start your mast
job with `monarch.notebook reserve`, pass
`--oilfs_workspace_dir "/path/relative/to/fuse-aidev/"` to the command. Your
mast job _must_ be in the same region as your OILFS mount, otherwise your
workers will fail.

Then, if you have a python file at `~/fuse-aidev/monarch/udf.py` with a function
`my_udf`, and you started your mast job with `--oilfs_workspace_dir "monarch"`,
after creating and activating a device mesh connected to your mast workers, you
should be able to call the remote function via:

```py
import udf
from monarch.common.device_mesh import call_remote

call_remote(udf.my_udf, *args, **kwargs)
```

If you change the implementation of `udf.my_udf` locally, disconnecting your
device mesh and connecting/activating a new one should be sufficient to have the
mast workers pick up the changes.

It is worth noting that you actually don't need `--oilfs_workspace_dir` at all,
as long as your mast job is running in the same region that you have mounted on
your devgpu. However, you would need to manually add the relevant filesystem
paths to `sys.path` both locally and via `call_remote` on the workers.

### Enabling NFS for Syncing from DevGPU to MAST (Optional)

**Note: NFS is being deprecated in favor of OILFS.**

To enable NFS on your devgpu so that files can be synced in real time to your
mast workers, follow the instructions in
[this wiki](https://www.internalfb.com/intern/wiki/XLFormers/Tutorials/TorchX-based_Launching/#i-nfs).
Upon success, you will have an NFS directory on your devgpu at
`/mnt/aidev/$USER`. When you start your mast job with
`monarch.notebook reserve`, pass
`--nfs_workspace_dir "/path/relative/to/mnt/aidev/$USER"` to the command. The
mast workers will have access to any code you put in this directory.

Then, if you have a python file at `/mnt/aidev/$USER/monarch/udf.py` with a
function `my_udf`, and you started your mast job with
`--nfs_workspace_dir "monarch"`, you can interact with it in the same way as
described in the OILFS section above.

### [Out of Date] Connecting to the Workers

To see your existing mast jobs that are potentially eligible for use in device
meshes, use the `monarch.notebook.list_mast_jobs()` function. You will see an
output like:

```
{
  "name": "monarch-q5lhkr4",
  "latest_attempt_start_time": "2024-10-24 16:39:11",
  "hosts": 2,
  "gpus_per_host": 8,
  "job_state": "RUNNING",
  "task_states": [
    "PENDING"
  ]
}
{
  "name": "monarch-xbj3ds91",
  "latest_attempt_start_time": "2024-10-24 16:18:38",
  "hosts": 1,
  "gpus_per_host": 8,
  "job_state": "PENDING",
  "task_states": []
}
```

Both `job_state` and all entires in `task_states` must be `RUNNING` before
connection will succeed, but as soon as the job exists you can use the call
below and it will poll until success:

```
device_mesh = monarch.notebook.mast_mesh("your mast job name", hosts=N, n_gpus_per_host=M)
```

If something goes wrong or crashes, you may need to create a new device mesh.
Before you can do this, you need to clean up the old one, either by calling
`device_mesh.exit()` or `monarch.notebook.cleanup()`, restarting the bento
kernel or rerunning your program.

It is also worth noting that the mast workers run forever, so they need to be
manually killed.

### Using Bento

All of the instructions above should just work on Bento notebooks as long as you
have set up the kernel correctly. You can follow the instructions in
[this wiki](https://www.internalfb.com/intern/wiki/Metaconda/Notebook_Support/).

If you make changes inside `monarch/python` (excluding `monarch/examples`), then
you need to reinstall `monarch/python` and restart the Bento kernel for your
notebook to pick up the changes. If you make changes inside `/mnt/aidev/$USER`,
all you need to do is restart the Bento kernel.

#### Live Functions

The Bento kernel also supports "live functions", which allow you to write a
function in your notebook and have the workers execute it:

```py
import monarch

@monarch.remote
def live_fn(x):
    # <do something>

# Executes live_fn on the workers of the active device mesh.
live_fn(1)
```

See [this example notebook](https://fburl.com/anp/v1tlz19e).
