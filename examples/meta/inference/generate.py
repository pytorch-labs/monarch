# pyre-unsafe

# Run command:
# python generate.py

import monarch
import torch
from deepseek.model import ModelArgs, Transformer
from monarch import fetch_shard, Future, local_mesh, no_mesh, remote


# Setting default dtype on remote workers
set_default_dtype = remote(
    "monarch.worker.worker.set_default_dtype",
    propagate=lambda dtype: torch.set_default_dtype(dtype),
)


# create a device mesh (python backend)
device_mesh = local_mesh(gpus=2)

# for interactive use lets keep this device_mesh active
device_mesh.activate()

# needed so that tensors are created on cuda
torch.set_default_device("cuda")

# set default dtype to bfloat16
set_default_dtype(torch.bfloat16)

# DeepSeek-V3 model creation
args: ModelArgs = ModelArgs(
    max_seq_len=2048,
    max_batch_size=32,
    vocab_size=32000,
    n_layers=3,  # smaller model for testing
)
model = Transformer(args)

# Run on workers
inputs = torch.randint(args.vocab_size, (args.max_batch_size, args.max_seq_len))
y = model(inputs)

# Observing results on controller
yf: Future = fetch_shard(y)
with no_mesh.activate():
    print(yf.result())

# just for local testing to shut down all the workers
device_mesh.exit()

# Necessary for clean exit when device_mesh.activate() is used outside
# of a with block
device_mesh.deactivate()
