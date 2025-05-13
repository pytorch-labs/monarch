# pyre-unsafe
from llama3.configs.llama8b import *  # noqa

max_iters = 10
eval_interval = 10
eval_iters = 10
log_interval = 1

dataset = "shakespeare"
xlformers_data = None
xlformers_tokenizer = None

# distributed training
n_hosts = 1
n_gpus = 8
pp = 4
tp = 1
dp = 2
use_te = False
mesh_type = "rust_local"
