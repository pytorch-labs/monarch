# pyre-unsafe
eval_interval = 100
eval_iters = 10
log_interval = 10

# it will OOM if any bigger
batch_size = 2
block_size = 8192

# TODO get init from working
# init_from = "llama3/checkpoints/meta-llama/Meta-Llama-3-8B/original/consolidated.00.pth"
init_from = "scratch"
model_name = "llama-3-8b"

# Model args
n_layer = 32
n_head = 32
n_local_heads = 8
dim = 4096
intermediate_size = 14336
vocab_size = 128256
rope_base = 500000
use_te = True

learning_rate = (
    3.45e-4  # https://www.internalfb.com/phabricator/paste/view/P1600444636?lines=46
)
max_iters = 20000
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

dataset = None
xlformers_data = "/mnt/wsfuse/fair_llm_v2/shuffled/c4/en:50,/mnt/wsfuse/fair_llm_v2/shuffled/c4/de:10,/mnt/wsfuse/fair_llm_v2/shuffled/c4/es:10,/mnt/wsfuse/fair_llm_v2/shuffled/c4/fr:10,/mnt/wsfuse/fair_llm_v2/shuffled/c4/it:10,/mnt/wsfuse/fair_llm_v2/shuffled/c4/pt:10"
xlformers_tokenizer = "cl_toplang_128k_llama3_final_tok"

# distributed training
n_hosts = 8
n_gpus = 8
pp = 4
tp = 8
dp = 2
