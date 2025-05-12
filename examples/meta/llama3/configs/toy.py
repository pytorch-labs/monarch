# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

# pyre-unsafe

eval_interval = 10  # keep frequent because we'll overfit
eval_iters = 10
log_interval = 1  # don't print too too often

data_root_dir = "./data"
dataset = "shakespeare"
batch_size = 2
block_size = 256  # context of up to 256 previous characters

init_from = "scratch"
model_name = "toy"

# model args
n_layer = 4
n_head = 8
n_local_heads = 1
dim = 512
intermediate_size = 2048
vocab_size = 128256
rope_base = 500000
use_te = False

learning_rate = 3e-5  # with baby networks can afford to go a bit higher
max_iters = 10000
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially
