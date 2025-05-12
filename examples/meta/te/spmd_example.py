import torch
from transformer_engine.pytorch.module import LayerNormMLP


# Layer configuration

hidden_size = 4096
sequence_length = 2048
batch_size = 4
ffn_hidden_size = 16384
num_attention_heads = 32
dtype = torch.float16
device = "cuda"

init_method = None
output_layer_init_method = None
eps = 1e-5
bias = False
activation = "swiglu"
normalization = "RMSNorm"
# TODO changes
sequence_parallel = False
set_parallel_mode = False
tp_group = None
tp_world_size = 1


layer_norm_mlp = LayerNormMLP(
    hidden_size,
    ffn_hidden_size,
    eps=eps,
    bias=bias,
    normalization=normalization,
    activation=activation,
    init_method=init_method,
    output_layer_init_method=output_layer_init_method,
    # parallel params
    sequence_parallel=sequence_parallel,
    set_parallel_mode=set_parallel_mode,
    tp_group=tp_group,
    tp_size=tp_world_size,
    # optimization params
    fuse_wgrad_accumulation=False,
    params_dtype=dtype,
    return_bias=False,
).cuda()
x = torch.rand(sequence_length, batch_size, hidden_size).to(device=device, dtype=dtype)
x.requires_grad = True
z = layer_norm_mlp(x)
z.sum().backward()
