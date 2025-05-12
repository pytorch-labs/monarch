import torch
from monarch import local_mesh
from monarch.cached_remote_function import (
    remote_autograd_function,
    RemoteAutogradFunction,
)
from monarch.common.device_mesh import remote
from monarch.common.function_caching import key_filters

# pyre-ignore
from transformer_engine.common.recipe import DelayedScaling

# pyre-ignore
from transformer_engine.pytorch.module import layernorm_mlp


log = remote("monarch.worker.worker.log", propagate="inspect")


def _ignore_scaling(v):
    if isinstance(v, DelayedScaling):
        return None
    return v


key_filters.append(_ignore_scaling)


def main():
    device_mesh = local_mesh(gpus=2)
    device_mesh.activate()
    # Have to set default device instead of initialize on cpu and cast to CUDA
    # because T194391401
    torch.set_default_device("cuda")
    tp_group = device_mesh.process_group(("gpu",))
    n_gpus = device_mesh.numdevices(("gpu",))

    if not issubclass(layernorm_mlp._LayerNormMLP, RemoteAutogradFunction):
        # monkey-patch inner torch.autograd.Function so it will our UDF will run on worker
        layernorm_mlp._LayerNormMLP = remote_autograd_function(
            layernorm_mlp._LayerNormMLP
        )
        # monkey-patch get_distributed_world_size so it's faked on controller
        layernorm_mlp.get_distributed_world_size = lambda _: n_gpus

    # Layer configuration
    hidden_size = 4096
    sequence_length = 2048
    batch_size = 4
    ffn_hidden_size = 16384
    dtype = torch.float16
    device = "cuda"

    init_method = None
    output_layer_init_method = None
    eps = 1e-5
    bias = False
    activation = "swiglu"
    normalization = "RMSNorm"
    sequence_parallel = True
    set_parallel_mode = True
    layer_norm_mlp = layernorm_mlp.LayerNormMLP(
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
        tp_size=n_gpus,
        # optimization params
        fuse_wgrad_accumulation=False,
        params_dtype=dtype,
        return_bias=False,
    ).cuda()
    for _ in range(2):
        x = torch.rand(sequence_length, batch_size, hidden_size).to(
            device=device, dtype=dtype
        )
        z = layer_norm_mlp(x)
        z.sum().backward()
        for p in layer_norm_mlp.parameters():
            assert p.grad is not None
            log(p.grad)
    device_mesh.exit()


if __name__ == "__main__":
    main()
