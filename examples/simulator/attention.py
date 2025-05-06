import torch
import torch.nn.functional as F
from monarch import fetch_shard, get_active_stream, simulator, Stream
from torch.nn.attention import sdpa_kernel, SDPBackend

device_mesh = simulator(hosts=2, gpus=4).mesh


def basic_context_parallelism_no_overlapping(q, k, v, is_causal=False):
    # CP size is 2
    full_kv_shape = list(k.shape)
    full_kv_shape[2] *= 2
    full_kv_shape = tuple(full_kv_shape)

    k = k.flatten().contiguous()
    v = v.flatten().contiguous()

    comms_stream = Stream("comms")
    k, borrow_comm_k = comms_stream.borrow(k)
    v, borrow_comm_v = comms_stream.borrow(v)
    with comms_stream.activate():
        k = k.reduce("host", "stack")
        v = v.reduce("host", "stack")
        borrow_comm_k.drop()
        borrow_comm_v.drop()
    k, borrow_common_k = get_active_stream().borrow(k)
    v, borrow_common_v = get_active_stream().borrow(v)
    full_k = k.reshape(full_kv_shape)
    full_v = v.reshape(full_kv_shape)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q, full_k, full_v, is_causal=is_causal)
    borrow_common_k.drop()
    borrow_common_v.drop()
    return out


def all_gather_next_kv(k, v, comms_stream):
    kv = torch.cat((k.flatten(), v.flatten())).contiguous()
    kv, borrow_comm_kv = comms_stream.borrow(kv)
    with comms_stream.activate():
        kv = kv.reduce("host", "stack")

    return kv, borrow_comm_kv


def basic_context_parallelism(q, k, v, is_causal=False):
    cp_size = 2
    full_kv_shape = list(k.shape)
    full_kv_shape[2] *= cp_size
    full_kv_shape = tuple(full_kv_shape)

    comms_stream = Stream("comms")
    out = None
    for i in range(cp_size):
        if i == 0:
            next_kv, borrow_comm_kv = all_gather_next_kv(k, v, comms_stream)
        else:
            # The reduce creates one extra dimension.
            next_kv, return_comm_kv = get_active_stream().borrow(next_kv)
            next_kv = next_kv.flatten()
            flat_k, flat_v = next_kv.chunk(2)
            k = flat_k.chunk(cp_size)[i].reshape(k.shape)
            v = flat_v.chunk(cp_size)[i].reshape(v.shape)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            local_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        if out is None:
            out = local_out
        else:
            # This is not correct, update is more complicated. But we just want to
            # know if we can accuractely track execution.
            out = out + local_out

    borrow_comm_kv.drop()
    return_comm_kv.drop()
    return out


def no_ff_attention():
    bs, query_tokens, context_tokens, dim, nheads = 8, 64, 64, 32, 8
    q = torch.rand(
        (bs, nheads, query_tokens, dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=False,
    )
    k = torch.rand(
        (bs, nheads, context_tokens, dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=False,
    )
    v = torch.rand(
        (bs, nheads, context_tokens, dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=False,
    )
    out = basic_context_parallelism(q, k, v, is_causal=True)
    # out = basic_context_parallelism_no_overlapping(q, k, v, is_causal=False)
    return out


with device_mesh.activate():
    out = no_ff_attention()
    local_out = fetch_shard(out, {"host": 0, "gpu": 0}).result()

device_mesh.exit()
