"""
This contains a simple FSDP implementation based on setattr for changing
between the FSDP sharded and unsharded parameters and on gradient generator for
the gradient reduce-scatter.
This only supports explicit prefetching for all-gather, dim-0 per-parameter
sharding without padding, and ``reshard_after_forward=False``.
"""

import argparse
import functools
import itertools
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monarch import (
    DeviceMesh,
    get_active_stream,
    local_mesh,
    no_mesh,
    remote,
    Simulator,
    SimulatorTraceMode,
    Stream,
    Tensor,
)
from monarch.gradient_generator import grad_generator
from monarch.profiler import profile, Schedule
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.utils.hooks import RemovableHandle


os.environ["GLOG_minloglevel"] = "1"  # silence PGNCCL logging
NHOSTS = 1
NGPUS = 2


log = remote("monarch.worker._testing_function.log", propagate="inspect")


def shard_tensor(tensor: Tensor, device_mesh: DeviceMesh, dim: str):
    shard_mesh_size = device_mesh.size(dim)
    shard_rank = device_mesh.rank(dim)
    if tensor.size(0) % shard_mesh_size != 0:
        raise NotImplementedError(
            f"Only even dim-0 sharding is supported: {tensor.shape=} {shard_mesh_size=}"
        )
    if not tensor.is_contiguous():
        raise NotImplementedError("Only contiguous tensors are supported")
    dim0_shard_size = tensor.size(0) // shard_mesh_size
    # Since `shard_rank` is a scalar tensor, we must use tensor indexing
    # instead of `torch.chunk`
    if isinstance(shard_rank, torch.Tensor) and shard_rank.ndim == 0:
        shard_rank = shard_rank.view(1)
    sharded_view = tensor.view(shard_mesh_size, dim0_shard_size, -1)[shard_rank]
    if tensor.ndim == 1:
        sharded_view = sharded_view.view(-1)
    else:
        sharded_view = sharded_view.flatten(0, 1)
    return sharded_view.detach().clone()


def unshard_tensor(tensor: Tensor, dims, *, out: Optional[Tensor] = None):
    # Assumes even sharding
    if out is None:
        unsharded_tensor = tensor.reduce(dims, reduction="stack")
        return unsharded_tensor.flatten(0, 1)
    device_mesh_size = tensor.mesh.size(dims)
    unsharded_size = [device_mesh_size] + list(tensor.size())
    out_view = out.view(unsharded_size)
    tensor.reduce(dims, reduction="stack", out=out_view)
    return out


def reduce_scatter_tensor(
    tensor: Tensor, dims, *, reduction: str = "sum", out: Optional[Tensor] = None
):
    # Assumes even sharding
    sharded_size = list(tensor.size())
    shard_mesh_size = tensor.mesh.size(dims)
    if tensor.size(0) % shard_mesh_size != 0:
        raise NotImplementedError(
            f"Only even dim-0 sharding is supported: {tensor.shape=} {shard_mesh_size=}"
        )
    sharded_size[0] //= shard_mesh_size
    tensor = tensor.view(shard_mesh_size, -1)
    if out is None:
        sharded_tensor = tensor.reduce(dims, reduction=reduction, scatter=True)
        sharded_tensor = sharded_tensor.view(sharded_size)
        return sharded_tensor
    assert list(out.size()) == sharded_size, f"{out.size()} != {sharded_size}"
    out = out.view(-1)
    tensor.reduce(dims, reduction=reduction, scatter=True, out=out)
    return out.view(sharded_size)


class FSDPParam:
    """
    This class tracks metadata for a parameter with FSDP applied and can be
    used as a handle for the parameter in FSDP operations.
    """

    def __init__(
        self,
        param: Tensor,
        module: nn.Module,
        param_name: str,
        device_mesh: DeviceMesh,
        dim: str,
        device: torch.device,
    ):
        self.module = module
        self.param_name = param_name  # unprefixed attribute name
        self.device_mesh = device_mesh
        self.dim = dim
        # Only support even dim-0 sharding for now
        sharded_tensor = shard_tensor(param.to(device), self.device_mesh, self.dim)
        self.sharded_param = nn.Parameter(
            sharded_tensor, requires_grad=param.requires_grad
        )
        self.unsharded_param: Optional[Tensor] = None
        self.sharded_size: torch.Size = self.sharded_param.size()
        self.unsharded_size: torch.Size = param.size()
        self.is_sharded = False
        self.param_fqn: Optional[str] = None  # fully qualified, for debugging

    def to_sharded(self):
        setattr(self.module, self.param_name, self.sharded_param)
        self.is_sharded = True

    def to_unsharded(self):
        assert self.unsharded_param is not None, "unsharded_param not all-gathered"
        setattr(self.module, self.param_name, self.unsharded_param)
        self.is_sharded = False


# Define this class to help make applying FSDP idempotent
class FSDPModule:
    # Parameter attribute name (e.g. "weight") to its `FSDPParam`
    _param_name_to_fsdp_param: Dict[str, FSDPParam]

    @staticmethod
    def named_fsdp_params(self, prefix: str = "", recurse: bool = True):
        memo = set()
        named_modules = self.named_modules(prefix=prefix) if recurse else [prefix, self]
        for module_name, module in named_modules:
            if not isinstance(module, FSDPModule):
                continue
            for param_name, fsdp_param in module._param_name_to_fsdp_param.items():
                if fsdp_param in memo:
                    continue
                memo.add(fsdp_param)
                prefixed_name = module_name + ("." if module_name else "") + param_name
                yield prefixed_name, fsdp_param

    @staticmethod
    def fsdp_params(self, recurse: bool = True):
        for _, fsdp_param in FSDPModule.named_fsdp_params(self):
            yield fsdp_param


def apply_fsdp(
    module: nn.Module,
    *,
    device_mesh: DeviceMesh,
    dim: str,
    device: torch.device,
) -> nn.Module:
    root_module = module
    for module in root_module.modules():
        if isinstance(module, FSDPModule):
            continue  # idempotent
        # Set the class to a new class that subclasses the original class and
        # `FSDPModule` to give new methods and make applying FSDP idempotent
        param_cls = type(
            f"FSDP{module.__class__.__name__}",
            (FSDPModule, module.__class__),
            {},
        )
        module.__class__ = param_cls
        module._param_name_to_fsdp_param = {}
        for param_name, param in module.named_parameters(recurse=False):
            fsdp_param = FSDPParam(param, module, param_name, device_mesh, dim, device)
            module._param_name_to_fsdp_param[param_name] = fsdp_param
            fsdp_param.to_sharded()
    return root_module


class BatchedAllGatherHandle:
    def __init__(
        self,
        input_tensor_sizes: List[torch.Size],
        device_mesh: DeviceMesh,
        dim: str,
        all_gather_output: Tensor,
        all_gather_input_borrow,
        all_gather_output_borrow,
    ):
        self.input_tensor_sizes = input_tensor_sizes
        self.device_mesh = device_mesh
        self.dim = dim
        self.all_gather_output = all_gather_output
        self.all_gather_input_borrow = all_gather_input_borrow
        self.all_gather_output_borrow = all_gather_output_borrow

    @torch.no_grad()
    def wait(self) -> List[Tensor]:
        current_stream = get_active_stream()
        self.all_gather_input_borrow.drop()
        self.all_gather_output_borrow.drop()
        if current_stream != self.all_gather_output.stream:
            raise NotImplementedError()
        # Copy-out (current stream)
        split_sizes = [size.numel() for size in self.input_tensor_sizes]
        mesh_size = self.device_mesh.size(self.dim)
        output_tensors = []
        for input_tensor_size in self.input_tensor_sizes:
            output_tensor_size = list(input_tensor_size)
            output_tensor_size[0] *= mesh_size
            output_tensor = torch.empty(
                output_tensor_size,
                dtype=self.all_gather_output.dtype,
                device=self.all_gather_output.device,
            )
            output_tensors.append(output_tensor)
        out = [t.view(mesh_size, -1) for t in output_tensors]
        all_gather_output = self.all_gather_output.view(mesh_size, -1)
        torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=1, out=out)
        return output_tensors


@torch.no_grad()
def batched_all_gather(
    input_tensors: List[Tensor],
    device_mesh: DeviceMesh,
    dim: str,
    comm_stream: Optional[Stream] = None,
    dtype: Optional[torch.dtype] = None,
) -> BatchedAllGatherHandle:
    assert input_tensors, "Requires at least one input tensor"
    # Copy-in (current stream)
    input_numels = [t.numel() for t in input_tensors]
    all_gather_input_numel = sum(input_numels)
    if dtype is None:
        dtypes = {t.dtype for t in input_tensors}
        dtype = functools.reduce(lambda t1, t2: torch.promote_types(t1, t2), dtypes)
    mesh_size, mesh_rank = device_mesh.size(dim), device_mesh.rank(dim)
    all_gather_output = torch.empty(
        (all_gather_input_numel * mesh_size,),
        dtype=dtype,
        device=input_tensors[0].device,
    )
    # TODO: Use in-place all-gather
    # DataDependentOutputException from using the mesh rank
    # all_gather_input = all_gather_output.narrow(
    #     0, all_gather_input_numel * mesh_rank, all_gather_input_numel
    # )
    all_gather_input = all_gather_output.new_empty((all_gather_input_numel,))
    all_gather_input_splits = all_gather_input.split(input_numels)
    flat_input_tensors = [t.view(-1) for t in input_tensors]
    torch._foreach_copy_(all_gather_input_splits, flat_input_tensors)
    # All-gather (comm stream)
    comm_stream = comm_stream or get_active_stream()
    all_gather_output_comm, all_gather_output_borrow = comm_stream.borrow(
        all_gather_output, mutable=True
    )
    all_gather_input_comm, all_gather_input_borrow = comm_stream.borrow(
        all_gather_input,
        mutable=True,  # False for out-of-place AG
    )
    with comm_stream.activate():
        unshard_tensor(all_gather_input_comm, dim, out=all_gather_output_comm)
    return BatchedAllGatherHandle(
        [t.size() for t in input_tensors],
        device_mesh,
        dim,
        all_gather_output,
        all_gather_input_borrow,
        all_gather_output_borrow,
    )


class BatchedReduceScatterHandle:
    def __init__(
        self,
        input_tensor_sizes: List[torch.Size],
        device_mesh: DeviceMesh,
        dim: str,
        reduce_scatter_output: Tensor,
        reduce_scatter_input_borrow,
        reduce_scatter_output_borrow,
    ):
        self.input_tensor_sizes = input_tensor_sizes
        self.device_mesh = device_mesh
        self.dim = dim
        self.reduce_scatter_output = reduce_scatter_output
        self.reduce_scatter_input_borrow = reduce_scatter_input_borrow
        self.reduce_scatter_output_borrow = reduce_scatter_output_borrow

    def wait(self) -> List[Tensor]:
        current_stream = get_active_stream()
        self.reduce_scatter_input_borrow.drop()
        self.reduce_scatter_output_borrow.drop()
        if current_stream != self.reduce_scatter_output.stream:
            raise NotImplementedError()
        # View-out (current stream)
        mesh_size = self.device_mesh.size(self.dim)
        offset = 0
        output_tensors = []
        for input_tensor_size in self.input_tensor_sizes:
            output_tensor_size = list(input_tensor_size)
            output_tensor_size[0] //= mesh_size
            output_tensor_numel = math.prod(output_tensor_size)
            output_tensor = self.reduce_scatter_output[
                offset : offset + output_tensor_numel
            ].view(output_tensor_size)
            output_tensors.append(output_tensor)
            offset += output_tensor_numel
        return output_tensors


@torch.no_grad()
def batched_reduce_scatter(
    input_tensors: List[Tensor],
    device_mesh: DeviceMesh,
    dim: str,
    comm_stream: Optional[Stream] = None,
    dtype: Optional[torch.dtype] = None,
) -> BatchedReduceScatterHandle:
    assert input_tensors, "Requires at least one input tensor"
    # Copy-in (current stream)
    input_numels = [t.numel() for t in input_tensors]
    reduce_scatter_input_numel = sum(input_numels)
    if dtype is None:
        dtypes = {t.dtype for t in input_tensors}
        dtype = functools.reduce(lambda t1, t2: torch.promote_types(t1, t2), dtypes)
    mesh_size = device_mesh.size(dim)
    reduce_scatter_input = torch.empty(
        (reduce_scatter_input_numel,), dtype=dtype, device=input_tensors[0].device
    )
    assert (
        reduce_scatter_input_numel % mesh_size == 0
    ), f"{reduce_scatter_input_numel=} {mesh_size=}"
    reduce_scatter_output = reduce_scatter_input.new_empty(
        (reduce_scatter_input_numel // mesh_size,)
    )
    torch._chunk_cat(
        input_tensors,
        dim=0,
        num_chunks=mesh_size,
        out=reduce_scatter_input.view(mesh_size, -1),
    )
    # Reduce-scatter (comm stream)
    comm_stream = comm_stream or get_active_stream()
    reduce_scatter_input_comm, reduce_scatter_input_borrow = comm_stream.borrow(
        reduce_scatter_input, mutable=False
    )
    reduce_scatter_output_comm, reduce_scatter_output_borrow = comm_stream.borrow(
        reduce_scatter_output, mutable=True
    )
    with comm_stream.activate():
        reduce_scatter_tensor(
            reduce_scatter_input_comm,
            dim,
            reduction="avg",
            out=reduce_scatter_output_comm,
        )
    return BatchedReduceScatterHandle(
        [t.size() for t in input_tensors],
        device_mesh,
        dim,
        reduce_scatter_output,
        reduce_scatter_input_borrow,
        reduce_scatter_output_borrow,
    )


def register_forward_prefetch_for_modules(
    source_module: nn.Module,
    target_modules: List[nn.Module],
    wait_module: nn.Module,
    device_mesh: DeviceMesh,
    dim: str,
    comm_stream: Stream,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[RemovableHandle, RemovableHandle]:
    """
    Registers explicit forward prefetching from ``source_module`` pre-forward
    for the FSDP parameters in ``target_modules``. The wait on the prefetched
    all-gather and copy-out will run in ``wait_module`` 's pre-forward.
    """
    target_fsdp_params = []
    target_fsdp_params_set = set()
    for target_module in target_modules:
        for fsdp_param in FSDPModule.fsdp_params(target_module):
            if fsdp_param not in target_fsdp_params_set:
                target_fsdp_params_set.add(fsdp_param)
                target_fsdp_params.append(fsdp_param)
    target_sharded_params = [
        fsdp_param.sharded_param for fsdp_param in target_fsdp_params
    ]

    def prefetch_hook(self, *args):
        handle = batched_all_gather(
            target_sharded_params,
            device_mesh,
            dim,
            comm_stream,
            dtype,
        )
        wait_module._fsdp_all_gather_handle = handle

    def wait_hook(self, *args):
        if (handle := getattr(self, "_fsdp_all_gather_handle", None)) is not None:
            unsharded_tensors = handle.wait()
            for unsharded_tensor, fsdp_param in zip(
                unsharded_tensors, target_fsdp_params
            ):
                fsdp_param.unsharded_param = nn.Parameter(
                    unsharded_tensor,
                    requires_grad=fsdp_param.sharded_param.requires_grad,
                )
                fsdp_param.to_unsharded()

    source_handle = source_module.register_forward_pre_hook(prefetch_hook)
    target_handle = wait_module.register_forward_pre_hook(wait_hook)
    return source_handle, target_handle


def finalize_batched_reduce_scatter(
    sharded_grads: List[Tensor],
    fsdp_params: List[FSDPParam],
) -> None:
    for sharded_grad, fsdp_param in zip(sharded_grads, fsdp_params):
        if sharded_grad.dtype != fsdp_param.sharded_param.dtype:
            sharded_grad = sharded_grad.to(fsdp_param.sharded_param.dtype)
        fsdp_param.sharded_param.grad = sharded_grad


def run_backward_nooverlap(loss: Tensor, module: FSDPModule):
    # If we do not reverse, we can get `None` gradients
    named_fsdp_params = list(reversed(list(FSDPModule.named_fsdp_params(module))))
    param_names = [t[0] for t in named_fsdp_params]
    fsdp_params = [t[1] for t in named_fsdp_params]
    unsharded_params = [fsdp_param.unsharded_param for fsdp_param in fsdp_params]
    grad_gen = grad_generator(loss, unsharded_params)
    for param_name, fsdp_param, grad in zip(param_names, fsdp_params, grad_gen):
        assert grad is not None, f"{param_name} has no grad!"
        fsdp_param.unsharded_param = None
        sharded_grad = reduce_scatter_tensor(grad, "gpu", reduction="avg")
        if sharded_grad.dtype != fsdp_param.sharded_param.dtype:
            sharded_grad = sharded_grad.to(fsdp_param.sharded_param.dtype)
        fsdp_param.sharded_param.grad = sharded_grad


def run_backward_overlap_batched(
    loss: Tensor,
    fsdp_param_partitions: List[List[FSDPParam]],
    device_mesh: DeviceMesh,
    dim: str,
    comm_stream: Stream,
    dtype: Optional[torch.dtype] = None,
):
    # Extract the unsharded parameters we expect to receive gradients and
    # filter the FSDP parameter partitions correspondingly
    unsharded_params = []
    for partition_idx in range(len(fsdp_param_partitions)):
        orig_fsdp_param_partition = fsdp_param_partitions[partition_idx]
        fsdp_param_partition = []
        for fsdp_param in orig_fsdp_param_partition:
            if (
                fsdp_param.unsharded_param is not None
                and fsdp_param.unsharded_param.requires_grad
            ):
                unsharded_params.append(fsdp_param.unsharded_param)
                fsdp_param_partition.append(fsdp_param)
        fsdp_param_partitions[partition_idx] = fsdp_param_partition
    # Generate gradients for each partition group and reduce-scatter
    grad_gen = grad_generator(loss, unsharded_params)
    prev_handle = None
    prev_fsdp_param_partition = []
    for fsdp_param_partition in fsdp_param_partitions:
        if not fsdp_param_partition:
            assert 0
            continue
        unsharded_grads: List[Tensor] = []
        for fsdp_param in fsdp_param_partition:
            unsharded_grad = next(grad_gen)
            assert unsharded_grad is not None, f"{fsdp_param.param_fqn} got no grad!"
            unsharded_grads.append(unsharded_grad)
            fsdp_param.unsharded_param = None
            fsdp_param.to_sharded()
        if prev_handle is not None:
            sharded_grads = prev_handle.wait()
            finalize_batched_reduce_scatter(sharded_grads, prev_fsdp_param_partition)
        prev_handle = batched_reduce_scatter(
            unsharded_grads, device_mesh, dim, comm_stream, dtype
        )
        prev_fsdp_param_partition = fsdp_param_partition
    if prev_handle is not None:
        sharded_grads = prev_handle.wait()
        finalize_batched_reduce_scatter(sharded_grads, prev_fsdp_param_partition)


set_worker_random_seed = remote(
    "monarch.worker.worker.set_random_seed_impl", propagate="inspect"
)

set_worker_logging_level = remote(
    "monarch.worker.worker.set_worker_logging_level", propagate="inspect"
)

check_equal = remote(
    "examples.fsdp.fsdp_setattr.check_equal_impl", propagate=lambda ts: False
)


def check_equal_impl(ts: Tuple[Tensor, Tensor]):
    return torch.equal(ts[0], ts[1])


def assert_equal_on_coordinates(ts: Tuple[Tensor, Tensor], coordinates):
    for coordinate in coordinates:
        are_equal = (
            remote(
                "examples.fsdp.fsdp_setattr.check_equal_impl",
            )
            .call_on_shard_and_fetch(
                ts,
                shard=coordinate,
            )
            .result()
        )
        with no_mesh.activate():
            assert are_equal, "mismatch!"


# NOTE: We take the GPT2 implementation from nanoGPT: https://github.com/karpathy/nanoGPT
@dataclass
class ModelArgs:
    seq_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    dim: int = 768
    bias: bool = False


class Attention(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.wqkv = nn.Linear(model_args.dim, 3 * model_args.dim, bias=model_args.bias)
        self.wo = nn.Linear(model_args.dim, model_args.dim, bias=model_args.bias)
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, T, C) = x.size()
        wq, wk, wv = self.wqkv(x).split(self.dim, dim=2)
        wk = wk.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        wq = wq.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        wv = wv.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(wq, wk, wv, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(model_args.dim, 4 * model_args.dim, bias=model_args.bias)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(4 * model_args.dim, model_args.dim, bias=model_args.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        return x


class Block(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.attn_norm = nn.LayerNorm(model_args.dim, bias=model_args.bias)
        self.attn = Attention(model_args)
        self.ffn_norm = nn.LayerNorm(model_args.dim, bias=model_args.bias)
        self.ffn = FeedForward(model_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        assert model_args.vocab_size is not None
        assert model_args.seq_len is not None
        self.model_args = model_args

        wte = nn.Embedding(model_args.vocab_size, model_args.dim)
        wpe = nn.Embedding(model_args.seq_len, model_args.dim)
        torch.nn.init.normal_(wte.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(wpe.weight, mean=0.0, std=0.02)
        blocks: List[Block] = []
        for _ in range(model_args.n_layers):
            block = Block(model_args)
            blocks.append(wrap(block))
        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                wpe=wpe,
                h=nn.ModuleList(blocks),
                norm=nn.LayerNorm(model_args.dim, bias=model_args.bias),
            )
        )
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)
        logits = self.output(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return loss

    def register_forward_prefetch(
        self,
        device_mesh: DeviceMesh,
        dim: str,
        comm_stream: Stream,
        dtype: Optional[torch.dtype] = None,
    ):
        register_prefetch_fn = functools.partial(
            register_forward_prefetch_for_modules,
            device_mesh=device_mesh,
            dim="gpu",
            comm_stream=comm_stream,
            dtype=dtype,
        )
        register_prefetch_fn(
            self, [self.transformer.wte, self.transformer.wpe], self.transformer.wte
        )
        num_blocks_per_prefetch = 4
        block_idx = 0
        while block_idx < len(self.transformer.h):
            blocks = self.transformer.h[block_idx : block_idx + num_blocks_per_prefetch]
            # First module in the batch waits for the batched all-gather
            register_prefetch_fn(self, blocks, blocks[0])
            block_idx += num_blocks_per_prefetch
        register_prefetch_fn(
            self, [self.transformer.norm, self.output], self.transformer.norm
        )

    def get_reduce_scatter_partitions(self) -> List[List[FSDPParam]]:
        # Each inner list represents a group of parameters for which to compute
        # gradients and reduce-scatter together, and the outer list should
        # follow the backward execution order.
        fsdp_param_partitions: List[List[FSDPParam]] = []
        fsdp_param_partitions.append(
            list(FSDPModule.fsdp_params(self.output))
            + list(FSDPModule.fsdp_params(self.transformer.norm))
        )
        num_blocks_per_partition = 4
        block_idx = 0
        block_partitions: List[List[FSDPParam]] = []
        while block_idx < len(self.transformer.h):
            blocks = self.transformer.h[
                block_idx : block_idx + num_blocks_per_partition
            ]
            block_fsdp_params: List[FSDPParam] = []
            for block in blocks:
                block_fsdp_params.extend(FSDPModule.fsdp_params(block))
            block_partitions.append(list(reversed(block_fsdp_params)))
            block_idx += num_blocks_per_partition
        for block_partition in reversed(block_partitions):
            fsdp_param_partitions.append(block_partition)
        fsdp_param_partitions.append(
            list(FSDPModule.fsdp_params(self.transformer.wpe))
            + list(FSDPModule.fsdp_params(self.transformer.wte))
        )
        return fsdp_param_partitions


def maybe_profile(should_profile: bool):
    if should_profile:
        return profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready="./traces/",
            schedule=Schedule(wait=0, warmup=0, active=2, repeat=1),
            record_shapes=True,
        )
    else:
        return nullcontext()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", help="Use SimulatorBackend")
    parser.add_argument("--profile", action="store_true", help="Run PyTorch profiler")
    args = parser.parse_args()

    if args.simulate:
        device_mesh = Simulator(
            hosts=NHOSTS,
            gpus=NGPUS,
            upload_trace=True,
            trace_mode=SimulatorTraceMode.STREAM_ONLY,
        ).mesh
        test_simulate(device_mesh)
    elif args.profile:
        device_mesh = local_mesh(hosts=NHOSTS, gpus=NGPUS)
        test_profile(device_mesh)
    else:
        device_mesh = local_mesh(hosts=NHOSTS, gpus=NGPUS)
        test_parity(device_mesh)


def test_parity(device_mesh: DeviceMesh):
    device_mesh.activate()
    set_worker_logging_level(logging.WARNING)
    coordinates = [
        {"host": i, "gpu": j}
        for i, j in itertools.product(
            range(device_mesh.size("host")), range(device_mesh.size("gpu"))
        )
    ]
    comm_stream = Stream("comm")

    # Ensure all workers have the same seed when initializing the model
    torch.manual_seed(0)
    set_worker_random_seed(0, 0)

    model_args = ModelArgs()
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.device(device):
        ref_model = Transformer(model_args)

    # Use the legacy `enable_wrap`/`wrap` APIs to enable GPU initialization
    # and be closer to xlformers
    fsdp_kwargs = {"device_mesh": device_mesh, "dim": "gpu", "device": device}
    with torch.device(device), enable_wrap(wrapper_cls=apply_fsdp, **fsdp_kwargs):
        model = wrap(Transformer(model_args))
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        ref_sharded_param = shard_tensor(ref_param, device_mesh, "gpu")
        assert (
            param.shape == ref_sharded_param.shape
        ), f"{param.shape=} != {ref_sharded_param.shape=}"
        param.detach().copy_(ref_sharded_param)

    print(model)
    for param_name, fsdp_param in FSDPModule.named_fsdp_params(model):
        fsdp_param.param_fqn = param_name

    optim_kwargs = {"lr": 1e-2, "fused": True, "foreach": False}
    ref_optim = torch.optim.AdamW(ref_model.parameters(), **optim_kwargs)
    optim = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    model.register_forward_prefetch(device_mesh, "gpu", comm_stream)

    # Use a different input on each data parallel rank
    set_worker_random_seed(0, device_mesh.rank("gpu"))
    seq_len = 64
    src = torch.randint(0, model_args.vocab_size, (2, seq_len), device="cuda")
    tgt = torch.randint(0, model_args.vocab_size, (2, seq_len), device="cuda")
    inp = (src, tgt)

    loss = model(*inp)
    ref_loss = ref_model(*inp)
    assert_equal_on_coordinates((loss, ref_loss), coordinates)

    # run_backward_nooverlap(loss, model)
    fsdp_param_partitions = model.get_reduce_scatter_partitions()
    run_backward_overlap_batched(
        loss, fsdp_param_partitions, device_mesh, "gpu", comm_stream
    )

    ref_loss.backward()
    for param in ref_model.parameters():
        param.grad.reduce_("gpu", reduction="avg")

    for (param_name, param), ref_param in zip(
        model.named_parameters(), ref_model.parameters()
    ):
        ref_sharded_grad = shard_tensor(ref_param.grad, device_mesh, "gpu")
        assert (
            param.shape == ref_sharded_grad.shape
        ), f"[{param_name}] {param.shape=} != {ref_sharded_grad.shape=}"
        assert param.grad is not None, f"{param_name}.grad is None!"
        assert_equal_on_coordinates((param.grad, ref_sharded_grad), coordinates)

    optim.step()
    ref_optim.step()
    optim.zero_grad()
    ref_optim.zero_grad()

    for (param_name, param), ref_param in zip(
        model.named_parameters(), ref_model.parameters()
    ):
        ref_sharded_param = shard_tensor(ref_param, device_mesh, "gpu")
        assert (
            param.shape == ref_sharded_param.shape
        ), f"[{param_name}] {param.shape=} != {ref_sharded_grad.shape=}"
        assert_equal_on_coordinates((param, ref_sharded_param), coordinates)

    print("All passed!")
    device_mesh.exit()


def test_profile(device_mesh: DeviceMesh):
    device_mesh.activate()
    set_worker_logging_level(logging.WARNING)

    # Ensure all workers have the same seed when initializing the model
    torch.manual_seed(0)
    set_worker_random_seed(0, 0)
    seq_len = 4096
    model_args = ModelArgs(seq_len=seq_len, dim=2048, n_layers=22, n_heads=16)
    device = torch.device("cuda", torch.cuda.current_device())

    # Use the legacy `enable_wrap`/`wrap` APIs to enable GPU initialization
    # and be closer to xlformers
    fsdp_kwargs = {"device_mesh": device_mesh, "dim": "gpu", "device": device}
    with torch.device(device), enable_wrap(wrapper_cls=apply_fsdp, **fsdp_kwargs):
        model = wrap(Transformer(model_args))
    print(model)
    for param_name, fsdp_param in FSDPModule.named_fsdp_params(model):
        fsdp_param.param_fqn = param_name
    optim_kwargs = {"lr": 1e-2, "fused": True, "foreach": False}
    optim = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    comm_stream = Stream("comm")
    model.register_forward_prefetch(device_mesh, "gpu", comm_stream, torch.bfloat16)

    # Use a different input on each data parallel rank
    set_worker_random_seed(0, device_mesh.rank("gpu"))
    batch_size = 2
    src = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device="cuda")
    tgt = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device="cuda")
    inp = (src, tgt)

    def inner():
        with maybe_profile(True) as prof:
            fsdp_param_partitions = model.get_reduce_scatter_partitions()
            for _ in range(3):
                loss = model(*inp)
                # run_backward_nooverlap(loss, model)
                run_backward_overlap_batched(
                    loss,
                    fsdp_param_partitions,
                    device_mesh,
                    "gpu",
                    comm_stream,
                    dtype=torch.float32,
                )
                optim.step()
                optim.zero_grad()
                prof.step()

    inner()
    device_mesh.exit()
    return


def test_simulate(device_mesh: DeviceMesh):
    device_mesh.activate()
    set_worker_logging_level(logging.WARNING)

    device = torch.device("cuda", torch.cuda.current_device())
    model_args = ModelArgs()
    # Use the legacy `enable_wrap`/`wrap` APIs to enable GPU initialization
    # and be closer to xlformers
    fsdp_kwargs = {"device_mesh": device_mesh, "dim": "gpu", "device": device}
    with torch.device(device), enable_wrap(wrapper_cls=apply_fsdp, **fsdp_kwargs):
        model = wrap(Transformer(model_args))
    optim_kwargs = {"lr": 1e-2, "fused": True, "foreach": False}
    optim = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    comm_stream = Stream("comm")
    model.register_forward_prefetch(device_mesh, "gpu", comm_stream, torch.bfloat16)

    set_worker_random_seed(0, device_mesh.rank("gpu"))
    src = torch.randint(0, model_args.vocab_size, (2, 64), device="cuda")
    tgt = torch.randint(0, model_args.vocab_size, (2, 64), device="cuda")
    inp = (src, tgt)

    loss = model(*inp)
    # run_backward_nooverlap(loss, model)
    fsdp_param_partitions = model.get_reduce_scatter_partitions()
    run_backward_overlap_batched(
        loss,
        fsdp_param_partitions,
        device_mesh,
        "gpu",
        comm_stream,
        dtype=torch.float32,
    )
    optim.step()
    optim.zero_grad()

    device_mesh.exit()


if __name__ == "__main__":
    main()
