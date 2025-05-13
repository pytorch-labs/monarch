# pyre-unsafe

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from monarch.cached_remote_function import (
    remote_autograd_function,
    RemoteAutogradFunction,
)

from monarch.common.function_caching import key_filters

from torch import Tensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.dim, dtype=config.dtype))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        head_dim = config.dim // config.n_head
        self.wq = nn.Linear(
            config.dim, head_dim * config.n_head, bias=config.bias, dtype=config.dtype
        )
        self.wk = nn.Linear(
            config.dim,
            head_dim * config.n_local_heads,
            bias=config.bias,
            dtype=config.dtype,
        )
        self.wv = nn.Linear(
            config.dim,
            head_dim * config.n_local_heads,
            bias=config.bias,
            dtype=config.dtype,
        )

        # output projection
        self.wo = nn.Linear(
            config.dim, config.dim, bias=config.bias, dtype=config.dtype
        )
        # regularization
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    def forward(self, x, freqs_cis):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(B, T, self.n_head, -1)
        k = k.view(B, T, self.n_local_heads, -1)
        v = v.view(B, T, self.n_local_heads, -1)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        from torch.nn.attention import sdpa_kernel, SDPBackend

        # efficient attention using Flash Attention CUDA kernels

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            config.dim, config.intermediate_size, bias=config.bias, dtype=config.dtype
        )
        self.w3 = nn.Linear(
            config.dim, config.intermediate_size, bias=config.bias, dtype=config.dtype
        )
        self.w2 = nn.Linear(
            config.intermediate_size, config.dim, bias=config.bias, dtype=config.dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _ignore_scaling(v):
    from transformer_engine.common.recipe import DelayedScaling

    if isinstance(v, DelayedScaling):
        return None
    return v


key_filters.append(_ignore_scaling)


class TransformerBlock(nn.Module):
    def __init__(self, config, tp_group) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.config = config
        if config.use_te:
            from transformer_engine.pytorch.module import layernorm_mlp

            if (
                not issubclass(layernorm_mlp._LayerNormMLP, RemoteAutogradFunction)
                and config.use_monarch
            ):
                # monkey-patch inner torch.autograd.Function so it will our UDF will run on worker
                # pyre-ignore[9]
                layernorm_mlp._LayerNormMLP = remote_autograd_function(
                    layernorm_mlp._LayerNormMLP
                )
                # monkey-patch get_distributed_world_size so it's faked on controller
                layernorm_mlp.get_distributed_world_size = lambda _: config.tp

            self.feed_forward = layernorm_mlp.LayerNormMLP(
                config.dim,
                config.intermediate_size,
                eps=config.norm_eps,
                bias=config.bias,
                normalization="RMSNorm",
                activation="swiglu",
                init_method=None,
                output_layer_init_method=None,
                # parallel params
                sequence_parallel=config.tp > 1,
                set_parallel_mode=config.tp > 1,
                tp_group=tp_group if config.tp > 1 else None,
                tp_size=config.tp,
                # optimization params
                fuse_wgrad_accumulation=False,
                params_dtype=config.dtype,
                return_bias=False,
            )
        else:
            self.feed_forward = FeedForward(config)
            self.ffn_norm = RMSNorm(config)
        self.attention_norm = RMSNorm(config)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.config.use_te:
            out = h + self.feed_forward(h)
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out, freqs_cis


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    # pyre-ignore
    freqs: torch.Tensor = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            torch.mul(xshaped[..., 0], freqs_cis[..., 0])
            - xshaped[..., 1] * freqs_cis[..., 1],
            torch.mul(xshaped[..., 1], freqs_cis[..., 0])
            + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class Transformer(nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert tp_group is not None if config.tp > 1 else True
        self.config = config

        self.layers = nn.ModuleList(
            [TransformerBlock(config, tp_group) for _ in range(config.n_layer)]
        )
        # Optional fields below may be None in the SPMD+pipeline parallel case
        self.tok_embeddings: Optional[nn.Embedding] = nn.Embedding(
            config.vocab_size, config.dim, dtype=config.dtype
        )
        self.norm: Optional[RMSNorm] = RMSNorm(config)
        self.output: Optional[nn.Linear] = nn.Linear(
            config.dim, config.vocab_size, bias=config.bias, dtype=config.dtype
        )

        # report number of parameters
        logger.info("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def get_total_size(self, non_embedding=True):
        size = sum(p.numel() * p.element_size() for p in self.parameters())
        size += sum(b.numel() * b.element_size() for b in self.buffers())

        if non_embedding:
            size -= (
                self.tok_embeddings.weight.numel()
                * self.tok_embeddings.weight.element_size()
            )
        return size

    def get_readable_total_size(self, non_embedding=True):
        size = self.get_total_size(non_embedding=non_embedding)
        if size >= 1024**3:
            return f"{size / 1024**3:.2f} GB"
        elif size >= 1024**2:
            return f"{size / 1024**2:.2f} MB"
        elif size >= 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size} B"

    def forward(self, idx):
        if len(idx.size()) == 2:
            b, t = idx.size()
        else:  # in pipeline setups, input shape is B, T, H
            b, t, _ = idx.size()

        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        freqs_cis = precompute_freqs_cis(
            t,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )

        # forward the  model itself
        # token embeddings of shape (b, t, dim)
        tok_emb = self.tok_embeddings(idx) if self.tok_embeddings else idx
        x = tok_emb
        for block in self.layers:
            x, freqs_cis = block(x, freqs_cis)
        x = self.norm(x) if self.norm else x

        logits = self.output(x) if self.output else x
        return logits

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
        optimizer_type="adam",
        params=None,
    ):
        params = params or [p for p in self.parameters() if p.requires_grad]
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=True, foreach=False
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, fused=True)
        else:
            raise RuntimeError(f"Unrecognized optimizer {optimizer_type}")
        logger.info("using fused AdamW")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def loss_fn(pred, labels):
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )
