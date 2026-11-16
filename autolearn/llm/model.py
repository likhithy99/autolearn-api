"""Decoder-only transformer language model, written from scratch in PyTorch.

Architecture (same family as modern open LLMs):
  * RMSNorm pre-normalisation
  * Rotary position embeddings (RoPE)
  * Grouped-query attention (GQA) via fused scaled-dot-product attention (FlashAttention on CUDA)
  * SwiGLU feed-forward
  * Tied input/output embeddings, no biases
  * KV cache for fast autoregressive generation
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # Options used when loading open-weight models (Qwen2 / Llama families); defaults = our own architecture.
    qkv_bias: bool = False
    ffn_hidden: int | None = None
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def to_dict(self):
        return asdict(self)


# Approximate parameter counts assume a 32k vocabulary.
PRESETS: dict[str, dict] = {
    "nano":   dict(n_layer=4,  n_head=4,  n_kv_head=2, n_embd=256,  block_size=512),    # ~10M  laptop smoke test
    "mini":   dict(n_layer=6,  n_head=6,  n_kv_head=2, n_embd=384,  block_size=512),    # ~22M  laptop (M-series)
    "micro":  dict(n_layer=8,  n_head=8,  n_kv_head=4, n_embd=512,  block_size=512),    # ~40M  laptop
    "small":  dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768,  block_size=1024),   # ~120M 1 GPU, hours
    "medium": dict(n_layer=24, n_head=16, n_kv_head=8, n_embd=1024, block_size=2048),   # ~350M 8 GPUs, ~1 day
    "large":  dict(n_layer=24, n_head=16, n_kv_head=8, n_embd=2048, block_size=2048),   # ~1.3B 8-64 GPUs
    "xl":     dict(n_layer=32, n_head=32, n_kv_head=8, n_embd=4096, block_size=4096),   # ~7B   cluster, weeks
}


def config_from_preset(name: str, vocab_size: int, **overrides) -> GPTConfig:
    return GPTConfig(vocab_size=vocab_size, **{**PRESETS[name], **overrides})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim: int, length: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(length).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D/2)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0 and c.n_head % c.n_kv_head == 0
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        self.wq = nn.Linear(c.n_embd, self.nh * self.hd, bias=c.qkv_bias)
        self.wkv = nn.Linear(c.n_embd, 2 * self.nkv * self.hd, bias=c.qkv_bias)
        self.wo = nn.Linear(self.nh * self.hd, c.n_embd, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin, cache: dict | None = None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k, v = self.wkv(x).view(B, T, 2, self.nkv, self.hd).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        prefix = 0
        if cache is not None:
            if cache.get("k") is not None:
                prefix = cache["k"].shape[2]
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if prefix and T > 1:
            raise ValueError("multi-token decode with a non-empty cache is not supported")
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        hidden = c.ffn_hidden or 64 * ((int(2 * 4 * c.n_embd / 3) + 63) // 64)
        self.w1 = nn.Linear(c.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(c.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, c.n_embd, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1, self.attn = RMSNorm(c.n_embd, c.norm_eps), Attention(c)
        self.ln2, self.mlp = RMSNorm(c.n_embd, c.norm_eps), SwiGLU(c)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, cache)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # weight tying
        cos, sin = rope_tables(config.n_embd // config.n_head, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled init of residual projections (GPT-2)
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches: list | None = None, start_pos: int = 0):
        B, T = idx.shape
        if start_pos + T > self.config.block_size:
            raise ValueError(f"sequence length {start_pos + T} exceeds block_size {self.config.block_size}")
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        x = self.drop(self.tok_emb(idx))
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, caches[i] if caches is not None else None)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x[:, -1:, :]), None  # inference: only last position
        logits = self.lm_head(x)
        # mean over target tokens; safe when a window has none (e.g. SFT window with only prompt tokens)
        n = (targets != -100).sum().clamp(min=1)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
                               reduction="sum") / n
        return logits, loss

    def sequence_logprobs(self, idx, targets):
        """Sum of log p(target) over non-ignored positions, per sequence (used by DPO)."""
        x = self.drop(self.tok_emb(idx))
        T = idx.shape[1]
        for blk in self.blocks:
            x = blk(x, self.rope_cos[:T], self.rope_sin[:T])
        logp = F.log_softmax(self.lm_head(self.norm(x)).float(), dim=-1)
        mask = targets != -100
        tok = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (tok * mask).sum(-1)

    def configure_optimizer(self, lr, weight_decay, betas=(0.9, 0.95), device_type="cpu"):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))
