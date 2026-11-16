"""Low-rank adapters (LoRA) for per-customer custom models.

The base model stays frozen and shared; each customer gets a few MB of adapter weights (W + B·A·alpha/r on the
attention and MLP projections). Switching customers = loading a different adapter state, so one GPU serves
every company's custom model. With no adapter loaded (B = 0) the model is exactly the base model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("wq", "wkv", "wo", "w1", "w2", "w3")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        self.enabled = False
        self.reset()

    @property
    def weight(self):  # so code that inspects .weight (e.g. tying, dtype checks) still works
        return self.base.weight

    def reset(self):
        """Fresh adapter: A random, B zero -> output identical to the base layer."""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            self.B.zero_()

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + ((x.to(self.A.dtype) @ self.A.T) @ self.B.T * self.scale).to(y.dtype)


def add_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, targets=DEFAULT_TARGETS) -> int:
    """Wrap the target Linear layers in place; returns the number of trainable adapter parameters."""
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, r, alpha)
                setattr(module, name, lora)
                n += lora.A.numel() + lora.B.numel()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in lora_modules(model):
        m.A.requires_grad_(True)
        m.B.requires_grad_(True)
    return n


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def has_lora(model) -> bool:
    return bool(lora_modules(model))


def lora_parameters(model):
    return [p for m in lora_modules(model) for p in (m.A, m.B)]


def get_adapter(model) -> dict:
    return {f"{i}.{k}": getattr(m, k).detach().cpu().clone() for i, m in enumerate(lora_modules(model))
            for k in ("A", "B")}


@torch.no_grad()
def set_adapter(model, state: dict | None):
    """Load an adapter (None = base model)."""
    mods = lora_modules(model)
    if state is None:
        for m in mods:
            m.enabled = False
        return
    for i, m in enumerate(mods):
        m.A.copy_(state[f"{i}.A"])
        m.B.copy_(state[f"{i}.B"])
        m.enabled = True


def new_adapter(model):
    """Reset to a fresh (no-op) trainable adapter."""
    for m in lora_modules(model):
        m.reset()
        m.enabled = True
