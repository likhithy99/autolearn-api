"""Evaluate an LLM checkpoint or published version: held-out perplexity + fixed prompt samples.

    python -m autolearn.llm.evaluate --ckpt runs/pretrain/best.pt --data data/llm/pretrain
    python -m autolearn.llm.evaluate --version v1 --llm-dir models/llm --data data/llm/sft --chat
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch

from .engine import Engine, LLMRegistry
from .train import TokenLoader, pick_device

TEXT_PROMPTS = [
    "Photosynthesis is the process by which",
    "The French Revolution began in",
    "To solve a quadratic equation, you",
    "The most important thing about writing good code is",
]
CHAT_PROMPTS = [
    "What is photosynthesis?",
    "Give me three tips for studying effectively.",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
]


@torch.no_grad()
def perplexity(engine: Engine, data_dir: str, iters: int = 50, batch: int = 8) -> dict:
    loader = TokenLoader(data_dir, "val", min(engine.config.block_size, 512), batch, engine.device, seed=0)
    engine.model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.get()
        # full-sequence forward with targets
        _, loss = engine.model(x, y)
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return {"val_loss": round(loss, 4), "perplexity": round(math.exp(loss), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="training checkpoint (.pt)")
    ap.add_argument("--tokenizer", default="data/llm/pretrain/tokenizer.json")
    ap.add_argument("--version", help="published version, e.g. v1")
    ap.add_argument("--llm-dir", default="models/llm")
    ap.add_argument("--data", help="token shard dir with val_*.bin for perplexity")
    ap.add_argument("--chat", action="store_true", help="use chat prompts (for SFT models)")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = str(pick_device(a.device))
    if a.version:
        engine = Engine(LLMRegistry(a.llm_dir).path(a.version), device)
    else:
        tmp = tempfile.mkdtemp()
        reg = LLMRegistry(tmp)
        engine = Engine(reg.path(reg.publish(a.ckpt, a.tokenizer, {"stage": "eval"})), device)

    report = {"params_M": round(engine.model.num_params() / 1e6, 1)}
    if a.data:
        report.update(perplexity(engine, a.data))
    print(json.dumps(report))
    for p in (CHAT_PROMPTS if a.chat else TEXT_PROMPTS):
        if a.chat:
            out = engine.chat([{"role": "user", "content": p}], max_tokens=a.max_tokens, temperature=0.7, seed=0)
        else:
            text = ""
            for ev in engine.complete_stream(p, max_tokens=a.max_tokens, temperature=0.7, seed=0):
                text += ev.get("delta", "")
            out = {"text": text}
        print(f"\n### {p}\n{out['text'].strip()}")


if __name__ == "__main__":
    main()
