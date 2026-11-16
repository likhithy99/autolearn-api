"""Standard multiple-choice benchmarks, scored the usual way for base models: each answer choice is appended
to the question and the model picks the choice with the highest log-likelihood (no generation needed).

    python -m autolearn.llm.benchmarks --ckpt runs/long_micro/best.pt
    python -m autolearn.llm.benchmarks --version v1 --tasks hellaswag --limit 1000

Reported: `acc` (sum of log-probs) and `acc_norm` (log-prob per token, which removes the bias toward short
answers). Random guessing scores 25% on all of these tasks. Reference points for scale: GPT-2 small (124M)
gets ~29% HellaSwag acc_norm and ~44% ARC-Easy; GPT-3 175B gets ~79% HellaSwag.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time

import torch
import torch.nn.functional as F


def _hellaswag(limit):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    clean = lambda t: re.sub(r"\[.*?\]", "", t.replace(" [title]", ". ")).replace("  ", " ").strip()
    for ex in ds.select(range(min(limit, len(ds)))):
        ctx = clean(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
        yield ctx, [" " + clean(e) for e in ex["endings"]], int(ex["label"])


def _arc(subset, limit):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    n = 0
    for ex in ds:
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        yield (f"Question: {ex['question']}\nAnswer:", [" " + t for t in ex["choices"]["text"]],
               labels.index(ex["answerKey"]))
        n += 1
        if n >= limit:
            break


TASKS = {
    "hellaswag": lambda n: _hellaswag(n),
    "arc_easy": lambda n: _arc("ARC-Easy", n),
    "arc_challenge": lambda n: _arc("ARC-Challenge", n),
}


@torch.no_grad()
def score_choices(model, tok, device, ctx: str, choices: list[str], max_len: int) -> tuple[list[float], list[float]]:
    """Total and per-token log-likelihood of each choice given ctx."""
    c_ids = [tok.eos] + tok.encode(ctx)  # documents are eos-separated in pretraining
    rows, spans = [], []
    for ch in choices:
        e_ids = tok.encode(ch)
        ids = (c_ids + e_ids)[-max_len:]
        rows.append(ids)
        spans.append(len(e_ids))
    T = max(len(r) for r in rows)
    x = torch.zeros(len(rows), T, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r)
    x = x.to(device)
    logits, _ = model(x, targets=torch.full_like(x, -100))  # targets => logits at every position
    logp = F.log_softmax(logits.float(), dim=-1).cpu()
    total, mean = [], []
    for i, r in enumerate(rows):
        n = spans[i]
        tgt = torch.tensor(r[len(r) - n:])
        lp = logp[i, len(r) - n - 1:len(r) - 1].gather(-1, tgt.unsqueeze(-1)).sum().item()
        total.append(lp)
        mean.append(lp / max(1, n))
    return total, mean


def run_benchmarks(engine, tasks=("hellaswag", "arc_easy"), limit=1000) -> dict:
    os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())
    model, tok, dev = engine.model, engine.tok, engine.device
    out = {}
    for name in tasks:
        t0, n, acc, acc_norm = time.time(), 0, 0, 0
        for ctx, choices, gold in TASKS[name](limit):
            total, mean = score_choices(model, tok, dev, ctx, choices, engine.config.block_size)
            acc += int(max(range(len(total)), key=total.__getitem__) == gold)
            acc_norm += int(max(range(len(mean)), key=mean.__getitem__) == gold)
            n += 1
        out[name] = {"n": n, "acc": round(acc / max(1, n), 4), "acc_norm": round(acc_norm / max(1, n), 4),
                     "seconds": round(time.time() - t0)}
    return out


def main():
    from .engine import Engine, LLMRegistry
    from .train import pick_device
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--tokenizer", default="data/llm/pretrain/tokenizer.json")
    ap.add_argument("--version")
    ap.add_argument("--llm-dir", default="models/llm")
    ap.add_argument("--tasks", default="hellaswag,arc_easy")
    ap.add_argument("--limit", type=int, default=1000, help="max examples per task")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = str(pick_device(a.device))
    if a.version:
        engine = Engine(LLMRegistry(a.llm_dir).path(a.version), device)
    else:
        reg = LLMRegistry(tempfile.mkdtemp())
        engine = Engine(reg.path(reg.publish(a.ckpt, a.tokenizer, {"stage": "eval"})), device)
    res = run_benchmarks(engine, a.tasks.split(","), a.limit)
    print(json.dumps({"params_M": round(engine.model.num_params() / 1e6, 1), **res}, indent=2))


if __name__ == "__main__":
    main()
