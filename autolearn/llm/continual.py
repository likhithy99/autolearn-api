"""Continual learning worker: the LLM improves itself from real user feedback.

Loop (every --interval seconds):
  1. Collect unused feedback from tenants that opted in to training.
       rating +1                -> SFT target = the model's own answer   (reinforce good behaviour)
       correction supplied      -> SFT target = the human correction     (learn the right answer)
       rating -1 + correction   -> preference pair (chosen=correction, rejected=model answer) -> DPO
  2. When enough examples exist, fine-tune a challenger from the current champion with
       SFT loss on feedback + replay loss on the original chat data (prevents catastrophic forgetting)
       + DPO loss on preference pairs (beta-scaled, frozen reference = champion).
  3. Champion/challenger gate on held-out data:
       promote only if feedback-holdout loss improves AND general replay loss does not regress > tolerance.
  4. Publish a new immutable version (promoted or not, for audit) and hot-swap the API to it if promoted.

Run alongside the API (ideally on a GPU box):
    python -m autolearn.llm.continual --data-dir data --llm-dir models/llm --replay data/llm/sft
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..db import DB
from .engine import LLMRegistry
from .model import GPT, GPTConfig
from .tokenizer import Tok
from .train import TokenLoader, pick_device

log = logging.getLogger("autolearn.llm.continual")
EVENT_PROJECT = "__llm__"


def build_examples(rows, tok: Tok):
    """Return (sft_examples, dpo_pairs) as token/target lists."""
    sft, pairs = [], []

    def encode(row, answer):
        if row["kind"] == "chat":
            msgs = row["request"] + [{"role": "assistant", "content": answer}]
            ids, mask = tok.render_chat(msgs, add_generation_prompt=False)
        else:
            p = [tok.bos] + tok.encode(row["request"])
            a = tok.encode(answer) + [tok.eos]
            ids, mask = p + a, [0] * len(p) + [1] * len(a)
        return ids, mask

    for r in rows:
        if r["correction"]:
            sft.append((r["completion_id"], *encode(r, r["correction"])))
            if r["rating"] < 0 and r["output"].strip():
                pairs.append((r["completion_id"], encode(r, r["correction"]), encode(r, r["output"])))
        elif r["rating"] > 0 and r["output"].strip():
            sft.append((r["completion_id"], *encode(r, r["output"])))
    return sft, pairs


def collate(examples, block, device):
    """examples: list of (ids, mask) -> x, y with -100 on non-target positions (left-truncated to block)."""
    L = min(block, max(len(ids) for ids, _ in examples))
    xs, ys = [], []
    for ids, mask in examples:
        ids, mask = ids[-(L + 1):], mask[-(L + 1):]
        x = ids[:-1]
        y = [t if m else -100 for t, m in zip(ids[1:], mask[1:])]
        pad = L - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [-100] * pad)
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)


@torch.no_grad()
def feedback_loss(model, examples, block, device, bs=8):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), bs):
        x, y = collate(examples[i:i + bs], block, device)
        _, loss = model(x, y)
        k = int((y != -100).sum())
        tot, n = tot + loss.item() * k, n + k
    return tot / max(n, 1)


@torch.no_grad()
def replay_loss(model, data_dir, block, device, iters=20, bs=8):
    model.eval()
    loader = TokenLoader(data_dir, "val", block, bs, device, seed=1234)  # same batches for every model
    return sum(model(*loader.get())[1].item() for _ in range(iters)) / iters


@torch.no_grad()
def preference_accuracy(model, pairs, block, device):
    if not pairs:
        return None
    model.eval()
    wins = 0
    for _, ch, rj in pairs:
        xc, yc = collate([ch], block, device)
        xr, yr = collate([rj], block, device)
        wins += int(model.sequence_logprobs(xc, yc) > model.sequence_logprobs(xr, yr))
    return wins / len(pairs)


def dpo_loss(policy, ref, pairs, block, device, beta):
    chosen = [c for _, c, _ in pairs]
    rejected = [r for _, _, r in pairs]
    xc, yc = collate(chosen, block, device)
    xr, yr = collate(rejected, block, device)
    pc, pr = policy.sequence_logprobs(xc, yc), policy.sequence_logprobs(xr, yr)
    with torch.no_grad():
        rc, rr = ref.sequence_logprobs(xc, yc), ref.sequence_logprobs(xr, yr)
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()


def run_once(a) -> dict | None:
    db = DB(Path(a.data_dir) / "autolearn.db")
    purged = db.purge_llm_logs(a.retention_days)
    if purged:
        log.info("retention: purged %d completion logs older than %d days", purged, a.retention_days)
    reg = LLMRegistry(a.llm_dir)
    current = reg.current()
    if not current:
        log.info("no published model yet")
        return None
    rows = db.unused_llm_feedback()
    mdir = reg.path(current)
    tok = Tok(str(mdir / "tokenizer.json"))
    sft, pairs = build_examples(rows, tok)
    if len(sft) + len(pairs) < a.min_examples:
        log.info("waiting for feedback: %d/%d usable examples", len(sft) + len(pairs), a.min_examples)
        return None

    device = pick_device(a.device)
    ckpt = torch.load(mdir / "model.pt", map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    champion = GPT(cfg)
    champion.load_state_dict(ckpt["model"])
    champion.to(device).eval()
    policy = copy.deepcopy(champion).train()
    block = cfg.block_size

    rng = random.Random(0)
    rng.shuffle(sft)
    rng.shuffle(pairs)
    n_hold = max(1, len(sft) // 5) if len(sft) >= 5 else 0
    hold_sft, train_sft = sft[:n_hold], sft[n_hold:] or sft
    p_hold = max(1, len(pairs) // 5) if len(pairs) >= 5 else 0
    hold_pairs, train_pairs = pairs[:p_hold], pairs[p_hold:] or pairs

    replay = TokenLoader(a.replay, "train", block, a.batch, device, seed=int(time.time())) if a.replay else None
    opt = policy.configure_optimizer(a.lr, 0.0, device_type=device.type)
    t0 = time.time()
    for step in range(a.steps):
        loss = torch.zeros((), device=device)
        if train_sft:
            batch = [(ids, m) for _, ids, m in rng.sample(train_sft, min(a.batch, len(train_sft)))]
            loss = loss + policy(*collate(batch, block, device))[1]
        if replay is not None:
            loss = loss + a.replay_weight * policy(*replay.get())[1]
        if train_pairs and a.dpo_weight > 0:
            loss = loss + a.dpo_weight * dpo_loss(policy, champion, rng.sample(train_pairs, min(4, len(train_pairs))),
                                                  block, device, a.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            log.info("continual step %d/%d loss=%.4f", step + 1, a.steps, loss.item())

    # ---- champion / challenger evaluation on held-out data ----
    eval_sft = [(ids, m) for _, ids, m in (hold_sft or train_sft)]
    res = {
        "feedback_loss": {"champion": feedback_loss(champion, eval_sft, block, device),
                          "challenger": feedback_loss(policy, eval_sft, block, device)},
        "replay_loss": ({"champion": replay_loss(champion, a.replay, block, device),
                         "challenger": replay_loss(policy, a.replay, block, device)} if a.replay else None),
        "preference_accuracy": ({"champion": preference_accuracy(champion, hold_pairs or train_pairs, block, device),
                                 "challenger": preference_accuracy(policy, hold_pairs or train_pairs, block, device)}
                                if pairs else None),
        "examples": {"sft": len(sft), "pairs": len(pairs), "holdout_sft": len(hold_sft)},
        "steps": a.steps, "minutes": round((time.time() - t0) / 60, 2), "base_version": current,
    }
    fb_ok = res["feedback_loss"]["challenger"] < res["feedback_loss"]["champion"]
    rp_ok = (res["replay_loss"] is None or
             res["replay_loss"]["challenger"] <= res["replay_loss"]["champion"] * (1 + a.regression_tolerance))
    promote = fb_ok and rp_ok

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        torch.save({"model": policy.state_dict(), "config": cfg.to_dict(),
                    "val_loss": res["replay_loss"]["challenger"] if res["replay_loss"] else None}, path)
        version = reg.publish(path, mdir / "tokenizer.json",
                              {"stage": "continual", "parent": current, "promoted": promote, "eval": res},
                              make_current=promote)
    db.mark_llm_feedback_used([r["completion_id"] for r in rows], version if promote else f"rejected:{version}")
    fl = res["feedback_loss"]
    rl = res["replay_loss"] or {}
    msg = (f"{version} (from {current}) on {len(sft)} SFT + {len(pairs)} preference examples: feedback loss "
           f"{fl['champion']:.3f}->{fl['challenger']:.3f}"
           + (f", general loss {rl['champion']:.3f}->{rl['challenger']:.3f}" if rl else "")
           + (" -> PROMOTED" if promote else " -> rejected, champion kept"))
    db.log_event(EVENT_PROJECT, "promote" if promote else "reject", msg)
    log.info(msg)
    return {"version": version, "promoted": promote, **res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="API data dir (holds autolearn.db)")
    ap.add_argument("--llm-dir", required=True)
    ap.add_argument("--replay", default=None, help="SFT shard dir for replay (anti-forgetting)")
    ap.add_argument("--interval", type=float, default=600)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--min-examples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--replay-weight", type=float, default=1.0)
    ap.add_argument("--dpo-weight", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--regression-tolerance", type=float, default=0.02)
    ap.add_argument("--retention-days", type=int, default=30)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    while True:
        try:
            out = run_once(a)
            if out:
                print(json.dumps(out, indent=2, default=str), flush=True)
        except Exception:
            log.exception("continual learning cycle failed")
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
