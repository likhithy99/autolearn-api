"""Per-customer custom models: each company gets its own LoRA adapter on top of a shared open-weight model,
trained only on that company's data and improved continually.

Training data per (company, model):
  * examples the company uploads   POST /v1/custom-models/{model}/examples   (ideal conversations)
  * feedback on that model's answers (if the company opted in to training):
      correction -> learn the corrected answer;  thumbs-up -> reinforce the answer;
      thumbs-down + correction -> preference pair for DPO (prefer the correction over the original)

Each training cycle:
  1. starts from the company's current adapter (or a fresh one),
  2. trains the adapter on new data + an "anchor" set (the base model's own answers to general prompts) so
     general ability is not lost,
  3. promotes the new adapter only if loss on the company's held-out examples improves AND anchor loss does
     not regress by more than the tolerance,
  4. publishes an immutable version (promoted or not, for audit). The API picks it up on the next request.

Worker (runs next to the API, ideally on the GPU box):
    python -m autolearn.llm.custom --data-dir data --open-dir models/open
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..db import DB
from .continual import build_examples, collate
from .lora import get_adapter, lora_parameters, new_adapter, set_adapter

log = logging.getLogger("autolearn.llm.custom")

ANCHOR_PROMPTS = [
    "Explain what photosynthesis is in two sentences.",
    "Give me three tips for writing a clear email.",
    "What is the difference between a list and a tuple in Python?",
    "Summarise the causes of World War I briefly.",
    "Write a short, friendly reply thanking a customer for their order.",
    "How do I convert Celsius to Fahrenheit?",
    "What are the main benefits of regular exercise?",
    "Translate 'Good morning, how are you?' into Spanish.",
    "Write a haiku about autumn.",
    "What is a good way to prepare for a job interview?",
    "Explain compound interest with a simple example.",
    "List the planets of the solar system in order.",
    "What does an API do, in simple terms?",
    "Suggest a name for a small coffee shop.",
    "How can I improve my sleep?",
    "What is machine learning?",
]


class AdapterStore:
    """<root>/<tenant>/<model>/versions/vN/{adapter.pt, meta.json}, CURRENT (absent = base model)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: dict = {}
        self._lock = threading.Lock()

    def dir(self, tenant: str, model: str) -> Path:
        return self.root / tenant / model

    def versions(self, tenant, model) -> list[dict]:
        d = self.dir(tenant, model) / "versions"
        out = []
        for v in sorted(d.glob("v*"), key=lambda p: int(p.name[1:])) if d.exists() else []:
            meta = json.loads((v / "meta.json").read_text()) if (v / "meta.json").exists() else {}
            out.append({"version": v.name, **meta})
        return out

    def current(self, tenant, model) -> str | None:
        f = self.dir(tenant, model) / "CURRENT"
        return f.read_text().strip() or None if f.exists() else None

    def set_current(self, tenant, model, version: str | None):
        d = self.dir(tenant, model)
        if version is not None and not (d / "versions" / version / "adapter.pt").exists():
            raise FileNotFoundError(version)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "CURRENT.tmp"
        tmp.write_text(version or "")
        tmp.replace(d / "CURRENT")

    def publish(self, tenant, model, state: dict, meta: dict, make_current: bool) -> str:
        d = self.dir(tenant, model) / "versions"
        d.mkdir(parents=True, exist_ok=True)
        nums = [int(p.name[1:]) for p in d.glob("v*")]
        version = f"v{(max(nums) if nums else 0) + 1}"
        (d / version).mkdir()
        torch.save(state, d / version / "adapter.pt")
        (d / version / "meta.json").write_text(json.dumps({**meta, "created_at": time.time()}, indent=2,
                                                          default=str))
        if make_current:
            self.set_current(tenant, model, version)
        return version

    def load(self, tenant, model, version) -> dict:
        return torch.load(self.dir(tenant, model) / "versions" / version / "adapter.pt", map_location="cpu",
                          weights_only=True)

    def active(self, tenant, model) -> tuple[str | None, dict | None]:
        """(version, adapter state) currently serving for this tenant; cached until CURRENT changes."""
        f = self.dir(tenant, model) / "CURRENT"
        if not f.exists():
            return None, None
        key = (tenant, model)
        mtime = f.stat().st_mtime_ns
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == mtime:
                return hit[1], hit[2]
        version = self.current(tenant, model)
        state = self.load(tenant, model, version) if version else None
        with self._lock:
            self._cache[key] = (mtime, version, state)
        return version, state

    def request_training(self, tenant, model):
        d = self.dir(tenant, model)
        d.mkdir(parents=True, exist_ok=True)
        (d / "TRAIN_REQUESTED").write_text(str(time.time()))

    def training_requested(self, tenant, model) -> bool:
        return (self.dir(tenant, model) / "TRAIN_REQUESTED").exists()

    def clear_request(self, tenant, model):
        (self.dir(tenant, model) / "TRAIN_REQUESTED").unlink(missing_ok=True)

    def requested(self) -> list[tuple[str, str]]:
        return [(f.parent.parent.name, f.parent.name) for f in self.root.glob("*/*/TRAIN_REQUESTED")]

    def delete(self, tenant, model):
        shutil.rmtree(self.dir(tenant, model), ignore_errors=True)
        with self._lock:
            self._cache.pop((tenant, model), None)


def validate_conversation(messages: list[dict]) -> str | None:
    """Returns an error message, or None if the conversation is a usable training example."""
    if not messages or messages[-1].get("role") != "assistant":
        return "each example must end with the ideal assistant reply"
    if not any(m.get("role") == "user" for m in messages):
        return "each example needs at least one user message"
    if any(m.get("role") not in ("system", "user", "assistant") for m in messages):
        return "roles must be system, user or assistant"
    if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        return "every message needs non-empty text content"
    return None


# ------------------------------------------------------------------ training --
def anchors(engine, store_root: Path, n: int, max_tokens: int) -> list[tuple[list[int], list[int]]]:
    """Base-model answers to general prompts, cached per base model (self-distillation replay set)."""
    f = Path(store_root) / "_anchors" / f"{engine.model_id}.json"
    if f.exists():
        data = json.loads(f.read_text())
    else:
        set_adapter(engine.model, None)
        data = []
        for p in ANCHOR_PROMPTS[:n]:
            msgs = [{"role": "user", "content": p}]
            ans = engine.chat(msgs, max_tokens=max_tokens, temperature=0.0, seed=0)["text"].strip()
            data.append(msgs + [{"role": "assistant", "content": ans}])
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data))
    return [engine.tok.render_chat(m, add_generation_prompt=False) for m in data]


def _loss(model, examples, block, device, bs=4):
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), bs):
            x, y = collate(examples[i:i + bs], block, device)
            _, loss = model(x, y)
            k = int((y != -100).sum())
            tot, n = tot + loss.item() * k, n + k
    return tot / max(n, 1)


def _seq_logprobs(model, ex, block, device):
    x, y = collate([ex], block, device)
    return model.sequence_logprobs(x, y)


def train_custom(engine, db: DB, store: AdapterStore, tenant: str, *, min_examples=8, force=False, steps=None,
                 lr=2e-4, batch=4, block=1024, anchor_n=16, anchor_tokens=128, anchor_weight=0.5,
                 dpo_weight=0.5, beta=0.1, tolerance=0.05) -> dict | None:
    model_id = engine.model_id
    t = db.get_tenant(tenant) or {}
    fb = db.unused_llm_feedback(model_id, tenant) if t.get("train_opt_in") else []
    ex_rows = db.unused_tuning_examples(tenant, model_id)
    tok, model, device = engine.tok, engine.model, engine.device
    sft, pairs = build_examples(fb, tok)
    sft = [(ids, m) for _, ids, m in sft]
    for r in ex_rows:
        sft.append(tok.render_chat(r["messages"], add_generation_prompt=False))
    n_new = len(sft) + len(pairs)
    if n_new == 0 or (n_new < min_examples and not force):
        return None
    block = min(block, engine.config.block_size)

    with engine.lock:  # in-process callers share the serving engine
        anchor = anchors(engine, store.root, anchor_n, anchor_tokens)
        engine._loaded_adapter = 0
        base_version, champion = store.active(tenant, model_id)
        rng = random.Random(0)
        rng.shuffle(sft)
        n_hold = max(1, len(sft) // 5) if len(sft) >= 5 else 0
        hold, train = sft[:n_hold], sft[n_hold:] or sft
        pairs = [(c, r) for _, c, r in pairs]

        def use(state):
            set_adapter(model, state) if state is not None else set_adapter(model, None)

        # reference log-probs for DPO, computed once with the champion adapter
        use(champion)
        model.eval()
        ref = []
        with torch.no_grad():
            for c, r in pairs:
                ref.append((_seq_logprobs(model, c, block, device).item(), _seq_logprobs(model, r, block, device).item()))
        eval_set = hold or train
        champ = {"examples_loss": _loss(model, eval_set, block, device), "anchor_loss": _loss(model, anchor, block, device)}

        # challenger starts from the champion adapter (cumulative learning) or a fresh one
        if champion is not None:
            set_adapter(model, champion)
        else:
            new_adapter(model)
        params = lora_parameters(model)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        steps = steps or int(min(300, max(30, 3 * len(train) / batch)))
        model.train()
        t0 = time.time()
        for step in range(steps):
            b = rng.sample(train, min(batch, len(train)))
            loss = model(*collate(b, block, device))[1]
            if anchor and anchor_weight > 0:
                loss = loss + anchor_weight * model(*collate(rng.sample(anchor, min(2, len(anchor))), block,
                                                             device))[1]
            if pairs and dpo_weight > 0:
                i = rng.randrange(len(pairs))
                (c, r), (rc, rr) = pairs[i], ref[i]
                pc, pr = _seq_logprobs(model, c, block, device), _seq_logprobs(model, r, block, device)
                loss = loss + dpo_weight * -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        model.eval()
        for p in params:
            p.requires_grad_(False)
        state = get_adapter(model)
        chall = {"examples_loss": _loss(model, eval_set, block, device), "anchor_loss": _loss(model, anchor, block, device)}
        use(champion)
        engine._loaded_adapter = 0  # weights changed in place; force the next request to reload its adapter

    better = chall["examples_loss"] < champ["examples_loss"]
    safe = chall["anchor_loss"] <= champ["anchor_loss"] * (1 + tolerance)
    promote = better and safe
    meta = {"base_model": model_id, "parent": base_version, "promoted": promote,
            "eval": {"champion": champ, "challenger": chall, "holdout_examples": len(hold),
                     "tolerance": tolerance},
            "data": {"examples": len(ex_rows), "feedback": len(fb), "sft": len(sft), "preference_pairs": len(pairs)},
            "steps": steps, "seconds": round(time.time() - t0, 1),
            "lora": {"r": model.blocks[0].attn.wq.r}}
    version = store.publish(tenant, model_id, state, meta, make_current=promote)
    tag = version if promote else f"rejected:{version}"
    db.mark_tuning_examples_used([r["id"] for r in ex_rows], tag)
    db.mark_llm_feedback_used([r["completion_id"] for r in fb], tag)
    store.clear_request(tenant, model_id)
    msg = (f"custom model {model_id} {version} for {tenant}: examples loss {champ['examples_loss']:.3f}->"
           f"{chall['examples_loss']:.3f}, general (anchor) loss {champ['anchor_loss']:.3f}->"
           f"{chall['anchor_loss']:.3f} -> {'PROMOTED' if promote else 'rejected, previous kept'}")
    db.log_event(f"__custom__/{tenant}", "promote" if promote else "reject", msg)
    log.info(msg)
    return {"version": version, **meta}


# -------------------------------------------------------------------- worker --
def run_once(a, engines: dict) -> list[dict]:
    from .engine import Engine
    from .open_models import catalog
    from ..config import PLANS
    db = DB(Path(a.data_dir) / "autolearn.db")
    store = AdapterStore(Path(a.data_dir) / "adapters")
    cat = catalog(a.open_dir)
    todo = {(r["tenant"], r["model"]) for r in db.tenants_with_pending_tuning(list(cat))}
    todo |= {(t, m) for t, m in store.requested() if m in cat}
    results = []
    for tenant, model_id in sorted(todo):
        t = db.get_tenant(tenant)
        if not t or not t.get("active", 1) or not PLANS.get(t["plan"], {}).get("self_training"):
            continue
        if model_id not in engines:
            engines[model_id] = Engine.open_model(cat[model_id]["path"], a.device)
        try:
            out = train_custom(engines[model_id], db, store, tenant, min_examples=a.min_examples,
                               force=store.training_requested(tenant, model_id), steps=a.steps)
            if out:
                results.append(out)
        except Exception:
            log.exception("custom model training failed for %s/%s", tenant, model_id)
    return results


def main():
    from .train import pick_device
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--open-dir", default="models/open")
    ap.add_argument("--interval", type=float, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--min-examples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    a.device = str(pick_device(a.device))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engines: dict = {}
    while True:
        for r in run_once(a, engines):
            print(json.dumps(r, default=str), flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
