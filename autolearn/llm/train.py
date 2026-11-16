"""Training: pretraining from scratch and supervised chat fine-tuning (SFT).

Single device (Mac MPS / one GPU / CPU):
    python -m autolearn.llm.train --stage pretrain --preset micro --data data/llm/pretrain --out runs/pretrain
    python -m autolearn.llm.train --stage sft --init runs/pretrain/best.pt --data data/llm/sft --out runs/sft

Multi-GPU (one node, 8 GPUs) - identical code, data-parallel via DDP:
    torchrun --standalone --nproc_per_node=8 -m autolearn.llm.train --stage pretrain --preset medium ...

Features: cosine LR with warmup, gradient accumulation, gradient clipping, bf16 autocast on CUDA,
torch.compile (optional), wall-clock time budget, resumable checkpoints, JSONL metrics log.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .model import GPT, GPTConfig, config_from_preset
from .tokenizer import Tok


# ------------------------------------------------------------------- data ----
class TokenLoader:
    """Samples random training windows from memory-mapped token shards (constant memory at any scale)."""

    def __init__(self, data_dir: str, split: str, block: int, batch: int, device, seed: int = 0):
        d = Path(data_dir)
        self.files = sorted(d.glob(f"{split}_*.bin"))
        if not self.files:
            raise FileNotFoundError(f"no {split}_*.bin shards in {d}")
        self.tokens = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.masks = [np.memmap(f.with_suffix(".mask"), dtype=np.uint8, mode="r")
                      if f.with_suffix(".mask").exists() else None for f in self.files]
        self.starts = [np.fromfile(f.with_suffix(".idx"), dtype=np.int64)
                       if f.with_suffix(".idx").exists() else None for f in self.files]
        sizes = np.array([len(t) for t in self.tokens], dtype=np.float64)
        self.p = sizes / sizes.sum()
        self.block, self.batch, self.device = block, batch, device
        self.rng = np.random.default_rng(seed)
        self.n_tokens = int(sizes.sum())

    def get(self):
        xs, ys = [], []
        for _ in range(self.batch):
            i = self.rng.choice(len(self.tokens), p=self.p)
            arr, mask, starts = self.tokens[i], self.masks[i], self.starts[i]
            hi = len(arr) - self.block - 1
            if hi <= 0:
                raise ValueError(f"shard {self.files[i]} is shorter than block_size")
            for _ in range(20):
                if starts is not None:  # SFT: start windows at conversation boundaries
                    cand = starts[starts <= hi]
                    s = int(cand[self.rng.integers(len(cand))]) if len(cand) else int(self.rng.integers(hi))
                else:
                    s = int(self.rng.integers(hi))
                # SFT: skip windows with no assistant tokens to learn from (prompt longer than the window)
                if mask is None or mask[s + 1:s + 1 + self.block].any():
                    break
            x = torch.from_numpy(arr[s:s + self.block].astype(np.int64))
            y = torch.from_numpy(arr[s + 1:s + 1 + self.block].astype(np.int64))
            if mask is not None:  # only learn to produce assistant tokens
                m = torch.from_numpy(mask[s + 1:s + 1 + self.block].astype(np.bool_))
                y = torch.where(m, y, torch.full_like(y, -100))
            xs.append(x)
            ys.append(y)
        x, y = torch.stack(xs), torch.stack(ys)
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


# ---------------------------------------------------------------- helpers ----
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device):
    if device.type == "cuda":  # bf16 on Ampere+ (A100/H100/RTX30+); fp16 on older GPUs (T4/P100/V100)
        return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return contextlib.nullcontext()  # MPS/CPU: full fp32 for numerical safety


def lr_at(step, max_steps, warmup, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, iters, device):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.get()
        with autocast_ctx(device):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path, model, cfg, step, val_loss, meta, optimizer=None):
    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)
    ckpt = {"model": raw.state_dict(), "config": cfg.to_dict(), "step": step, "val_loss": val_loss, "meta": meta}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_model(path, device="cpu") -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="micro")
    ap.add_argument("--tokenizer", default=None, help="defaults to <data>/tokenizer.json or the manifest's")
    ap.add_argument("--init", default=None, help="checkpoint to start from (required for sft)")
    ap.add_argument("--resume", action="store_true", help="resume from <out>/last.pt")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16, help="micro-batch per device")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--time-budget-min", type=float, default=0, help="stop after N minutes (0 = off)")
    ap.add_argument("--stop-after-min", type=float, default=0,
                    help="save and exit after N minutes WITHOUT changing the LR schedule (multi-session runs)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    # distributed setup
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, local, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local}")
        torch.cuda.set_device(device)
    else:
        rank, world, device = 0, 1, pick_device(a.device)
    master = rank == 0
    torch.manual_seed(a.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    tok_path = a.tokenizer or data_manifest.get("tokenizer") or str(Path(a.data) / "tokenizer.json")
    tok = Tok(tok_path)

    # model
    start_step, best_val = 0, float("inf")
    if a.resume and (out / "last.pt").exists():
        model, ckpt = load_model(out / "last.pt", device)
        cfg, start_step, best_val = GPTConfig(**ckpt["config"]), ckpt["step"], ckpt.get("meta", {}).get("best_val", 1e9)
    elif a.init:
        model, ckpt = load_model(a.init, device)
        cfg = GPTConfig(**ckpt["config"])
    else:
        if a.stage == "sft":
            raise SystemExit("--init <pretrained checkpoint> is required for --stage sft")
        overrides = {"dropout": a.dropout}
        if a.block:
            overrides["block_size"] = a.block
        cfg = config_from_preset(a.preset, tok.vocab_size, **overrides)
        model = GPT(cfg).to(device)
    block = min(a.block or cfg.block_size, cfg.block_size)

    lr = a.lr or (6e-4 if a.stage == "pretrain" else 5e-5)
    warmup = a.warmup if a.warmup is not None else (min(500, a.max_steps // 20) if a.stage == "pretrain" else 50)
    optimizer = model.configure_optimizer(lr, a.wd, device_type=device.type)
    if a.resume and (out / "last.pt").exists() and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    # fp16 needs loss scaling to avoid underflow; bf16/fp32 do not
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    raw_model = model
    if a.compile and device.type == "cuda":
        model = torch.compile(model)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index])

    train_loader = TokenLoader(a.data, "train", block, a.batch, device, seed=a.seed + rank)
    val_loader = TokenLoader(a.data, "val", block, a.batch, device, seed=999)
    tokens_per_step = a.batch * block * a.accum * world
    if master:
        print(json.dumps({"stage": a.stage, "device": str(device), "world": world, "params_M":
                          round(raw_model.num_params() / 1e6, 2), "config": cfg.to_dict(),
                          "tokens_per_step": tokens_per_step, "train_tokens_available": train_loader.n_tokens}),
              flush=True)

    max_steps = a.max_steps
    meta = {"stage": a.stage, "tokenizer": str(Path(tok_path).resolve()), "data": data_manifest,
            "preset": a.preset}
    log_f = open(out / "metrics.jsonl", "a") if master else None
    t_start = time.time()
    t_last, tokens_seen = time.time(), 0
    model.train()
    step = start_step
    while step < max_steps:
        cur_lr = lr_at(step, max_steps, warmup, lr, lr * a.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        loss_accum = 0.0
        for micro in range(a.accum):
            x, y = train_loader.get()
            sync = (not ddp) or micro == a.accum - 1
            ctx = contextlib.nullcontext() if sync else model.no_sync()
            with ctx, autocast_ctx(device):
                _, loss = model(x, y)
                loss = loss / a.accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tokens_per_step

        # wall-clock budget: after a few steps, shrink the schedule to fit (keeps cosine decay intact)
        if a.time_budget_min and step == start_step + 10:
            per_step = (time.time() - t_start) / 10
            fit = start_step + int(a.time_budget_min * 60 / per_step)
            if fit < max_steps:
                max_steps = max(fit, step + 1)
                if master:
                    print(f"time budget {a.time_budget_min} min -> {max_steps} steps "
                          f"({per_step:.2f}s/step, ~{max_steps * tokens_per_step / 1e6:.0f}M tokens)", flush=True)

        if master and (step % 10 == 0 or step == max_steps):
            dt = time.time() - t_last
            rec = {"step": step, "loss": round(loss_accum, 4), "lr": round(cur_lr, 7),
                   "grad_norm": round(float(norm), 3), "tok_per_s": round(tokens_seen / dt),
                   "elapsed_min": round((time.time() - t_start) / 60, 2)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if step % 50 == 0:
                print(json.dumps(rec), flush=True)
            t_last, tokens_seen = time.time(), 0

        stop_now = bool(a.stop_after_min) and (time.time() - t_start) > a.stop_after_min * 60
        if ddp and a.stop_after_min:  # all ranks must agree to stop together
            flag = torch.tensor([int(stop_now)], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop_now = bool(flag.item())
        if master and (step % a.eval_every == 0 or step == max_steps or stop_now):
            val = evaluate(raw_model, val_loader, a.eval_iters, device)
            rec = {"step": step, "val_loss": round(val, 4), "val_ppl": round(math.exp(min(val, 20)), 2)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(json.dumps(rec), flush=True)
            meta["best_val"] = min(best_val, val)
            if val < best_val:
                best_val = val
                save_checkpoint(out / "best.pt", raw_model, cfg, step, val, meta)
            save_checkpoint(out / "last.pt", raw_model, cfg, step, val, meta, optimizer)
        if stop_now:
            if master:
                print(json.dumps({"session_stop": True, "step": step, "of": max_steps}), flush=True)
            break

    if master:
        print(json.dumps({"done": True, "steps": step, "best_val_loss": round(best_val, 4),
                          "minutes": round((time.time() - t_start) / 60, 1)}), flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
