"""Inference engine (KV-cached sampling with streaming) and the versioned LLM model registry."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Iterator

import torch

from .model import GPT, GPTConfig
from .tokenizer import Tok


class LLMRegistry:
    """Versioned model store:  <root>/versions/v<N>/{model.pt, tokenizer.json, meta.json}, <root>/CURRENT."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)

    def versions(self) -> list[dict]:
        out = []
        for d in sorted(self.root.glob("versions/v*"), key=lambda p: int(p.name[1:])):
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            out.append({"version": d.name, **meta})
        return out

    def current(self) -> str | None:
        f = self.root / "CURRENT"
        return f.read_text().strip() if f.exists() else None

    def path(self, version: str) -> Path:
        return self.root / "versions" / version

    def publish(self, checkpoint: str | Path, tokenizer: str | Path, meta: dict, make_current=True) -> str:
        """Export weights only (no optimizer state) as a new immutable version."""
        nums = [int(v["version"][1:]) for v in self.versions()]
        version = f"v{(max(nums) if nums else 0) + 1}"
        d = self.path(version)
        d.mkdir(parents=True)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        torch.save({"model": ckpt["model"], "config": ckpt["config"]}, d / "model.pt")
        shutil.copy(tokenizer, d / "tokenizer.json")
        params = sum(t.numel() for k, t in ckpt["model"].items() if k != "lm_head.weight")
        (d / "meta.json").write_text(json.dumps({**meta, "params": params, "created_at": time.time(),
                                                 "val_loss": ckpt.get("val_loss")}, indent=2, default=str))
        if make_current:
            self.set_current(version)
        return version

    def set_current(self, version: str):
        if not self.path(version).exists():
            raise FileNotFoundError(version)
        tmp = self.root / "CURRENT.tmp"
        tmp.write_text(version)
        tmp.replace(self.root / "CURRENT")


class Engine:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        model_dir = Path(model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
        self.config = GPTConfig(**ckpt["config"])
        self.model = GPT(self.config)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.tok = Tok(str(model_dir / "tokenizer.json"))
        self.lock = threading.Lock()  # one generation at a time per device
        self.version = model_dir.name

    # ---------------------------------------------------------------- core --
    @torch.no_grad()
    def _sample_ids(self, prompt: list[int], max_new: int, temperature: float, top_p: float, top_k: int,
                    repetition_penalty: float, stop_ids: set[int], seed: int | None) -> Iterator[int]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed if seed is not None else int(time.time_ns() % 2**31))
        caches = [{} for _ in range(self.config.n_layer)]
        idx = torch.tensor([prompt], dtype=torch.long, device=self.device)
        logits, _ = self.model(idx, caches=caches, start_pos=0)
        pos, seen = len(prompt), list(prompt)
        for _ in range(max_new):
            l = logits[0, -1].float().cpu()
            if repetition_penalty != 1.0 and seen:
                ids = torch.tensor(sorted(set(seen[-256:])))
                vals = l[ids]
                l[ids] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            if temperature <= 0:
                nxt = int(l.argmax())
            else:
                l = l / temperature
                if top_k:
                    kth = torch.topk(l, min(top_k, l.numel())).values[-1]
                    l[l < kth] = -float("inf")
                probs = torch.softmax(l, -1)
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cut = torch.cumsum(sp, 0) - sp > top_p
                    sp[cut] = 0
                    probs = torch.zeros_like(probs).scatter_(0, si, sp)
                    probs /= probs.sum()
                nxt = int(torch.multinomial(probs, 1, generator=gen))
            if nxt in stop_ids:
                return
            yield nxt
            seen.append(nxt)
            if pos >= self.config.block_size:
                return
            logits, _ = self.model(torch.tensor([[nxt]], device=self.device), caches=caches, start_pos=pos)
            pos += 1

    def stream(self, prompt_ids: list[int], max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9,
               top_k: int = 50, repetition_penalty: float = 1.1, stop: list[str] | None = None,
               seed: int | None = None) -> Iterator[dict]:
        """Yields {'delta': str} chunks, then a final {'finish_reason', 'prompt_tokens', 'completion_tokens'}."""
        max_tokens = max(1, min(max_tokens, self.config.block_size - 1))
        keep = self.config.block_size - max_tokens
        if len(prompt_ids) > keep:  # context window: drop the oldest tokens
            prompt_ids = [prompt_ids[0]] + prompt_ids[-(keep - 1):]
        stop_ids = {self.tok.end, self.tok.eos}
        out_ids, text_sent, finish = [], "", "length"
        with self.lock:
            for tid in self._sample_ids(prompt_ids, max_tokens, temperature, top_p, top_k, repetition_penalty,
                                        stop_ids, seed):
                out_ids.append(tid)
                text = self.tok.decode(out_ids)
                if text.endswith("�"):
                    continue  # partial UTF-8 character; wait for the rest
                if stop:
                    hit = min((text.find(s) for s in stop if s and s in text), default=-1)
                    if hit >= 0:
                        if hit > len(text_sent):
                            yield {"delta": text[len(text_sent):hit]}
                        finish = "stop"
                        break
                if len(text) > len(text_sent):
                    yield {"delta": text[len(text_sent):]}
                    text_sent = text
            else:
                finish = "length" if len(out_ids) >= max_tokens else "stop"
        yield {"finish_reason": finish, "prompt_tokens": len(prompt_ids), "completion_tokens": len(out_ids)}

    # ------------------------------------------------------------ helpers --
    def chat_stream(self, messages: list[dict], **kw):
        ids, _ = self.tok.render_chat(messages, add_generation_prompt=True)
        return self.stream(ids, **kw)

    def complete_stream(self, prompt: str, **kw):
        return self.stream([self.tok.bos] + self.tok.encode(prompt), **kw)

    def chat(self, messages: list[dict], **kw) -> dict:
        text, final = "", {}
        for ev in self.chat_stream(messages, **kw):
            if "delta" in ev:
                text += ev["delta"]
            else:
                final = ev
        return {"text": text, **final}
