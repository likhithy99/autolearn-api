"""Data pipeline: stream huge public corpora, clean them, train the tokenizer, and write token shards.

    # pretraining corpus (FineWeb-Edu: 1.3T tokens of educational web text, ODC-By)
    python -m autolearn.llm.data pretrain --out data/llm/pretrain --tokens 300M

    # chat / instruction data (SmolTalk, Apache-2.0)
    python -m autolearn.llm.data sft --out data/llm/sft --tokenizer data/llm/pretrain/tokenizer.json

The corpus is streamed, never downloaded whole, so the only limit is --tokens (disk budget):
each token costs 2 bytes. 1B tokens ~= 2 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from .tokenizer import Tok, train_tokenizer

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
WS = re.compile(r"\s+")


def _ssl():
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def parse_count(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


class Cleaner:
    """Quality filtering, exact de-duplication and PII scrubbing."""

    def __init__(self, min_chars=300, max_chars=100_000):
        self.min_chars, self.max_chars = min_chars, max_chars
        self.seen: set[bytes] = set()
        self.stats = {"seen": 0, "kept": 0, "too_short": 0, "too_long": 0, "duplicate": 0, "low_quality": 0}

    def __call__(self, text: str) -> str | None:
        self.stats["seen"] += 1
        if len(text) < self.min_chars:
            self.stats["too_short"] += 1
            return None
        if len(text) > self.max_chars:
            self.stats["too_long"] += 1
            return None
        letters = sum(c.isalpha() for c in text[:5000]) / min(len(text), 5000)
        lines = text.splitlines() or [text]
        if letters < 0.6 or sum(len(l) < 20 for l in lines) / len(lines) > 0.7:
            self.stats["low_quality"] += 1  # mostly symbols/numbers, or boilerplate menus
            return None
        h = hashlib.blake2b(WS.sub(" ", text.lower()).encode(), digest_size=8).digest()
        if h in self.seen:
            self.stats["duplicate"] += 1
            return None
        self.seen.add(h)
        text = PHONE.sub("<PHONE>", EMAIL.sub("<EMAIL>", text))
        self.stats["kept"] += 1
        return text


class ShardWriter:
    def __init__(self, out: Path, split: str, shard_tokens: int = 100_000_000, with_mask=False):
        self.out, self.split, self.shard_tokens, self.with_mask = out, split, shard_tokens, with_mask
        self.idx, self.in_shard, self.total = 0, 0, 0
        self.starts: list[int] = []

    def _path(self, ext):
        return self.out / f"{self.split}_{self.idx:04d}.{ext}"

    def write(self, ids: list[int], mask: list[int] | None = None):
        if self.in_shard + len(ids) > self.shard_tokens and self.in_shard:
            self._flush_index()
            self.idx, self.in_shard = self.idx + 1, 0
        self.starts.append(self.in_shard)
        with open(self._path("bin"), "ab") as f:
            f.write(np.asarray(ids, dtype=np.uint16).tobytes())
        if self.with_mask:
            with open(self._path("mask"), "ab") as f:
                f.write(np.asarray(mask, dtype=np.uint8).tobytes())
        self.in_shard += len(ids)
        self.total += len(ids)

    def _flush_index(self):
        if self.with_mask and self.starts:
            np.asarray(self.starts, dtype=np.int64).tofile(self._path("idx"))
        self.starts = []

    def close(self):
        self._flush_index()


def _fresh(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*_*.*"):
        if f.suffix in (".bin", ".mask", ".idx"):
            f.unlink()


def prepare_pretrain(a):
    _ssl()
    from datasets import load_dataset
    out = Path(a.out)
    _fresh(out)
    target, val_target = parse_count(a.tokens), parse_count(a.val_tokens)
    if a.data_files:  # pick specific parquet files (e.g. a different slice per training session)
        ds = load_dataset(a.dataset, data_files=a.data_files.split(","), split="train", streaming=True)
    else:
        ds = load_dataset(a.dataset, name=a.subset, split="train", streaming=True)
    if a.shuffle_buffer:
        ds = ds.shuffle(seed=0, buffer_size=a.shuffle_buffer)
    cleaner = Cleaner()
    it = (cleaner(ex[a.text_field]) for ex in ds if not a.lang or ex.get("language", a.lang) == a.lang)
    it = (t for t in it if t)

    t0 = time.time()
    tok_path = out / "tokenizer.json"
    buffered: list[str] = []
    if a.tokenizer:
        tok = Tok(a.tokenizer)
    elif tok_path.exists() and not a.retrain_tokenizer:
        tok = Tok(str(tok_path))
    else:
        print(f"collecting {a.tokenizer_docs} documents to train a {a.vocab}-token BPE vocabulary...", flush=True)
        for t in it:
            buffered.append(t)
            if len(buffered) >= a.tokenizer_docs:
                break
        tok = train_tokenizer(buffered, a.vocab, str(tok_path))
        print(f"tokenizer trained in {time.time() - t0:.0f}s (vocab={tok.vocab_size})", flush=True)
    assert tok.vocab_size <= 65535, "uint16 shards require vocab <= 65535"

    train_w, val_w = ShardWriter(out, "train", a.shard_tokens), ShardWriter(out, "val", a.shard_tokens)

    def docs():
        yield from buffered
        yield from it

    batch, n_docs, last = [], 0, time.time()
    for text in docs():
        batch.append(text)
        if len(batch) < 512:
            continue
        for ids in tok.encode_batch(batch):
            n_docs += 1
            w = val_w if (n_docs % 200 == 0 and val_w.total < val_target) else train_w
            w.write(ids + [tok.eos])
        batch = []
        if time.time() - last > 10:
            rate = train_w.total / (time.time() - t0)
            print(f"  {train_w.total / 1e6:,.1f}M / {target / 1e6:,.0f}M train tokens  "
                  f"({rate / 1e3:,.0f}K tok/s, {n_docs:,} docs)", flush=True)
            last = time.time()
        if train_w.total >= target:
            break
    train_w.close()
    val_w.close()
    manifest = {"dataset": a.dataset, "subset": a.subset, "train_tokens": train_w.total,
                "val_tokens": val_w.total, "documents": n_docs, "vocab_size": tok.vocab_size,
                "tokenizer": str(tok_path if not a.tokenizer else a.tokenizer), "cleaning": cleaner.stats,
                "seconds": round(time.time() - t0)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def prepare_sft(a):
    _ssl()
    from datasets import load_dataset
    out = Path(a.out)
    _fresh(out)
    tok = Tok(a.tokenizer)
    ds = load_dataset(a.dataset, name=a.subset, split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    train_w = ShardWriter(out, "train", with_mask=True)
    val_w = ShardWriter(out, "val", with_mask=True)
    n, skipped, t0 = 0, 0, time.time()
    for ex in ds:
        msgs = ex.get(a.messages_field) or []
        if not any(m.get("role") == "assistant" for m in msgs):
            skipped += 1
            continue
        ids, mask = tok.render_chat(msgs, add_generation_prompt=False)
        if len(ids) > a.max_len:
            skipped += 1
            continue
        n += 1
        (val_w if n % 50 == 0 else train_w).write(ids, mask)
        if n % 20000 == 0:
            print(f"  {n:,} conversations, {train_w.total / 1e6:.1f}M tokens", flush=True)
        if n >= a.max_conversations:
            break
    train_w.close()
    val_w.close()
    manifest = {"dataset": a.dataset, "subset": a.subset, "conversations": n, "skipped": skipped,
                "train_tokens": train_w.total, "val_tokens": val_w.total, "tokenizer": a.tokenizer,
                "seconds": round(time.time() - t0)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pretrain")
    p.add_argument("--out", default="data/llm/pretrain")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--data-files", default=None, help="comma-separated repo file paths, overrides --subset")
    p.add_argument("--text-field", default="text")
    p.add_argument("--lang", default="en")
    p.add_argument("--tokens", default="300M", help="training-token budget, e.g. 300M, 10B")
    p.add_argument("--val-tokens", default="2M")
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--tokenizer", default=None, help="reuse an existing tokenizer.json")
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--tokenizer-docs", type=int, default=40_000)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--shuffle-buffer", type=int, default=0)
    p.set_defaults(fn=prepare_pretrain)
    s = sub.add_parser("sft")
    s.add_argument("--out", default="data/llm/sft")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    s.add_argument("--subset", default="all")
    s.add_argument("--messages-field", default="messages")
    s.add_argument("--max-conversations", type=int, default=200_000)
    s.add_argument("--max-len", type=int, default=2048)
    s.set_defaults(fn=prepare_sft)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
