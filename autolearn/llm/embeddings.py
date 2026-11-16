"""Text embeddings: a BERT-style encoder written in PyTorch (no `transformers` dependency) that loads published
sentence-embedding checkpoints such as BAAI/bge-small-en-v1.5 (MIT, 33M parameters, 384 dimensions).

Served at POST /v1/embeddings (OpenAI-compatible) and used by knowledge bases for hybrid (keyword + meaning)
search.

    python -m autolearn.llm.embeddings add BAAI/bge-small-en-v1.5 --name autolearn-embed
    python -m autolearn.llm.embeddings similar autolearn-embed "how do I get my money back" "refund policy" "shipping"
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MANIFEST = "autolearn.json"


class _Embeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.word_embeddings = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.position_embeddings = nn.Embedding(c["max_position_embeddings"], c["hidden_size"])
        self.token_type_embeddings = nn.Embedding(c["type_vocab_size"], c["hidden_size"])
        self.LayerNorm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.word_embeddings(ids) + self.position_embeddings(pos)[None] + self.token_type_embeddings.weight[0]
        return self.LayerNorm(x)


class _SelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c["hidden_size"]
        self.nh = c["num_attention_heads"]
        self.query, self.key, self.value = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)


class _AttnOutput(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c["hidden_size"], c["hidden_size"])
        self.LayerNorm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])


class _Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self, self.output = _SelfAttention(c), _AttnOutput(c)

    def forward(self, x, mask):
        B, T, H = x.shape
        s = self.self
        split = lambda t: t.view(B, T, s.nh, H // s.nh).transpose(1, 2)  # noqa: E731
        y = F.scaled_dot_product_attention(split(s.query(x)), split(s.key(x)), split(s.value(x)), attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, H)
        return self.output.LayerNorm(x + self.output.dense(y))  # post-LN (original BERT)


class _Dense(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.dense = nn.Linear(i, o)


class _Output(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c["intermediate_size"], c["hidden_size"])
        self.LayerNorm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])


class _Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = _Attention(c)
        self.intermediate = _Dense(c["hidden_size"], c["intermediate_size"])
        self.output = _Output(c)

    def forward(self, x, mask):
        x = self.attention(x, mask)
        h = F.gelu(self.intermediate.dense(x))
        return self.output.LayerNorm(x + self.output.dense(h))


class _LayerList(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer = nn.ModuleList(_Layer(c) for _ in range(c["num_hidden_layers"]))


class BertEncoder(nn.Module):
    """Parameter names match published BERT checkpoints, so weights load without renaming."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embeddings = _Embeddings(config)
        self.encoder = _LayerList(config)

    def forward(self, ids, attention_mask):
        mask = attention_mask[:, None, None, :].bool()  # (B, 1, 1, T): attend to real tokens only
        x = self.embeddings(ids)
        for layer in self.encoder.layer:
            x = layer(x, mask)
        return x


class Embedder:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        model_dir = Path(model_dir)
        self.manifest = json.loads((model_dir / MANIFEST).read_text()) if (model_dir / MANIFEST).exists() else {}
        cfg = json.loads((model_dir / "config.json").read_text())
        if (cfg.get("architectures") or ["?"])[0] not in ("BertModel", "BertForMaskedLM"):
            raise ValueError("only BERT-architecture embedding models are supported")
        self.model = BertEncoder(cfg)
        sd = {k.removeprefix("bert."): v for k, v in load_file(str(model_dir / "model.safetensors")).items()}
        missing, _ = self.model.load_state_dict(sd, strict=False)  # pooler / position_ids are unused
        if missing:
            raise ValueError(f"checkpoint is missing weights: {missing[:5]}")
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.max_len = min(512, cfg["max_position_embeddings"])
        self.tok.enable_truncation(self.max_len)
        self.dim = cfg["hidden_size"]
        self.model_id = self.manifest.get("id", model_dir.name)
        self.query_prefix = self.manifest.get("query_prefix", "")
        self.lock = threading.Lock()

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(e.ids) for e in self.tok.encode_batch(texts))

    @torch.no_grad()
    def encode(self, texts: list[str], query: bool = False, batch_size: int = 32) -> np.ndarray:
        """L2-normalised embeddings (cosine similarity = dot product). `query=True` adds the model's retrieval
        instruction, recommended for short search queries matched against passages."""
        if query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        out = []
        with self.lock:
            # sort by length so each batch pads as little as possible
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            for s in range(0, len(texts), batch_size):
                idx = order[s:s + batch_size]
                enc = self.tok.encode_batch([texts[i] for i in idx])
                T = max(len(e.ids) for e in enc)
                ids = torch.zeros(len(enc), T, dtype=torch.long)
                am = torch.zeros(len(enc), T, dtype=torch.long)
                for j, e in enumerate(enc):
                    ids[j, :len(e.ids)] = torch.tensor(e.ids)
                    am[j, :len(e.ids)] = 1
                h = self.model(ids.to(self.device), am.to(self.device))
                v = F.normalize(h[:, 0].float(), dim=-1)  # CLS pooling (bge)
                out.append((idx, v.cpu().numpy()))
        res = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, v in out:
            res[idx] = v
        return res


def find_embedder(root: str | Path | None) -> Path | None:
    """First registered embedding model under root."""
    if root and Path(root).exists():
        for d in sorted(Path(root).iterdir()):
            f = d / MANIFEST
            if f.exists() and json.loads(f.read_text()).get("type") == "embedding":
                return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="models/open")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("add")
    a_.add_argument("repo")
    a_.add_argument("--name", required=True)
    a_.add_argument("--query-prefix", default="Represent this sentence for searching relevant passages: ")
    s_ = sub.add_parser("similar")
    s_.add_argument("name")
    s_.add_argument("query")
    s_.add_argument("candidates", nargs="+")
    a = ap.parse_args()
    if a.cmd == "add":
        os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())
        from huggingface_hub import snapshot_download
        d = Path(a.root) / a.name
        snapshot_download(a.repo, local_dir=str(d), allow_patterns=["config.json", "tokenizer.json",
                                                                    "model.safetensors", "LICENSE*", "README.md"])
        (d / MANIFEST).write_text(json.dumps({"id": a.name, "type": "embedding", "repo": a.repo,
                                              "query_prefix": a.query_prefix}, indent=2))
        e = Embedder(d)
        print(json.dumps({"id": a.name, "dimensions": e.dim, "repo": a.repo}))
    else:
        e = Embedder(Path(a.root) / a.name)
        q = e.encode([a.query], query=True)
        c = e.encode(a.candidates)
        for text, s in sorted(zip(a.candidates, (c @ q[0]).tolist()), key=lambda x: -x[1]):
            print(f"{s:.3f}  {text}")


if __name__ == "__main__":
    main()
