"""Embedding model (BERT encoder) correctness, /v1/embeddings, and hybrid knowledge-base search."""

import base64
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from tokenizers.implementations import BertWordPieceTokenizer

from autolearn.llm.embeddings import MANIFEST, Embedder

H, L, NH, FF = 32, 2, 4, 64


@pytest.fixture(scope="module")
def embed_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    d = root / "open" / "tiny-embed"
    d.mkdir(parents=True)
    tok = BertWordPieceTokenizer(lowercase=True)
    tok.train_from_iterator(["refunds are paid within five days", "shipping costs fifteen dollars",
                             "how do i get my money back", "our office is closed on holidays"] * 20,
                            vocab_size=200, min_frequency=1)
    tok.save(str(d / "tokenizer.json"))
    V = tok.get_vocab_size()
    cfg = {"architectures": ["BertModel"], "vocab_size": V, "hidden_size": H, "num_hidden_layers": L,
           "num_attention_heads": NH, "intermediate_size": FF, "max_position_embeddings": 64, "type_vocab_size": 2,
           "layer_norm_eps": 1e-12}
    (d / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3
    w = {"embeddings.word_embeddings.weight": r(V, H), "embeddings.position_embeddings.weight": r(64, H),
         "embeddings.token_type_embeddings.weight": r(2, H), "embeddings.LayerNorm.weight": 1 + r(H),
         "embeddings.LayerNorm.bias": r(H), "pooler.dense.weight": r(H, H), "pooler.dense.bias": r(H)}
    for i in range(L):
        p = f"encoder.layer.{i}."
        for n in ("query", "key", "value"):
            w[p + f"attention.self.{n}.weight"], w[p + f"attention.self.{n}.bias"] = r(H, H), r(H)
        w |= {p + "attention.output.dense.weight": r(H, H), p + "attention.output.dense.bias": r(H),
              p + "attention.output.LayerNorm.weight": 1 + r(H), p + "attention.output.LayerNorm.bias": r(H),
              p + "intermediate.dense.weight": r(FF, H), p + "intermediate.dense.bias": r(FF),
              p + "output.dense.weight": r(H, FF), p + "output.dense.bias": r(H),
              p + "output.LayerNorm.weight": 1 + r(H), p + "output.LayerNorm.bias": r(H)}
    save_file(w, str(d / "model.safetensors"))
    (d / MANIFEST).write_text(json.dumps({"id": "tiny-embed", "type": "embedding", "query_prefix": ""}))
    return d


def reference(w, ids):
    """Plain BERT forward (post-LN), CLS pooled and L2-normalised."""
    ln = lambda x, p: F.layer_norm(x, (H,), w[p + ".weight"], w[p + ".bias"], 1e-12)
    lin = lambda x, p: x @ w[p + ".weight"].T + w[p + ".bias"]
    T = len(ids)
    x = ln(w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:T]
           + w["embeddings.token_type_embeddings.weight"][0], "embeddings.LayerNorm")
    hd = H // NH
    for i in range(L):
        p = f"encoder.layer.{i}."
        q, k, v = (lin(x, p + f"attention.self.{n}").view(T, NH, hd).transpose(0, 1) for n in ("query", "key", "value"))
        a = ((q @ k.transpose(1, 2)) / math.sqrt(hd)).softmax(-1) @ v
        x = ln(x + lin(a.transpose(0, 1).reshape(T, H), p + "attention.output.dense"), p + "attention.output.LayerNorm")
        hmid = lin(x, p + "intermediate.dense")
        hmid = 0.5 * hmid * (1 + torch.erf(hmid / math.sqrt(2)))
        x = ln(x + lin(hmid, p + "output.dense"), p + "output.LayerNorm")
    return F.normalize(x[0], dim=-1)


def test_encoder_matches_reference_and_padding(embed_dir):
    from safetensors.torch import load_file
    e = Embedder(embed_dir)
    texts = ["refunds are paid", "how do i get my money back on holidays please"]
    got = e.encode(texts)  # batched together -> the shorter one is padded
    w = load_file(str(embed_dir / "model.safetensors"))
    for t, v in zip(texts, got):
        want = reference(w, e.tok.encode(t).ids)
        assert np.allclose(v, want.numpy(), atol=1e-5)
    assert np.allclose(np.linalg.norm(got, axis=1), 1, atol=1e-5)


def test_embeddings_endpoint_and_hybrid_search(tmp_path, embed_dir):
    from autolearn.api import create_app
    from autolearn.config import Settings
    s = Settings(data_dir=tmp_path / "data", admin_key="adm", learn_interval=3600, start_learner=False,
                 open_models_dir=embed_dir.parent)
    with TestClient(create_app(s)) as c:
        t = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": "acme", "plan": "pro"}).json()
        h = {"X-API-Key": t["admin_key"]["api_key"]}
        r = c.post("/v1/embeddings", headers=h, json={"input": ["refunds are paid", "shipping costs"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "tiny-embed" and len(body["data"]) == 2 and body["usage"]["prompt_tokens"] > 0
        assert len(body["data"][0]["embedding"]) == H
        b64 = c.post("/v1/embeddings", headers=h, json={"input": "refunds are paid",
                                                         "encoding_format": "base64"}).json()
        v = np.frombuffer(base64.b64decode(b64["data"][0]["embedding"]), dtype="<f4")
        assert np.allclose(v, body["data"][0]["embedding"], atol=1e-6)
        assert c.post("/v1/embeddings", headers=h, json={"input": "x", "model": "nope"}).status_code == 404
        assert "tiny-embed" in [m["id"] for m in c.get("/v1/models", headers=h).json()["data"]]

        docs = [{"id": "refunds", "title": "Refunds", "text": "Refunds are paid within five days."},
                {"id": "ship", "title": "Shipping", "text": "Shipping costs fifteen dollars."},
                {"id": "office", "title": "Office", "text": "Our office is closed on holidays."}]
        c.post("/v1/knowledge/policies/documents", headers=h, json={"documents": docs})
        res = c.post("/v1/knowledge/policies/search", headers=h,
                     json={"query": "shipping costs", "top_k": 3}).json()["results"]
        assert res[0]["document_id"] == "ship" and res[0]["match"] == "both"
        # no shared keywords at all: only the embedding side can return anything
        res = c.post("/v1/knowledge/policies/search", headers=h,
                     json={"query": "zzqx", "top_k": 2}).json()["results"]
        assert len(res) == 2 and all(x["match"] == "semantic" for x in res)
        c.delete("/v1/knowledge/policies/documents/ship", headers=h)
        res = c.post("/v1/knowledge/policies/search", headers=h, json={"query": "zzqx", "top_k": 5}).json()["results"]
        assert {x["document_id"] for x in res} == {"refunds", "office"}  # vectors deleted with the document
