"""LLM stack tests with a tiny model built on the fly (CPU, seconds)."""

import json
from argparse import Namespace

import pytest
import torch
from fastapi.testclient import TestClient

from autolearn.api import create_app
from autolearn.config import Settings
from autolearn.llm.continual import run_once
from autolearn.llm.data import Cleaner, ShardWriter
from autolearn.llm.engine import Engine, LLMRegistry
from autolearn.llm.model import GPT, GPTConfig
from autolearn.llm.tokenizer import train_tokenizer

ADMIN = "test-platform-key"
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models learn patterns from data.",
    "Paris is the capital of France. Berlin is the capital of Germany.",
    "Water boils at one hundred degrees Celsius at sea level.",
] * 50


@pytest.fixture(scope="module")
def llm_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("llm")
    tok = train_tokenizer(CORPUS, 400, str(root / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=2, n_kv_head=1, n_embd=64)
    model = GPT(cfg)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "val_loss": None}, root / "ckpt.pt")
    LLMRegistry(root / "registry").publish(root / "ckpt.pt", root / "tokenizer.json", {"stage": "test"})
    # replay shards for continual learning
    sft = root / "sft"
    sft.mkdir()
    for split in ("train", "val"):
        w = ShardWriter(sft, split, with_mask=True)
        for i in range(40):
            ids, mask = tok.render_chat([{"role": "user", "content": "What is the capital of France?"},
                                         {"role": "assistant", "content": "Paris is the capital of France."}],
                                        add_generation_prompt=False)
            w.write(ids, mask)
        w.close()
    (sft / "manifest.json").write_text(json.dumps({"tokenizer": str(root / "tokenizer.json")}))
    return root


@pytest.fixture
def ctx(tmp_path, llm_dir):
    s = Settings(data_dir=tmp_path, admin_key=ADMIN, learn_interval=3600, start_learner=False,
                 llm_dir=llm_dir / "registry", llm_device="cpu")
    app = create_app(s)
    with TestClient(app) as c:
        r = c.post("/v1/admin/tenants", json={"name": "Acme", "plan": "pro"}, headers={"X-API-Key": ADMIN}).json()
        yield c, {"Authorization": f"Bearer {r['admin_key']['api_key']}"}, app, s


def test_engine_generation_and_stop(llm_dir):
    e = Engine(llm_dir / "registry" / "versions" / "v1", "cpu")
    out = e.chat([{"role": "user", "content": "hello"}], max_tokens=20, temperature=0, seed=1)
    assert out["completion_tokens"] <= 20 and out["finish_reason"] in ("stop", "length")
    a = e.chat([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.9, seed=7)
    b = e.chat([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.9, seed=7)
    assert a["text"] == b["text"]  # seeded sampling is reproducible


def test_chat_completion_openai_format(ctx):
    c, h, _, _ = ctx
    models = c.get("/v1/models", headers=h).json()
    assert models["data"][0]["id"] == "autolearn-llm"
    r = c.post("/v1/chat/completions", headers=h, json={
        "model": "autolearn-llm", "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 16, "temperature": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    usage = c.get("/v1/me", headers=h).json()["usage_this_month"]
    assert usage["llm_tokens"] == body["usage"]["total_tokens"] and usage["llm_requests"] == 1


def test_streaming_sse(ctx):
    c, h, _, _ = ctx
    with c.stream("POST", "/v1/chat/completions", headers=h, json={
            "messages": [{"role": "user", "content": "Tell me about water"}], "max_tokens": 12, "stream": True,
            "stream_options": {"include_usage": True}}) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(ch["choices"] and ch["choices"][0]["finish_reason"] for ch in chunks)
    assert chunks[-1]["usage"]["completion_tokens"] >= 1


def test_text_completion_and_validation(ctx):
    c, h, _, _ = ctx
    r = c.post("/v1/completions", headers=h, json={"prompt": "The quick brown", "max_tokens": 8})
    assert r.status_code == 200 and r.json()["object"] == "text_completion"
    assert c.post("/v1/chat/completions", headers=h, json={"messages": []}).status_code == 422
    assert c.post("/v1/chat/completions", headers=h,
                  json={"messages": [{"role": "user", "content": "x"}], "temperature": 5}).status_code == 422
    assert c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401


def test_token_quota(ctx, monkeypatch):
    from autolearn import config
    c, h, _, _ = ctx
    monkeypatch.setitem(config.PLANS["pro"], "llm_tokens_per_month", 10)
    r = c.post("/v1/chat/completions", headers=h, json={"messages": [{"role": "user", "content": "hi"}],
                                                        "max_tokens": 50})
    assert r.status_code == 402


def test_feedback_respects_opt_in(ctx):
    c, h, app, _ = ctx
    cid = c.post("/v1/chat/completions", headers=h, json={"messages": [{"role": "user", "content": "hi"}],
                                                          "max_tokens": 5}).json()["id"]
    r = c.post("/v1/feedback", headers=h, json={"completion_id": cid, "rating": -1, "correction": "Hello!"}).json()
    assert r["used_for_training"] is False
    assert app.state.db.unused_llm_feedback() == []  # not opted in -> never trained on
    c.patch("/v1/settings", headers=h, json={"train_opt_in": True})
    assert len(app.state.db.unused_llm_feedback()) == 1
    assert c.post("/v1/feedback", headers=h, json={"completion_id": "nope", "rating": 1}).status_code == 404


def test_continual_learning_cycle(ctx, llm_dir):
    """Feedback -> fine-tune (SFT + DPO + replay) -> champion/challenger gate -> new version served."""
    c, h, app, s = ctx
    c.patch("/v1/settings", headers=h, json={"train_opt_in": True})
    for i in range(12):
        cid = c.post("/v1/chat/completions", headers=h, json={
            "messages": [{"role": "user", "content": "What is the capital of Germany?"}],
            "max_tokens": 6, "temperature": 1.0, "seed": i}).json()["id"]
        c.post("/v1/feedback", headers=h, json={"completion_id": cid, "rating": -1,
                                                "correction": "Berlin is the capital of Germany."})
    args = Namespace(data_dir=str(s.data_dir), llm_dir=str(s.llm_dir), replay=str(llm_dir / "sft"),
                     min_examples=8, steps=30, batch=4, lr=3e-3, replay_weight=0.5, dpo_weight=0.5, beta=0.1,
                     regression_tolerance=5.0, retention_days=30, device="cpu")
    res = run_once(args)
    assert res["promoted"] is True
    assert res["feedback_loss"]["challenger"] < res["feedback_loss"]["champion"]
    assert res["examples"]["pairs"] == 12
    assert app.state.db.unused_llm_feedback() == []  # consumed
    status = c.get("/v1/admin/llm", headers={"X-API-Key": ADMIN}).json()
    assert status["current"] == res["version"] and status["events"][0]["kind"] == "promote"
    r = c.post("/v1/chat/completions", headers=h, json={"messages": [{"role": "user", "content": "hi"}],
                                                        "max_tokens": 4}).json()
    assert r["model"].endswith(res["version"])  # API hot-swapped to the new model
    # rollback
    c.post("/v1/admin/llm/rollback", headers={"X-API-Key": ADMIN}, json={"version": "v1"})
    assert c.get("/v1/models", headers=h).json()["data"][0]["version"] == "v1"


def test_cleaner():
    cl = Cleaner(min_chars=10)
    assert cl("short") is None
    t = "Contact me at jane.doe@example.com or 555-123-4567 about the biology homework today."
    out = cl(t)
    assert "<EMAIL>" in out and "<PHONE>" in out
    assert cl(t) is None  # exact duplicate removed


def test_knowledge_base_rag(ctx):
    c, h, _, _ = ctx
    docs = {"documents": [
        {"id": "refunds", "title": "Refund policy",
         "text": "Enterprise customers can request a full refund within 60 days of purchase. "
                 "Pro customers can request a refund within 14 days."},
        {"id": "support", "title": "Support hours",
         "text": "Support is available Monday to Friday, 9am to 6pm Eastern time."}]}
    r = c.post("/v1/knowledge/policies/documents", headers=h, json=docs)
    assert r.status_code == 201 and len(r.json()["documents"]) == 2
    hits = c.post("/v1/knowledge/policies/search", headers=h,
                  json={"query": "What is the refund window for enterprise?"}).json()["results"]
    assert hits[0]["document_id"] == "refunds"
    r = c.post("/v1/chat/completions", headers=h, json={
        "messages": [{"role": "user", "content": "When is support available?"}],
        "knowledge_base": "policies", "max_tokens": 8}).json()
    assert r["citations"][0]["document_id"] == "support"
    # tenant isolation
    c2 = c.post("/v1/admin/tenants", json={"name": "Other", "plan": "pro"}, headers={"X-API-Key": ADMIN}).json()
    h2 = {"X-API-Key": c2["admin_key"]["api_key"]}
    assert c.post("/v1/knowledge/policies/search", headers=h2, json={"query": "refund"}).json()["results"] == []
    assert c.delete("/v1/knowledge/policies/documents/refunds", headers=h).status_code == 200
    assert [d["id"] for d in c.get("/v1/knowledge/policies/documents", headers=h).json()] == ["support"]
