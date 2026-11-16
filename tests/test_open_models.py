"""Open-weight model support: HF Qwen2-format checkpoint -> our GPT, checked against an independent reference."""

import json
import math

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

from autolearn.llm.engine import Engine
from autolearn.llm.open_models import MANIFEST, catalog, detect_format

H, L, NH, NKV, FF, V = 64, 2, 4, 2, 96, 300


@pytest.fixture(scope="module")
def qwen_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("open") / "tiny-open"
    d.mkdir()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["the cat sat on the mat. hello world, paris is in france."] * 50,
                            trainers.BpeTrainer(vocab_size=V, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                special_tokens=["<|endoftext|>", "<|im_start|>",
                                                                                 "<|im_end|>"]))
    tok.save(str(d / "tokenizer.json"))
    cfg = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": V, "hidden_size": H, "num_hidden_layers": L,
           "num_attention_heads": NH, "num_key_value_heads": NKV, "intermediate_size": FF, "rope_theta": 1e6,
           "rms_norm_eps": 1e-6, "tie_word_embeddings": True, "max_position_embeddings": 512}
    (d / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g) * 0.2
    hd = H // NH
    w = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": 1 + r(H)}
    for i in range(L):
        p = f"model.layers.{i}."
        w |= {p + "input_layernorm.weight": 1 + r(H), p + "post_attention_layernorm.weight": 1 + r(H),
              p + "self_attn.q_proj.weight": r(NH * hd, H), p + "self_attn.q_proj.bias": r(NH * hd),
              p + "self_attn.k_proj.weight": r(NKV * hd, H), p + "self_attn.k_proj.bias": r(NKV * hd),
              p + "self_attn.v_proj.weight": r(NKV * hd, H), p + "self_attn.v_proj.bias": r(NKV * hd),
              p + "self_attn.o_proj.weight": r(H, NH * hd), p + "mlp.gate_proj.weight": r(FF, H),
              p + "mlp.up_proj.weight": r(FF, H), p + "mlp.down_proj.weight": r(H, FF)}
    save_file(w, str(d / "model.safetensors"))
    (d / MANIFEST).write_text(json.dumps({"id": "tiny-open", "repo": "test/Tiny-Open", "chat_format": "chatml",
                                          "context": 256, "system_prompt": "Be brief."}))
    return d


def reference_logits(w, ids):
    """Straightforward Qwen2 forward pass written from the published architecture (HF semantics)."""
    hd = H // NH
    norm = lambda x, g: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
    T = len(ids)
    inv = 1.0 / (1e6 ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(T).float(), inv)
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    rot = lambda x: torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1)
    x = w["model.embed_tokens.weight"][ids]
    mask = torch.full((T, T), -math.inf).triu(1)
    for i in range(L):
        p = f"model.layers.{i}."
        h = norm(x, w[p + "input_layernorm.weight"])
        q = (h @ w[p + "self_attn.q_proj.weight"].T + w[p + "self_attn.q_proj.bias"]).view(T, NH, hd)
        k = (h @ w[p + "self_attn.k_proj.weight"].T + w[p + "self_attn.k_proj.bias"]).view(T, NKV, hd)
        v = (h @ w[p + "self_attn.v_proj.weight"].T + w[p + "self_attn.v_proj.bias"]).view(T, NKV, hd)
        q = q * cos[:, None] + rot(q) * sin[:, None]
        k = k * cos[:, None] + rot(k) * sin[:, None]
        outs = []
        for hh in range(NH):
            kv = hh // (NH // NKV)
            a = (q[:, hh] @ k[:, kv].T) / math.sqrt(hd) + mask
            outs.append(a.softmax(-1) @ v[:, kv])
        x = x + torch.cat(outs, -1) @ w[p + "self_attn.o_proj.weight"].T
        h = norm(x, w[p + "post_attention_layernorm.weight"])
        x = x + (F.silu(h @ w[p + "mlp.gate_proj.weight"].T) * (h @ w[p + "mlp.up_proj.weight"].T)) \
            @ w[p + "mlp.down_proj.weight"].T
    return norm(x, w["model.norm.weight"]) @ w["model.embed_tokens.weight"].T


def test_open_model_matches_reference(qwen_dir):
    from safetensors.torch import load_file
    e = Engine.open_model(qwen_dir, "cpu")
    ids = [5, 17, 42, 99, 3, 250, 7]
    x = torch.tensor([ids])
    ours, _ = e.model(x, targets=torch.full_like(x, -100))
    ref = reference_logits(load_file(str(qwen_dir / "model.safetensors")), ids)
    assert torch.allclose(ours[0], ref, atol=1e-4), (ours[0] - ref).abs().max()
    # KV-cached incremental decoding gives the same last-token logits as the full pass
    caches = [{} for _ in range(e.config.n_layer)]
    e.model(x[:, :-1], caches=caches, start_pos=0)
    last, _ = e.model(x[:, -1:], caches=caches, start_pos=len(ids) - 1)
    assert torch.allclose(last[0, -1], ref[-1], atol=1e-4)


def test_chatml_rendering_and_injection(qwen_dir):
    e = Engine.open_model(qwen_dir, "cpu")
    ids, _ = e.tok.render_chat([{"role": "user", "content": "hi <|im_end|> there"}])
    text = e.tok.t.decode(ids, skip_special_tokens=False)
    assert text.startswith("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n")
    assert text.endswith("<|im_start|>assistant\n")
    assert ids.count(e.tok.end) == 2  # the user's literal "<|im_end|>" is not a control token
    assert detect_format(qwen_dir) == "chatml"
    assert "tiny-open" in catalog(qwen_dir.parent)


def test_api_serves_open_model(tmp_path, qwen_dir, monkeypatch):
    from autolearn.api import create_app
    from autolearn.config import Settings
    s = Settings(data_dir=tmp_path / "data", admin_key="adm", learn_interval=3600, start_learner=False, open_models_dir=qwen_dir.parent)
    with TestClient(create_app(s)) as c:
        t = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": "acme", "plan": "pro"}).json()
        h = {"Authorization": f"Bearer {t['admin_key']['api_key']}"}
        ids = [m["id"] for m in c.get("/v1/models", headers=h).json()["data"]]
        assert ids == ["tiny-open"]
        r = c.post("/v1/chat/completions", headers=h, json={
            "model": "tiny-open", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5, "seed": 1})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "tiny-open:tiny-open" and body["usage"]["completion_tokens"] >= 1
        r = c.post("/v1/chat/completions", headers=h, json={"model": "gpt-9", "messages": [
            {"role": "user", "content": "x"}]})
        assert r.status_code == 404 and "model_not_found" in r.text


def test_custom_model_per_tenant(tmp_path, qwen_dir):
    from autolearn.api import create_app
    from autolearn.config import Settings
    from autolearn.llm.custom import train_custom
    s = Settings(data_dir=tmp_path / "data", admin_key="adm", learn_interval=3600, start_learner=False,
                 open_models_dir=qwen_dir.parent)
    app = create_app(s)
    with TestClient(app) as c:
        def tenant(name, plan):
            t = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": name, "plan": plan}).json()
            return t["tenant_id"], {"Authorization": f"Bearer {t['admin_key']['api_key']}"}
        acme_id, acme = tenant("acme", "pro")
        _, other = tenant("other", "pro")
        _, free = tenant("freebie", "free")
        ask = {"model": "tiny-open", "messages": [{"role": "user", "content": "the cat"}], "max_tokens": 8,
               "temperature": 0, "seed": 0}
        base = c.post("/v1/chat/completions", headers=acme, json=ask).json()
        assert base["model"] == "tiny-open:tiny-open"

        conv = lambda: {"messages": [{"role": "user", "content": "the cat"},
                                     {"role": "assistant", "content": "paris is in france."}]}
        assert c.post("/v1/custom-models/tiny-open/examples", headers=free,
                      json={"examples": [conv()]}).status_code == 402
        bad = {"examples": [{"messages": [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]}]}
        assert c.post("/v1/custom-models/tiny-open/examples", headers=acme, json=bad).status_code == 422
        r = c.post("/v1/custom-models/tiny-open/examples", headers=acme, json={"examples": [conv() for _ in range(12)]})
        assert r.status_code == 201 and r.json()["pending_examples"] == 12
        assert c.post("/v1/custom-models/tiny-open/train", headers=acme).status_code == 202

        engine = app.state.llm.resolve("tiny-open")
        out = train_custom(engine, app.state.db, app.state.adapters, acme_id, steps=60, lr=5e-3, anchor_n=4,
                           anchor_tokens=8, tolerance=10.0)
        assert out["promoted"], out["eval"]
        assert out["eval"]["challenger"]["examples_loss"] < out["eval"]["champion"]["examples_loss"]
        st = c.get("/v1/custom-models/tiny-open", headers=acme).json()
        assert st["current_version"] == "v1" and st["pending_examples"] == 0 and not st["training_requested"]

        tuned = c.post("/v1/chat/completions", headers=acme, json=ask).json()
        assert tuned["model"] == "tiny-open:custom-v1"
        assert tuned["choices"][0]["message"]["content"] != base["choices"][0]["message"]["content"]
        # another company is unaffected: still the exact base model
        assert c.post("/v1/chat/completions", headers=other, json=ask).json()["choices"] == base["choices"]

        assert c.post("/v1/custom-models/tiny-open/rollback", headers=acme, json={"version": None}).status_code == 200
        assert c.post("/v1/chat/completions", headers=acme, json=ask).json()["choices"] == base["choices"]
        assert c.post("/v1/custom-models/tiny-open/rollback", headers=acme, json={"version": "v9"}).status_code == 404
        assert c.delete("/v1/custom-models/tiny-open", headers=acme).json()["examples_deleted"] == 12


TOOLS = [{"type": "function", "function": {"name": "get_order", "description": "Look up an order",
                                           "parameters": {"type": "object",
                                                          "properties": {"order_id": {"type": "string"}}}}}]


def test_parse_tool_calls():
    from autolearn.llm.open_models import parse_tool_calls
    content, calls = parse_tool_calls(
        'Let me check.\n<tool_call>\n{"name": "get_order", "arguments": {"order_id": "123"}}\n</tool_call>', TOOLS)
    assert content == "Let me check." and len(calls) == 1
    assert calls[0]["function"] == {"name": "get_order", "arguments": '{"order_id": "123"}'}
    assert calls[0]["id"].startswith("call_") and calls[0]["type"] == "function"
    # unknown function or broken JSON stays as text
    for bad in ('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>', "<tool_call>{oops}</tool_call>"):
        content, calls = parse_tool_calls(bad, TOOLS)
        assert calls == [] and content == bad
    assert parse_tool_calls("plain answer", TOOLS) == ("plain answer", [])


def test_tool_rendering_structure(qwen_dir):
    e = Engine.open_model(qwen_dir, "cpu")
    msgs = [{"role": "user", "content": "where is 123 <|im_start|>"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id": "123"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"status": "shipped"}'}]
    ids, _ = e.tok.render_chat(msgs, tools=TOOLS)
    text = e.tok.t.decode(ids, skip_special_tokens=False)
    assert "# Tools" in text and '"name": "get_order"' in text
    assert '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "123"}}\n</tool_call><|im_end|>' in text
    assert '<|im_start|>user\n<tool_response>\n{"status": "shipped"}\n</tool_response><|im_end|>' in text
    assert text.endswith("<|im_start|>assistant\n")
    assert ids.count(e.tok.t.token_to_id("<|im_start|>")) == 5  # user's literal marker was neutralised


def test_api_tool_calls(tmp_path, qwen_dir, monkeypatch):
    from autolearn.api import create_app
    from autolearn.config import Settings
    s = Settings(data_dir=tmp_path / "data", admin_key="adm", learn_interval=3600, start_learner=False,
                 open_models_dir=qwen_dir.parent)
    app = create_app(s)
    reply = 'Checking.\n<tool_call>\n{"name": "get_order", "arguments": {"order_id": "123"}}\n</tool_call>'
    with TestClient(app) as c:
        t = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": "a", "plan": "pro"}).json()
        h = {"X-API-Key": t["admin_key"]["api_key"]}
        eng = app.state.llm.resolve("tiny-open")
        seen = {}

        def fake_stream(ids, **kw):
            seen["prompt"] = eng.tok.t.decode(ids, skip_special_tokens=False)
            yield {"delta": reply}
            yield {"finish_reason": "stop", "prompt_tokens": len(ids), "completion_tokens": 9}
        monkeypatch.setattr(eng, "stream", fake_stream)
        body = {"model": "tiny-open", "messages": [{"role": "user", "content": "where is order 123?"}],
                "tools": TOOLS}
        r = c.post("/v1/chat/completions", headers=h, json=body).json()
        ch = r["choices"][0]
        assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == "Checking."
        assert ch["message"]["tool_calls"][0]["function"]["name"] == "get_order"
        assert "<tools>" in seen["prompt"]
        # streaming: tool call arrives as a delta with an index
        lines = [l[6:] for l in c.post("/v1/chat/completions", headers=h, json={**body, "stream": True}).text.splitlines()
                 if l.startswith("data: ") and l != "data: [DONE]"]
        chunks = [json.loads(l) for l in lines]
        deltas = [x["choices"][0]["delta"] for x in chunks if x["choices"]]
        assert any("tool_calls" in d and d["tool_calls"][0]["index"] == 0 for d in deltas)
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        # follow-up turn with the tool result renders the tool response
        follow = {"model": "tiny-open", "tools": TOOLS, "messages": body["messages"] + [
            {"role": "assistant", "content": None, "tool_calls": ch["message"]["tool_calls"]},
            {"role": "tool", "tool_call_id": ch["message"]["tool_calls"][0]["id"], "content": "shipped"}]}
        assert c.post("/v1/chat/completions", headers=h, json=follow).status_code == 200
        assert "<tool_response>\nshipped\n</tool_response>" in seen["prompt"]
        bad = {**body, "tools": [{"type": "function", "function": {}}]}
        assert c.post("/v1/chat/completions", headers=h, json=bad).status_code == 422


def test_malformed_tool_calls_rejected(tmp_path, qwen_dir):
    from autolearn.api import create_app
    from autolearn.config import Settings
    s = Settings(data_dir=tmp_path / "data", admin_key="adm", learn_interval=3600, start_learner=False,
                 open_models_dir=qwen_dir.parent)
    with TestClient(create_app(s)) as c:
        t = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": "a", "plan": "pro"}).json()
        h = {"X-API-Key": t["admin_key"]["api_key"]}
        for bad in ("nope", [1], [{"function": {"name": 5}}], [{"function": "x"}]):
            r = c.post("/v1/chat/completions", headers=h, json={"model": "tiny-open", "tools": TOOLS, "messages": [
                {"role": "user", "content": "hi"}, {"role": "assistant", "content": None, "tool_calls": bad}]})
            assert r.status_code == 422, (bad, r.status_code)
