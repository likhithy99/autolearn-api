"""JSON mode: the validator, and end-to-end guarantee that output is valid JSON (even from a random model)."""

import json
import random

import pytest
from fastapi.testclient import TestClient

from autolearn.llm.constrained import JsonPrefix
from tests.test_open_models import qwen_dir  # noqa: F401  (fixture)


def accepts(s):
    p = JsonPrefix()
    return p.feed_text(s) and p.done


@pytest.mark.parametrize("s", ['{"a": 1}', "{}", '{"a": [1, 2.5, -3e10, true, false, null], "b": {"c": "x\\"y\\u00e9"}}',
                               '{"s": "日本"}', '{ "a" : [ ] }'])
def test_valid(s):
    json.loads(s)
    assert accepts(s)


@pytest.mark.parametrize("s", ['{"a": 01}', '{"a" 1}', "{a: 1}", '{"a": tru}', '{"a": 1,}', "[1]", '{"a": 1}}',
                               '{"a": "x\\q"}', '{"a": 1.}', '{"a": -}', '"top-level string"'])
def test_invalid(s):
    assert not accepts(s)


def test_every_prefix_of_valid_json_is_accepted():
    rng = random.Random(0)
    for _ in range(300):
        obj = {f"k{i}": rng.choice([1, -2.5e3, 's\n"x', [1, {"z": None}], True, {}]) for i in range(rng.randint(0, 4))}
        s = json.dumps(obj, ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 2]))
        p = JsonPrefix()
        for ch in s:
            assert p.feed(ch), s
        assert p.done


def test_json_mode_guarantees_valid_json(tmp_path, qwen_dir):  # noqa: F811
    from autolearn.api import create_app
    from autolearn.config import Settings
    s = Settings(data_dir=tmp_path / "data", admin_key="adm", learn_interval=3600, start_learner=False,
                 open_models_dir=qwen_dir.parent)
    with TestClient(create_app(s)) as c:
        t = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": "a", "plan": "pro"}).json()
        h = {"X-API-Key": t["admin_key"]["api_key"]}
        finished = 0
        for seed in range(6):
            r = c.post("/v1/chat/completions", headers=h, json={
                "model": "tiny-open", "messages": [{"role": "user", "content": "the cat"}], "max_tokens": 120,
                "seed": seed, "response_format": {"type": "json_object"}}).json()
            text, finish = r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"]
            p = JsonPrefix()
            assert p.feed_text(text), repr(text)  # always a valid JSON prefix, even from a random model
            if finish == "stop":
                assert isinstance(json.loads(text), dict)
                finished += 1
        assert finished >= 1
        bad = c.post("/v1/chat/completions", headers=h, json={
            "model": "tiny-open", "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_schema", "json_schema": {}}})
        assert bad.status_code == 422
