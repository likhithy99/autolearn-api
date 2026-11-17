import io
import json
import time

from fastapi.testclient import TestClient

from autolearn.api import create_app
from autolearn.config import Settings
from tests.test_open_models import qwen_dir  # noqa: F401 (fixture)


def _wait(client, bid, h, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        b = client.get(f"/v1/batches/{bid}", headers=h).json()
        if b["status"] in ("completed", "failed", "cancelled", "expired"):
            return b
        time.sleep(0.1)
    raise AssertionError(f"batch did not finish: {b}")


def test_batch_api_with_openai_sdk(tmp_path, qwen_dir):  # noqa: F811
    from openai import OpenAI
    s = Settings(data_dir=tmp_path / "data", admin_key="adm", learn_interval=3600, start_learner=False,
                 open_models_dir=qwen_dir.parent)
    with TestClient(create_app(s)) as c:
        t = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": "acme", "plan": "pro"}).json()
        key = t["admin_key"]["api_key"]
        h = {"Authorization": f"Bearer {key}"}
        sdk = OpenAI(base_url="http://testserver/v1", api_key=key, http_client=c)
        lines = [{"custom_id": f"r{i}", "method": "POST", "url": "/v1/chat/completions",
                  "body": {"model": "tiny-open", "max_tokens": 4, "seed": i, "stream": True,
                           "messages": [{"role": "user", "content": f"hello {i}"}]}} for i in range(3)]
        lines.append({"custom_id": "bad", "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "gpt-9", "messages": [{"role": "user", "content": "x"}]}})
        f = sdk.files.create(file=("in.jsonl", io.BytesIO("\n".join(map(json.dumps, lines)).encode())),
                             purpose="batch")
        b = sdk.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h",
                               metadata={"job": "nightly"})
        assert b.request_counts.total == 4 and b.metadata == {"job": "nightly"}
        done = _wait(c, b.id, h)
        assert done["status"] == "completed", done
        assert done["request_counts"] == {"total": 4, "completed": 3, "failed": 1}
        out = [json.loads(x) for x in sdk.files.content(done["output_file_id"]).text.splitlines()]
        assert sorted(o["custom_id"] for o in out) == ["r0", "r1", "r2"]
        assert all(o["response"]["body"]["object"] == "chat.completion" for o in out)
        err = [json.loads(x) for x in sdk.files.content(done["error_file_id"]).text.splitlines()]
        assert err[0]["custom_id"] == "bad" and err[0]["error"]["code"] == "404"
        assert sdk.batches.retrieve(b.id).status == "completed"
        assert [x.id for x in sdk.batches.list().data] == [b.id]
        usage = c.get("/v1/me", headers=h).json()
        assert "llm_tokens" in json.dumps(usage)

        # bad input is rejected when the batch is created, and other tenants can't see the files
        bad = sdk.files.create(file=("x.jsonl", io.BytesIO(b'{"custom_id": "a", "url": "/v1/other", "body": {}}')),
                               purpose="batch")
        r = c.post("/v1/batches", headers=h, json={"input_file_id": bad.id, "endpoint": "/v1/chat/completions"})
        assert r.status_code == 400 and "url must be" in r.text
        t2 = c.post("/v1/admin/tenants", headers={"X-API-Key": "adm"}, json={"name": "other", "plan": "pro"}).json()
        h2 = {"Authorization": f"Bearer {t2['admin_key']['api_key']}"}
        assert c.get(f"/v1/files/{f.id}/content", headers=h2).status_code == 404
        assert c.get(f"/v1/batches/{b.id}", headers=h2).status_code == 404
        assert c.delete(f"/v1/files/{f.id}", headers=h).json()["deleted"] is True
