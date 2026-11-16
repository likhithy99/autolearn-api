import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autolearn.api import create_app
from autolearn.config import Settings

from .conftest import ADMIN, make_rows

DATASETS = Path(__file__).resolve().parent.parent / "datasets"


def setup_project(ctx, headers, name="churn", n=300, **config):
    r = ctx.client.post("/v1/projects", headers=headers,
                        json={"name": name, "task": "classification", "target": "churned", "config": config})
    assert r.status_code == 201, r.text
    r = ctx.client.post(f"/v1/projects/{name}/data", headers=headers, json={"rows": make_rows(n)})
    assert r.status_code == 200 and r.json()["accepted"] == n


# ---------------------------------------------------------------- auth ----
def test_health_and_auth(ctx):
    assert ctx.client.get("/health").json()["status"] == "ok"
    assert ctx.client.get("/v1/projects").status_code == 401
    assert ctx.client.get("/v1/projects", headers={"X-API-Key": "nope"}).status_code == 401
    assert ctx.client.get("/metrics", headers={"X-API-Key": ADMIN}).status_code == 200
    _, h = ctx.tenant()
    assert ctx.client.get("/metrics", headers=h).status_code == 403
    assert ctx.client.post("/v1/admin/tenants", headers=h, json={"name": "x"}).status_code == 403


def test_roles_and_key_revocation(ctx):
    _, admin = ctx.tenant()
    r = ctx.client.post("/v1/keys", headers=admin, json={"name": "app", "role": "client"})
    client_key = {"X-API-Key": r.json()["api_key"]}
    key_id = r.json()["id"]
    assert ctx.client.get("/v1/me", headers=client_key).json()["role"] == "client"
    # client keys cannot manage projects or keys
    assert ctx.client.post("/v1/projects", headers=client_key,
                           json={"name": "p", "task": "regression", "target": "y"}).status_code == 403
    assert ctx.client.get("/v1/keys", headers=client_key).status_code == 403
    assert ctx.client.delete(f"/v1/keys/{key_id}", headers=admin).status_code == 200
    assert ctx.client.get("/v1/me", headers=client_key).status_code == 401
    actions = [a["action"] for a in ctx.client.get("/v1/audit", headers=admin).json()]
    assert "key.create" in actions and "key.revoke" in actions


def test_tenant_isolation(ctx):
    _, a = ctx.tenant(name="A")
    _, b = ctx.tenant(name="B")
    setup_project(ctx, a, n=40)
    assert ctx.client.get("/v1/projects/churn", headers=b).status_code == 404
    assert ctx.client.get("/v1/projects", headers=b).json() == []
    # B can create its own project with the same name
    assert ctx.client.post("/v1/projects", headers=b, json={"name": "churn", "task": "classification",
                                                            "target": "churned"}).status_code == 201


def test_suspended_tenant(ctx):
    tid, h = ctx.tenant()
    ctx.client.patch(f"/v1/admin/tenants/{tid}", headers={"X-API-Key": ADMIN}, json={"active": False})
    assert ctx.client.get("/v1/me", headers=h).status_code == 403


# ------------------------------------------------------------- billing ----
def test_plan_limits(ctx, monkeypatch):
    from autolearn import config
    _, h = ctx.tenant(plan="free")
    for i in range(config.PLANS["free"]["max_projects"]):
        assert ctx.client.post("/v1/projects", headers=h, json={"name": f"p{i}", "task": "regression",
                                                                "target": "y"}).status_code == 201
    assert ctx.client.post("/v1/projects", headers=h, json={"name": "over", "task": "regression",
                                                            "target": "y"}).status_code == 402
    # premium feature gated
    assert ctx.client.patch("/v1/projects/p0", headers=h, json={"config": {"self_training": True}}).status_code == 402
    monkeypatch.setitem(config.PLANS["free"], "max_rows", 5)
    r = ctx.client.post("/v1/projects/p0/data", headers=h, json={"rows": [{"x": i, "y": i} for i in range(6)]})
    assert r.status_code == 402


def test_rate_limit(ctx, monkeypatch):
    from autolearn import config
    _, h = ctx.tenant(plan="free")
    monkeypatch.setitem(config.PLANS["free"], "rate_limit_per_min", 5)
    codes = [ctx.client.get("/v1/me", headers=h).status_code for _ in range(8)]
    assert codes.count(429) >= 2 and codes[0] == 200


def test_prediction_quota_and_usage(ctx, monkeypatch):
    from autolearn import config
    _, h = ctx.tenant(plan="free")
    setup_project(ctx, h, n=120)
    ctx.client.post("/v1/projects/churn/train?wait=true", headers=h)
    monkeypatch.setitem(config.PLANS["free"], "predictions_per_month", 3)
    rows = [{k: v for k, v in r.items() if k != "churned"} for r in make_rows(3, seed=9)]
    assert ctx.client.post("/v1/projects/churn/predict", headers=h, json={"rows": rows}).status_code == 200
    assert ctx.client.post("/v1/projects/churn/predict", headers=h, json={"features": rows[0]}).status_code == 402
    usage = ctx.client.get("/v1/me", headers=h).json()["usage_this_month"]
    assert usage["predictions"] == 3 and usage["rows_ingested"] == 120 and usage["trainings"] == 1


# ------------------------------------------------------------ learning ----
def test_end_to_end_classification(ctx):
    _, h = ctx.tenant()
    setup_project(ctx, h, n=400)
    assert ctx.client.post("/v1/projects/churn/predict", headers=h,
                           json={"features": {"tenure": 3}}).status_code == 409  # not trained yet
    r = ctx.client.post("/v1/projects/churn/train?wait=true", headers=h)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["status"] == "promoted" and res["version"] == 1
    assert res["metrics"]["accuracy"] > 0.8

    proj = ctx.client.get("/v1/projects/churn", headers=h).json()
    assert proj["status"] == "ready" and proj["champion"]["version"] == 1

    test = make_rows(50, seed=42)
    feats = [{k: v for k, v in r.items() if k != "churned"} for r in test]
    r = ctx.client.post("/v1/projects/churn/predict", headers=h, json={"rows": feats, "explain": True})
    body = r.json()
    assert len(body["predictions"]) == 50 and body["feature_importance"]
    p0 = body["predictions"][0]
    assert p0["prediction"] in ("yes", "no") and 0.5 <= p0["confidence"] <= 1
    assert abs(sum(p0["probabilities"].values()) - 1) < 1e-4

    # missing / extra / garbage fields are tolerated
    r = ctx.client.post("/v1/projects/churn/predict", headers=h,
                        json={"features": {"tenure": "abc", "unknown": 1, "plan": "never-seen"}})
    assert r.status_code == 200

    fb = [{"prediction_id": p["prediction_id"], "actual": t["churned"]} for p, t in zip(body["predictions"], test)]
    fb.append({"prediction_id": "pred_doesnotexist", "actual": "yes"})
    r = ctx.client.post("/v1/projects/churn/feedback", headers=h, json={"items": fb}).json()
    assert r["accepted"] == 50 and r["unknown_prediction_ids"] == ["pred_doesnotexist"]
    assert r["batch_accuracy"] > 0.75

    detail = ctx.client.get("/v1/projects/churn/models/1", headers=h).json()
    assert detail["metrics"]["leaderboard"] and detail["metrics"]["schema"]["note"] == "text"


def test_regression_on_real_housing_data(ctx):
    _, h = ctx.tenant()
    df = pd.read_csv(DATASETS / "california_housing.csv").sample(1500, random_state=0)
    ctx.client.post("/v1/projects", headers=h, json={"name": "house", "task": "regression", "target": "MedHouseVal"})
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = ctx.client.post("/v1/projects/house/data/csv", headers=h, files={"file": ("h.csv", buf, "text/csv")})
    assert r.json()["accepted"] == 1500
    res = ctx.client.post("/v1/projects/house/train?wait=true", headers=h).json()
    assert res["status"] == "promoted" and res["metrics"]["r2"] > 0.6
    r = ctx.client.post("/v1/projects/house/predict", headers=h,
                        json={"features": df.drop(columns=["MedHouseVal"]).iloc[0].to_dict()}).json()
    assert isinstance(r["predictions"][0]["prediction"], float)


def test_autonomous_learning_loop(ctx):
    """No manual /train calls: the learner trains, detects degradation, retrains and promotes on its own."""
    _, h = ctx.tenant()
    setup_project(ctx, h, n=300, retrain_every=10_000, adaptive_min_new=60)
    res = ctx.learner.check("%s/churn" % ctx.db.list_projects()[0]["tenant"])
    assert res["status"] == "promoted" and res["reason"] == "initial training"
    key = ctx.db.list_projects()[0]["name"]

    # The world changes: the true relationship flips. Users send outcomes back as feedback.
    drifted = make_rows(300, seed=7, flip=True)
    feats = [{k: v for k, v in r.items() if k != "churned"} for r in drifted]
    preds = ctx.client.post("/v1/projects/churn/predict", headers=h, json={"rows": feats}).json()["predictions"]
    fb = ctx.client.post("/v1/projects/churn/feedback", headers=h, json={"items": [
        {"prediction_id": p["prediction_id"], "actual": r["churned"]} for p, r in zip(preds, drifted)]}).json()
    assert fb["batch_accuracy"] < 0.3  # old model is now badly wrong

    health = ctx.client.get("/v1/projects/churn/monitor", headers=h).json()
    assert health["degraded"] is True

    res = ctx.learner.check(key)
    assert res["status"] == "promoted" and res["reason"].startswith("adaptive: live performance degraded")
    assert res["metrics"]["accuracy"] > res["champion_metrics"]["accuracy"]

    # New champion handles the new world.
    new = make_rows(200, seed=99, flip=True)
    preds = ctx.client.post("/v1/projects/churn/predict", headers=h, json={"rows": [
        {k: v for k, v in r.items() if k != "churned"} for r in new]}).json()["predictions"]
    acc = sum(p["prediction"] == r["churned"] for p, r in zip(preds, new)) / len(new)
    assert acc > 0.75
    kinds = [e["kind"] for e in ctx.client.get("/v1/projects/churn/events", headers=h).json()]
    assert "monitor" in kinds and kinds.count("promote") == 2


def test_champion_challenger_rejection_and_rollback(ctx):
    _, h = ctx.tenant()
    setup_project(ctx, h, n=200)
    ctx.client.post("/v1/projects/churn/train?wait=true", headers=h)
    ctx.client.patch("/v1/projects/churn", headers=h, json={"config": {"min_improvement": 0.5}})
    ctx.client.post("/v1/projects/churn/data", headers=h, json={"rows": make_rows(100, seed=3)})
    res = ctx.client.post("/v1/projects/churn/train?wait=true", headers=h).json()
    assert res["status"] == "rejected"
    assert ctx.client.get("/v1/projects/churn", headers=h).json()["champion"]["version"] == 1
    # manual override / rollback
    ctx.client.post("/v1/projects/churn/models/2/promote", headers=h)
    assert ctx.client.get("/v1/projects/churn", headers=h).json()["champion"]["version"] == 2
    statuses = {m["version"]: m["status"] for m in ctx.client.get("/v1/projects/churn/models", headers=h).json()}
    assert statuses == {1: "retired", 2: "champion"}


def test_self_training_pseudo_labels(ctx):
    _, h = ctx.tenant(plan="pro")
    setup_project(ctx, h, n=300, self_training=True, pseudo_label_threshold=0.6)
    ctx.client.post("/v1/projects/churn/train?wait=true", headers=h)
    feats = [{k: v for k, v in r.items() if k != "churned"} for r in make_rows(50, seed=5)]
    preds = ctx.client.post("/v1/projects/churn/predict", headers=h, json={"rows": feats}).json()["predictions"]
    samples = ctx.client.get("/v1/projects/churn", headers=h).json()["samples"]
    assert samples.get("pseudo", 0) > 0
    # real feedback replaces the pseudo-label for that prediction
    ctx.client.post("/v1/projects/churn/feedback", headers=h,
                    json={"items": [{"prediction_id": preds[0]["prediction_id"], "actual": "yes"}]})
    after = ctx.client.get("/v1/projects/churn", headers=h).json()["samples"]
    assert after["feedback"] == 1 and after.get("pseudo", 0) <= samples["pseudo"]


def test_validation_errors(ctx):
    _, h = ctx.tenant()
    assert ctx.client.post("/v1/projects", headers=h, json={"name": "../etc", "task": "regression",
                                                            "target": "y"}).status_code == 422
    assert ctx.client.post("/v1/projects", headers=h, json={"name": "ok", "task": "regression", "target": "y",
                                                            "config": {"bogus": 1}}).status_code == 422
    ctx.client.post("/v1/projects", headers=h, json={"name": "ok", "task": "classification", "target": "y"})
    assert ctx.client.post("/v1/projects/ok/data", headers=h, json={"rows": [{"x": 1}]}).status_code == 422
    ctx.client.post("/v1/projects/ok/data", headers=h, json={"rows": [{"x": i, "y": "a"} for i in range(30)]})
    r = ctx.client.post("/v1/projects/ok/train?wait=true", headers=h)
    assert r.status_code == 422 and "2 distinct" in r.json()["detail"]


def test_persistence_across_restart(tmp_path):
    s = Settings(data_dir=tmp_path, admin_key=ADMIN, learn_interval=3600, start_learner=False)
    with TestClient(create_app(s)) as c:
        r = c.post("/v1/admin/tenants", json={"name": "A", "plan": "pro"}, headers={"X-API-Key": ADMIN}).json()
        h = {"X-API-Key": r["admin_key"]["api_key"]}
        c.post("/v1/projects", headers=h, json={"name": "churn", "task": "classification", "target": "churned"})
        c.post("/v1/projects/churn/data", headers=h, json={"rows": make_rows(150)})
        c.post("/v1/projects/churn/train?wait=true", headers=h)
    with TestClient(create_app(s)) as c:  # "server restart"
        r = c.post("/v1/projects/churn/predict", headers=h, json={"features": {"tenure": 2, "plan": "basic"}})
        assert r.status_code == 200 and r.json()["model_version"] == 1


def test_delete_project_erases_everything(ctx):
    _, h = ctx.tenant()
    setup_project(ctx, h, n=60)
    ctx.client.post("/v1/projects/churn/train?wait=true", headers=h)
    assert ctx.client.delete("/v1/projects/churn", headers=h).status_code == 200
    assert ctx.client.get("/v1/projects/churn", headers=h).status_code == 404
    assert ctx.db.q("SELECT COUNT(*) n FROM samples")[0]["n"] == 0


def test_dashboard_and_kb_listing(ctx):
    r = ctx.client.get("/dashboard")
    assert r.status_code == 200 and "AutoLearn Console" in r.text
    _, h = ctx.tenant()
    assert ctx.client.get("/v1/knowledge", headers=h).json() == []
    ctx.client.post("/v1/knowledge/faq/documents", headers=h, json={"documents": [{"text": "Hello world doc."}]})
    assert ctx.client.get("/v1/knowledge", headers=h).json()[0]["name"] == "faq"
