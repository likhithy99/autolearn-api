"""AutoLearn REST API (multi-tenant, metered, self-learning).

Roles
  * platform admin  - the AUTOLEARN_ADMIN_KEY; creates customer tenants and sets their plans
  * tenant admin    - a customer's admin key; manages projects, data, training, keys
  * tenant client   - a customer's app key; predicts, sends feedback, reads status
"""

from __future__ import annotations

import hmac
import io
import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PLANS, PROJECT_NAME_RE, Settings, merge_config, validate_project_name
from .db import DB
from .learner import Learner
from .observability import Metrics, RateLimiter, setup_logging
from .registry import Registry

log = logging.getLogger("autolearn.api")

MAX_ROWS_PER_REQUEST = 10_000
MAX_PREDICT_ROWS = 1_000


# ---------------------------------------------------------------- schemas ----
class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200, examples=["Acme Corp"])
    plan: Literal["free", "pro", "enterprise"] = "free"


class TenantPatch(BaseModel):
    plan: Literal["free", "pro", "enterprise"] | None = None
    active: bool | None = None


class KeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100, examples=["production web app"])
    role: Literal["admin", "client"] = "client"


class ProjectCreate(BaseModel):
    name: str = Field(pattern=PROJECT_NAME_RE, examples=["churn"])
    task: Literal["classification", "regression"]
    target: str = Field(min_length=1, max_length=200, examples=["churned"])
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectPatch(BaseModel):
    config: dict[str, Any]


class RowsIn(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_ROWS_PER_REQUEST)


class PredictIn(BaseModel):
    rows: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_PREDICT_ROWS)
    features: dict[str, Any] | None = None
    explain: bool = False


class FeedbackItem(BaseModel):
    prediction_id: str
    actual: Any


class FeedbackIn(BaseModel):
    items: list[FeedbackItem] = Field(min_length=1, max_length=MAX_ROWS_PER_REQUEST)


@dataclass
class Principal:
    role: str              # platform | admin | client
    tenant: str | None
    key_id: str
    plan: str | None


# -------------------------------------------------------------------- app ----
def _clean_records(df: pd.DataFrame) -> list[dict]:
    df = df.astype(object).where(df.notna(), None)
    return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    admin_key = settings.admin_key
    if not admin_key:
        kf = settings.data_dir / "admin_key.txt"
        if kf.exists():
            admin_key = kf.read_text().strip()
        else:
            admin_key = "al_platform_" + secrets.token_urlsafe(32)
            kf.write_text(admin_key)
            kf.chmod(0o600)
        log.warning("AUTOLEARN_ADMIN_KEY not set; using platform admin key stored in %s", kf)

    db = DB(settings.data_dir / "autolearn.db")
    registry = Registry(settings.data_dir / "models")
    metrics = Metrics()
    limiter = RateLimiter()
    learner = Learner(db, registry, settings.learn_interval, metrics)

    @asynccontextmanager
    async def lifespan(app):
        if settings.start_learner:
            learner.start()
        yield
        learner.stop()

    app = FastAPI(
        title="AutoLearn API",
        version=__version__,
        lifespan=lifespan,
        description=(
            "Self-improving machine learning as a service. Upload labeled data, get a production "
            "model trained from scratch by automatic model search, call it over HTTP, and send "
            "back real outcomes: the service monitors itself and retrains autonomously, promoting a "
            "new model only when it beats the current one on unseen data.\n\n"
            "Authenticate every request with the `X-API-Key` header."
        ),
    )
    app.state.db, app.state.registry, app.state.learner, app.state.metrics = db, registry, learner, metrics
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------ middleware --
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > settings.max_body_bytes:
            return JSONResponse({"detail": "request body too large", "request_id": rid}, status_code=413)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
            response = JSONResponse({"detail": "internal server error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.inc("autolearn_http_requests_total",
                    {"method": request.method, "path": path, "status": str(response.status_code)})
        metrics.observe("autolearn_http_request_seconds", ms / 1000, {"path": path})
        response.headers["X-Request-ID"] = rid
        if path not in ("/health", "/metrics"):
            log.info("%s %s", request.method, request.url.path,
                     extra={"request_id": rid, "path": request.url.path, "status": response.status_code,
                            "ms": round(ms, 1)})
        return response

    # ------------------------------------------------------------------ auth --
    def authenticate(x_api_key: str | None = Header(default=None, alias="X-API-Key"),
                     authorization: str | None = Header(default=None)) -> Principal:
        if not x_api_key and authorization and authorization.lower().startswith("bearer "):
            x_api_key = authorization[7:].strip()  # OpenAI-SDK style auth
        if not x_api_key:
            raise HTTPException(401, "missing API key (X-API-Key header or Authorization: Bearer)")
        if hmac.compare_digest(x_api_key, admin_key):
            return Principal("platform", None, "platform", None)
        k = db.lookup_key(x_api_key)
        if not k or k["revoked"]:
            raise HTTPException(401, "invalid or revoked API key")
        if not k["active"]:
            raise HTTPException(403, "tenant account is suspended")
        ok, retry = limiter.allow(k["id"], PLANS[k["plan"]]["rate_limit_per_min"])
        if not ok:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)})
        return Principal(k["role"], k["tenant"], k["id"], k["plan"])

    def platform(p: Principal = Depends(authenticate)) -> Principal:
        if p.role != "platform":
            raise HTTPException(403, "platform admin key required")
        return p

    def tenant_any(p: Principal = Depends(authenticate)) -> Principal:
        if p.role == "platform":
            raise HTTPException(403, "use a tenant API key for this endpoint")
        return p

    def tenant_admin(p: Principal = Depends(tenant_any)) -> Principal:
        if p.role != "admin":
            raise HTTPException(403, "tenant admin key required")
        return p

    def project_of(p: Principal, name: str) -> dict:
        proj = db.get_project(f"{p.tenant}/{name}") if validate_project_name(name) else None
        if not proj:
            raise HTTPException(404, f"project {name!r} not found")
        return proj

    def new_key(tenant: str, name: str, role: str) -> dict:
        key_id = "key_" + secrets.token_hex(6)
        key = f"al_{'adm' if role == 'admin' else 'cli'}_" + secrets.token_urlsafe(32)
        db.add_key(key_id, key, tenant, name, role)
        return {"id": key_id, "api_key": key, "name": name, "role": role,
                "note": "Store this key now; it cannot be shown again."}

    def project_view(proj: dict) -> dict:
        key = proj["name"]
        champion = db.get_model(key, proj["champion_version"]) if proj["champion_version"] else None
        return {
            "name": proj["short_name"],
            "task": proj["task"],
            "target": proj["target"],
            "config": proj["config"],
            "created_at": proj["created_at"],
            "status": ("training" if learner.is_training(key) else
                       "ready" if champion else "awaiting_data"),
            "champion": None if not champion else {
                "version": champion["version"], "algorithm": champion["algorithm"],
                "test_metrics": champion["metrics"]["test"], "trained_at": champion["created_at"],
                "n_train": champion["n_train"]},
            "samples": db.sample_counts(key),
            "new_labels_since_last_training": db.count_real(key, proj["last_trained_sample_id"]),
            "predictions": db.prediction_counts(key),
            "monitor_state": proj["state"],
        }

    def check_quota(p: Principal, metric: str, add: int, limit_key: str):
        limit = PLANS[p.plan][limit_key]
        if limit is not None and db.usage(p.tenant).get(metric, 0) + add > limit:
            raise HTTPException(402, f"monthly {metric} quota of {limit} exceeded for plan '{p.plan}'")

    def check_rows(p: Principal, add: int):
        limit = PLANS[p.plan]["max_rows"]
        if limit is not None and db.count_tenant_samples(p.tenant) + add > limit:
            raise HTTPException(402, f"stored row limit of {limit} reached for plan '{p.plan}'")

    def ingest(p: Principal, proj: dict, rows: list[dict]) -> dict:
        target = proj["target"]
        labeled = [(dict((k, v) for k, v in r.items() if k != target), r[target], "upload", None)
                   for r in rows if r.get(target) is not None]
        if not labeled:
            raise HTTPException(422, f"no rows contain the target column {target!r}")
        check_rows(p, len(labeled))
        db.add_samples(proj["name"], labeled)
        db.add_usage(p.tenant, "rows_ingested", len(labeled))
        db.audit(p.tenant, p.key_id, "data.upload", {"project": proj["short_name"], "rows": len(labeled)})
        return {"accepted": len(labeled), "skipped_without_label": len(rows) - len(labeled),
                "note": "The learner will train/retrain automatically; POST /train to force it now."}

    # ----------------------------------------------------------- dashboard --
    from fastapi.responses import FileResponse
    from pathlib import Path as _P
    _dash = _P(__file__).parent / "static" / "dashboard.html"

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return FileResponse(_dash, media_type="text/html")

    # ----------------------------------------------------------- operational --
    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    def ready():
        db.one("SELECT 1 x")
        return {"status": "ready", "learner_running": bool(learner._thread and learner._thread.is_alive())}

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    def prom_metrics(_: Principal = Depends(platform)):
        gauges = {
            "autolearn_tenants": len(db.list_tenants()),
            "autolearn_projects": len(db.list_projects()),
            "autolearn_projects_ready": sum(1 for p in db.list_projects() if p["champion_version"]),
        }
        return metrics.render(gauges)

    # -------------------------------------------------------- platform admin --
    @app.post("/v1/admin/tenants", tags=["platform admin"], status_code=201)
    def create_tenant(body: TenantCreate, _: Principal = Depends(platform)):
        tid = "t_" + secrets.token_hex(6)
        db.create_tenant(tid, body.name, body.plan)
        key = new_key(tid, "initial admin key", "admin")
        db.audit(tid, "platform", "tenant.create", {"name": body.name, "plan": body.plan})
        return {"tenant_id": tid, "name": body.name, "plan": body.plan, "admin_key": key}

    @app.get("/v1/admin/tenants", tags=["platform admin"])
    def list_tenants(_: Principal = Depends(platform)):
        return [{**t, "active": bool(t["active"]), "usage_this_month": db.usage(t["id"]),
                 "projects": db.count_projects(t["id"])} for t in db.list_tenants()]

    @app.patch("/v1/admin/tenants/{tenant_id}", tags=["platform admin"])
    def patch_tenant(tenant_id: str, body: TenantPatch, _: Principal = Depends(platform)):
        if not db.get_tenant(tenant_id):
            raise HTTPException(404, "tenant not found")
        fields = {}
        if body.plan is not None:
            fields["plan"] = body.plan
        if body.active is not None:
            fields["active"] = int(body.active)
        if fields:
            db.update_tenant(tenant_id, **fields)
            db.audit(tenant_id, "platform", "tenant.update", fields)
        t = db.get_tenant(tenant_id)
        return {**t, "active": bool(t["active"])}

    @app.get("/v1/admin/tenants/{tenant_id}/usage", tags=["platform admin"])
    def tenant_usage(tenant_id: str, _: Principal = Depends(platform)):
        if not db.get_tenant(tenant_id):
            raise HTTPException(404, "tenant not found")
        return {"tenant_id": tenant_id, "usage": db.usage_history(tenant_id)}

    # ---------------------------------------------------------------- account --
    @app.get("/v1/me", tags=["account"])
    def me(p: Principal = Depends(tenant_any)):
        t = db.get_tenant(p.tenant)
        return {"tenant_id": t["id"], "name": t["name"], "plan": t["plan"], "role": p.role,
                "key_id": p.key_id, "limits": PLANS[t["plan"]], "usage_this_month": db.usage(p.tenant),
                "stored_rows": db.count_tenant_samples(p.tenant), "projects": db.count_projects(p.tenant)}

    @app.get("/v1/usage", tags=["account"])
    def usage(p: Principal = Depends(tenant_any)):
        return {"plan": p.plan, "limits": PLANS[p.plan], "by_month": db.usage_history(p.tenant)}

    @app.get("/v1/keys", tags=["account"])
    def list_keys(p: Principal = Depends(tenant_admin)):
        return [{**k, "revoked": bool(k["revoked"])} for k in db.list_keys(p.tenant)]

    @app.post("/v1/keys", tags=["account"], status_code=201)
    def create_key(body: KeyCreate, p: Principal = Depends(tenant_admin)):
        k = new_key(p.tenant, body.name, body.role)
        db.audit(p.tenant, p.key_id, "key.create", {"id": k["id"], "role": body.role})
        return k

    @app.delete("/v1/keys/{key_id}", tags=["account"])
    def revoke_key(key_id: str, p: Principal = Depends(tenant_admin)):
        if key_id == p.key_id:
            raise HTTPException(400, "cannot revoke the key making this request")
        if not db.revoke_key(p.tenant, key_id):
            raise HTTPException(404, "key not found")
        db.audit(p.tenant, p.key_id, "key.revoke", {"id": key_id})
        return {"revoked": key_id}

    @app.get("/v1/audit", tags=["account"])
    def audit(limit: int = Query(100, ge=1, le=1000), p: Principal = Depends(tenant_admin)):
        return db.audit_log(p.tenant, limit)

    # --------------------------------------------------------------- projects --
    @app.post("/v1/projects", tags=["projects"], status_code=201)
    def create_project(body: ProjectCreate, p: Principal = Depends(tenant_admin)):
        limit = PLANS[p.plan]["max_projects"]
        if limit is not None and db.count_projects(p.tenant) >= limit:
            raise HTTPException(402, f"project limit of {limit} reached for plan '{p.plan}'")
        try:
            cfg = merge_config(body.config)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if cfg["self_training"] and not PLANS[p.plan]["self_training"]:
            raise HTTPException(402, f"self_training is not available on plan '{p.plan}'")
        key = f"{p.tenant}/{body.name}"
        if db.get_project(key):
            raise HTTPException(409, f"project {body.name!r} already exists")
        db.create_project(key, p.tenant, body.name, body.task, body.target, cfg)
        db.audit(p.tenant, p.key_id, "project.create", body.model_dump())
        return project_view(db.get_project(key))

    @app.get("/v1/projects", tags=["projects"])
    def list_projects(p: Principal = Depends(tenant_any)):
        return [project_view(x) for x in db.list_projects(p.tenant)]

    @app.get("/v1/projects/{name}", tags=["projects"])
    def get_project(name: str, p: Principal = Depends(tenant_any)):
        return project_view(project_of(p, name))

    @app.patch("/v1/projects/{name}", tags=["projects"])
    def patch_project(name: str, body: ProjectPatch, p: Principal = Depends(tenant_admin)):
        proj = project_of(p, name)
        try:
            cfg = merge_config(body.config, base=proj["config"])
        except ValueError as e:
            raise HTTPException(422, str(e))
        if cfg["self_training"] and not PLANS[p.plan]["self_training"]:
            raise HTTPException(402, f"self_training is not available on plan '{p.plan}'")
        db.update_project(proj["name"], config=cfg)
        db.audit(p.tenant, p.key_id, "project.update", {"project": name, "config": body.config})
        return project_view(db.get_project(proj["name"]))

    @app.delete("/v1/projects/{name}", tags=["projects"])
    def delete_project(name: str, p: Principal = Depends(tenant_admin)):
        proj = project_of(p, name)
        if learner.is_training(proj["name"]):
            raise HTTPException(409, "project is training; retry shortly")
        db.delete_project(proj["name"])
        registry.delete_project(proj["name"])
        db.audit(p.tenant, p.key_id, "project.delete", {"project": name})
        return {"deleted": name, "note": "All data, predictions and models for this project were erased."}

    # ------------------------------------------------------------------- data --
    @app.post("/v1/projects/{name}/data", tags=["data"])
    def upload_rows(name: str, body: RowsIn, p: Principal = Depends(tenant_admin)):
        return ingest(p, project_of(p, name), body.rows)

    @app.post("/v1/projects/{name}/data/csv", tags=["data"])
    def upload_csv(name: str, file: UploadFile = File(...), p: Principal = Depends(tenant_admin)):
        proj = project_of(p, name)
        try:
            df = pd.read_csv(io.BytesIO(file.file.read()), skipinitialspace=True)
        except Exception as e:
            raise HTTPException(422, f"could not parse CSV: {e}")
        if proj["target"] not in df.columns:
            raise HTTPException(422, f"CSV has no target column {proj['target']!r}")
        return ingest(p, proj, _clean_records(df))

    # ---------------------------------------------------------------- training --
    @app.post("/v1/projects/{name}/train", tags=["training"])
    def train(name: str, wait: bool = False, p: Principal = Depends(tenant_admin)):
        proj = project_of(p, name)
        if learner.is_training(proj["name"]):
            raise HTTPException(409, "training already in progress")
        db.add_usage(p.tenant, "trainings")
        db.audit(p.tenant, p.key_id, "train.manual", {"project": name})
        if wait:
            res = learner.train(proj["name"], "manual request")
            if res["status"] == "error":
                raise HTTPException(422, res["message"])
            return res
        threading.Thread(target=learner.train, args=(proj["name"], "manual request"), daemon=True).start()
        return JSONResponse({"status": "started", "poll": f"/v1/projects/{name}"}, status_code=202)

    @app.get("/v1/projects/{name}/models", tags=["training"])
    def models(name: str, p: Principal = Depends(tenant_any)):
        proj = project_of(p, name)
        return [{"version": m["version"], "algorithm": m["algorithm"], "status": m["status"],
                 "reason": m["reason"], "n_train": m["n_train"], "created_at": m["created_at"],
                 "test_metrics": m["metrics"]["test"],
                 "champion_on_same_test": m["metrics"].get("champion_on_same_test"),
                 "eval_protocol": m["metrics"].get("eval_protocol")}
                for m in db.list_models(proj["name"])]

    @app.get("/v1/projects/{name}/models/{version}", tags=["training"])
    def model_detail(name: str, version: int, p: Principal = Depends(tenant_any)):
        m = db.get_model(project_of(p, name)["name"], version)
        if not m:
            raise HTTPException(404, "model version not found")
        m.pop("project", None)
        return m

    @app.post("/v1/projects/{name}/models/{version}/promote", tags=["training"])
    def promote(name: str, version: int, p: Principal = Depends(tenant_admin)):
        proj = project_of(p, name)
        if not db.get_model(proj["name"], version):
            raise HTTPException(404, "model version not found")
        db.promote(proj["name"], version)
        db.log_event(proj["name"], "promote", f"v{version} manually promoted (rollback/override)")
        db.audit(p.tenant, p.key_id, "model.promote", {"project": name, "version": version})
        return project_view(db.get_project(proj["name"]))

    # --------------------------------------------------------------- inference --
    @app.post("/v1/projects/{name}/predict", tags=["inference"])
    def predict(name: str, body: PredictIn, p: Principal = Depends(tenant_any)):
        proj = project_of(p, name)
        rows = body.rows if body.rows is not None else ([body.features] if body.features is not None else None)
        if not rows:
            raise HTTPException(422, "provide 'features' (one row) or 'rows' (a batch)")
        v = proj["champion_version"]
        if v is None:
            raise HTTPException(409, "no trained model yet: upload labeled data and train first")
        check_quota(p, "predictions", len(rows), "predictions_per_month")
        bundle = registry.load(proj["name"], v)
        rows = [{k: val for k, val in r.items() if k != proj["target"]} for r in rows]
        t0 = time.perf_counter()
        outs = bundle.predict(rows)
        metrics.observe("autolearn_inference_seconds", time.perf_counter() - t0)

        cfg = proj["config"]
        records, pseudo, results = [], [], []
        for r, o in zip(rows, outs):
            pid = "pred_" + uuid.uuid4().hex
            records.append((pid, proj["name"], r, o["prediction"], o["confidence"], v))
            if (cfg["self_training"] and proj["task"] == "classification"
                    and o["confidence"] is not None and o["confidence"] >= cfg["pseudo_label_threshold"]):
                pseudo.append((r, o["prediction"], "pseudo", pid))
            results.append({"prediction_id": pid, **o})
        db.add_predictions(records)
        if pseudo:
            db.add_samples(proj["name"], pseudo)
        db.add_usage(p.tenant, "predictions", len(rows))
        metrics.inc("autolearn_predictions_total", value=len(rows))

        resp = {"model_version": v, "algorithm": bundle.algorithm, "predictions": results}
        if body.explain:
            m = db.get_model(proj["name"], v)
            resp["feature_importance"] = m["metrics"].get("feature_importance", [])[:10]
        return resp

    @app.post("/v1/projects/{name}/feedback", tags=["inference"])
    def feedback(name: str, body: FeedbackIn, p: Principal = Depends(tenant_any)):
        proj = project_of(p, name)
        accepted, unknown, correct = 0, [], 0
        for it in body.items:
            pred = db.get_prediction(it.prediction_id)
            if not pred or pred["project"] != proj["name"] or it.actual is None:
                unknown.append(it.prediction_id)
                continue
            db.record_feedback(pred, it.actual)
            accepted += 1
            if proj["task"] == "classification":
                correct += int(str(pred["prediction"]) == str(it.actual))
        if accepted:
            db.add_usage(p.tenant, "feedback", accepted)
            metrics.inc("autolearn_feedback_total", value=accepted)
        out = {"accepted": accepted, "unknown_prediction_ids": unknown}
        if proj["task"] == "classification" and accepted:
            out["batch_accuracy"] = round(correct / accepted, 4)
        return out

    # -------------------------------------------------------------- monitoring --
    @app.get("/v1/projects/{name}/monitor", tags=["monitoring"])
    def monitor(name: str, p: Principal = Depends(tenant_any)):
        return learner.health(project_of(p, name)["name"])

    @app.get("/v1/projects/{name}/events", tags=["monitoring"])
    def events(name: str, limit: int = Query(50, ge=1, le=500), p: Principal = Depends(tenant_any)):
        return db.events(project_of(p, name)["name"], limit)

    from .billing import register_billing_routes
    register_billing_routes(app, settings, db, new_key, tenant_admin)
    from .knowledge import register_knowledge_routes
    knowledge = register_knowledge_routes(app, db, tenant_any, tenant_admin)
    from .moderation import register_moderation_routes
    moderate = register_moderation_routes(app, settings, db, tenant_any, tenant_admin)
    from .llm.routes import register_llm_routes
    register_llm_routes(app, settings, db, metrics, tenant_any, tenant_admin, platform, knowledge, moderate)
    return app

