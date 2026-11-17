"""Files + Batch API (OpenAI-compatible): customers upload a JSONL file of requests and collect the results
later, e.g. to classify 50,000 support tickets overnight without managing concurrency or rate limits.

    POST   /v1/files                    upload a JSONL file (multipart, purpose="batch")
    GET    /v1/files                    list files
    GET    /v1/files/{id}               file metadata
    GET    /v1/files/{id}/content       download (input or results)
    DELETE /v1/files/{id}
    POST   /v1/batches                  {"input_file_id", "endpoint", "completion_window": "24h"}
    GET    /v1/batches                  list batches
    GET    /v1/batches/{id}             status and request counts
    POST   /v1/batches/{id}/cancel

Each input line is {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions", "body": {...}}; each
output line is {"id", "custom_id", "response": {"status_code", "body"}, "error"}. Works with the official SDK:
client.files.create(...), client.batches.create(...), client.files.content(...).

A single background worker runs batches in order, one request at a time, through the same handlers (and so
the same quotas, moderation, custom models and metering) as live requests. Progress is written as it goes, so a
server restart resumes a batch where it stopped instead of re-running (and re-billing) finished lines.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

log = logging.getLogger("autolearn.batches")

MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_LINES = 50_000
WINDOW_SECONDS = 24 * 3600
OPEN = ("validating", "in_progress", "finalizing", "cancelling")

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, purpose TEXT NOT NULL, filename TEXT NOT NULL,
  bytes INTEGER NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, key_id TEXT NOT NULL, endpoint TEXT NOT NULL,
  input_file_id TEXT NOT NULL, output_file_id TEXT, error_file_id TEXT, status TEXT NOT NULL,
  total INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
  metadata TEXT, errors TEXT, created_at REAL NOT NULL, in_progress_at REAL, finalizing_at REAL,
  completed_at REAL, failed_at REAL, cancelling_at REAL, cancelled_at REAL, expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS batches_status ON batches(status, created_at);
"""


class BatchIn(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: dict[str, str] | None = None


def _ts(x):
    return int(x) if x else None


def register_batch_routes(app, settings, db, tenant_any, handlers: dict):
    """handlers: endpoint path -> (request model, fn(request, principal) -> response dict)."""
    with db.lock:
        db.conn.executescript(SCHEMA)
    root = Path(settings.data_dir) / "files"
    root.mkdir(parents=True, exist_ok=True)
    wake = threading.Event()
    state = {"thread": None}

    def path_of(fid: str) -> Path:
        return root / f"{fid}.jsonl"

    def file_obj(r):
        return {"id": r["id"], "object": "file", "bytes": r["bytes"], "created_at": _ts(r["created_at"]),
                "filename": r["filename"], "purpose": r["purpose"]}

    def batch_obj(b):
        return {"id": b["id"], "object": "batch", "endpoint": b["endpoint"], "input_file_id": b["input_file_id"],
                "completion_window": "24h", "status": b["status"], "output_file_id": b["output_file_id"],
                "error_file_id": b["error_file_id"],
                "errors": json.loads(b["errors"]) if b["errors"] else None,
                "created_at": _ts(b["created_at"]), "in_progress_at": _ts(b["in_progress_at"]),
                "expires_at": _ts(b["expires_at"]), "finalizing_at": _ts(b["finalizing_at"]),
                "completed_at": _ts(b["completed_at"]), "failed_at": _ts(b["failed_at"]),
                "cancelling_at": _ts(b["cancelling_at"]), "cancelled_at": _ts(b["cancelled_at"]),
                "expired_at": None,
                "request_counts": {"total": b["total"], "completed": b["completed"], "failed": b["failed"]},
                "metadata": json.loads(b["metadata"]) if b["metadata"] else None}

    def own_file(p, fid):
        r = db.one("SELECT * FROM files WHERE id=? AND tenant=?", (fid, p.tenant))
        if not r:
            raise HTTPException(404, f"no file '{fid}'")
        return r

    def own_batch(p, bid):
        b = db.one("SELECT * FROM batches WHERE id=? AND tenant=?", (bid, p.tenant))
        if not b:
            raise HTTPException(404, f"no batch '{bid}'")
        return b

    def save_file(tenant, purpose, filename, data: bytes) -> dict:
        fid = "file-" + uuid.uuid4().hex[:24]
        path_of(fid).write_bytes(data)
        db.x("INSERT INTO files (id, tenant, purpose, filename, bytes, created_at) VALUES (?,?,?,?,?,?)",
             (fid, tenant, purpose, filename, len(data), time.time()))
        return db.one("SELECT * FROM files WHERE id=?", (fid,))

    def parse_input(data: bytes, endpoint: str) -> list[dict]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "file must be UTF-8 JSONL")
        lines, seen = [], set()
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise HTTPException(400, f"line {n}: not valid JSON")
            if not isinstance(obj, dict) or not isinstance(obj.get("custom_id"), str) or not obj["custom_id"]:
                raise HTTPException(400, f"line {n}: needs a non-empty string custom_id")
            if obj["custom_id"] in seen:
                raise HTTPException(400, f"line {n}: duplicate custom_id '{obj['custom_id']}'")
            seen.add(obj["custom_id"])
            if obj.get("method", "POST") != "POST" or obj.get("url") != endpoint:
                raise HTTPException(400, f"line {n}: method must be POST and url must be {endpoint}")
            if not isinstance(obj.get("body"), dict):
                raise HTTPException(400, f"line {n}: body must be an object")
            lines.append(obj)
        if not lines:
            raise HTTPException(400, "file has no requests")
        if len(lines) > MAX_LINES:
            raise HTTPException(400, f"at most {MAX_LINES:,} requests per batch")
        return lines

    # ------------------------------------------------------------ worker --
    def principal(b):
        from .api import Principal
        t = db.get_tenant(b["tenant"])
        k = db.one("SELECT role, revoked FROM api_keys WHERE id=?", (b["key_id"],))
        if not t or not t["active"]:
            raise HTTPException(403, "tenant account is suspended")
        if not k or k["revoked"]:
            raise HTTPException(401, "the API key that created this batch was revoked")
        return Principal(k["role"], b["tenant"], b["key_id"], t["plan"])

    def call(b, line) -> tuple[int, dict]:
        model, fn = handlers[b["endpoint"]]
        try:
            p = principal(b)
            req = model(**{**line["body"], "stream": False}) if "stream" in model.model_fields \
                else model(**line["body"])
            return 200, fn(req, p)
        except HTTPException as e:
            d = e.detail if isinstance(e.detail, dict) and "error" in e.detail else \
                {"error": {"message": str(e.detail), "type": "invalid_request_error"}}
            return e.status_code, d
        except ValidationError as e:
            return 400, {"error": {"message": e.errors(include_url=False, include_context=False).__repr__()[:2000],
                                   "type": "invalid_request_error"}}
        except Exception as e:  # noqa: BLE001 (one bad request must not stop the batch)
            log.exception("batch %s line %s failed", b["id"], line.get("custom_id"))
            return 500, {"error": {"message": f"internal error: {type(e).__name__}", "type": "server_error"}}

    def finish(b, status, **extra):
        now = time.time()
        f = {"finalizing_at": now, f"{status}_at": now, "status": status, **extra}
        for fid_col, suffix in (("output_file_id", "out"), ("error_file_id", "err")):
            tmp = root / f"{b['id']}.{suffix}"
            if tmp.exists() and tmp.stat().st_size > 0:
                r = save_file(b["tenant"], "batch_output", f"{b['id']}_{'output' if suffix == 'out' else 'errors'}"
                                                           ".jsonl", tmp.read_bytes())
                f[fid_col] = r["id"]
            tmp.unlink(missing_ok=True)
        sets = ", ".join(f"{k}=?" for k in f)
        db.x(f"UPDATE batches SET {sets} WHERE id=?", (*f.values(), b["id"]))

    def process(b):
        if b["status"] == "validating":
            db.x("UPDATE batches SET status='in_progress', in_progress_at=? WHERE id=?", (time.time(), b["id"]))
        lines = parse_input(path_of(b["input_file_id"]).read_bytes(), b["endpoint"])
        done = b["completed"] + b["failed"]  # resume after a restart
        out, err = root / f"{b['id']}.out", root / f"{b['id']}.err"
        for line in lines[done:]:
            cur = db.one("SELECT status, expires_at FROM batches WHERE id=?", (b["id"],))
            if cur["status"] == "cancelling":
                return finish(b, "cancelled")
            if time.time() > cur["expires_at"]:
                return finish(b, "expired")
            status, body = call(b, line)
            ok = status == 200
            rec = {"id": "batch_req_" + uuid.uuid4().hex[:24], "custom_id": line["custom_id"],
                   "response": {"status_code": status, "request_id": uuid.uuid4().hex, "body": body} if ok else None,
                   "error": None if ok else {"code": str(status), "message": body["error"]["message"]}}
            with open(out if ok else err, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            db.x(f"UPDATE batches SET {'completed' if ok else 'failed'} = {'completed' if ok else 'failed'} + 1 "
                 "WHERE id=?", (b["id"],))
        finish(b, "completed")

    def loop():
        while True:
            b = db.one("SELECT * FROM batches WHERE status IN ('validating','in_progress','cancelling') "
                       "ORDER BY created_at LIMIT 1")
            if b is None:
                wake.wait(30)
                wake.clear()
                continue
            try:
                if b["status"] == "cancelling":
                    finish(b, "cancelled")
                else:
                    process(b)
            except Exception as e:  # noqa: BLE001
                log.exception("batch %s failed", b["id"])
                finish(b, "failed", errors=json.dumps({"object": "list", "data": [
                    {"code": "internal_error", "message": f"{type(e).__name__}: {e}"[:500]}]}))

    def ensure_worker():
        if state["thread"] is None or not state["thread"].is_alive():
            state["thread"] = threading.Thread(target=loop, name="batch-worker", daemon=True)
            state["thread"].start()
        wake.set()

    if db.one("SELECT 1 AS x FROM batches WHERE status IN ('validating','in_progress','cancelling') LIMIT 1"):
        ensure_worker()  # resume batches interrupted by a restart

    # ------------------------------------------------------------ routes --
    @app.post("/v1/files", tags=["batch"], status_code=200)
    async def upload_file(file: UploadFile = File(...), purpose: str = Form(...), p=Depends(tenant_any)):
        if purpose != "batch":
            raise HTTPException(400, "only purpose='batch' is supported")
        data = await file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"files are limited to {MAX_FILE_BYTES // 2**20} MB")
        if not data.strip():
            raise HTTPException(400, "file is empty")
        return file_obj(save_file(p.tenant, purpose, file.filename or "upload.jsonl", data))

    @app.get("/v1/files", tags=["batch"])
    def list_files(p=Depends(tenant_any)):
        rows = db.q("SELECT * FROM files WHERE tenant=? ORDER BY created_at DESC LIMIT 1000", (p.tenant,))
        return {"object": "list", "data": [file_obj(r) for r in rows], "has_more": False}

    @app.get("/v1/files/{fid}", tags=["batch"])
    def get_file(fid: str, p=Depends(tenant_any)):
        return file_obj(own_file(p, fid))

    @app.get("/v1/files/{fid}/content", tags=["batch"])
    def file_content(fid: str, p=Depends(tenant_any)):
        r = own_file(p, fid)
        return FileResponse(path_of(r["id"]), media_type="application/jsonl", filename=r["filename"])

    @app.delete("/v1/files/{fid}", tags=["batch"])
    def delete_file(fid: str, p=Depends(tenant_any)):
        r = own_file(p, fid)
        if db.one("SELECT 1 AS x FROM batches WHERE input_file_id=? AND status IN "
                  "('validating','in_progress','finalizing','cancelling')", (fid,)):
            raise HTTPException(409, "file is the input of a running batch")
        path_of(r["id"]).unlink(missing_ok=True)
        db.x("DELETE FROM files WHERE id=?", (fid,))
        return {"id": fid, "object": "file", "deleted": True}

    @app.post("/v1/batches", tags=["batch"])
    def create_batch(body: BatchIn, p=Depends(tenant_any)):
        if body.endpoint not in handlers:
            raise HTTPException(400, "endpoint must be one of: " + ", ".join(handlers))
        if body.completion_window != "24h":
            raise HTTPException(400, "completion_window must be '24h'")
        f = own_file(p, body.input_file_id)
        if f["purpose"] != "batch":
            raise HTTPException(400, "input file must have purpose 'batch'")
        n = len(parse_input(path_of(f["id"]).read_bytes(), body.endpoint))  # reject bad files up front
        bid = "batch_" + uuid.uuid4().hex[:24]
        now = time.time()
        db.x("INSERT INTO batches (id, tenant, key_id, endpoint, input_file_id, status, total, metadata, created_at, "
             "expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
             (bid, p.tenant, p.key_id, body.endpoint, f["id"], "validating", n,
              json.dumps(body.metadata) if body.metadata else None, now, now + WINDOW_SECONDS))
        db.add_usage(p.tenant, "batches", 1)
        ensure_worker()
        return batch_obj(own_batch(p, bid))

    @app.get("/v1/batches", tags=["batch"])
    def list_batches(limit: int = 20, p=Depends(tenant_any)):
        rows = db.q("SELECT * FROM batches WHERE tenant=? ORDER BY created_at DESC LIMIT ?",
                    (p.tenant, max(1, min(limit, 100))))
        return {"object": "list", "data": [batch_obj(b) for b in rows], "has_more": False,
                "first_id": rows[0]["id"] if rows else None, "last_id": rows[-1]["id"] if rows else None}

    @app.get("/v1/batches/{bid}", tags=["batch"])
    def get_batch(bid: str, p=Depends(tenant_any)):
        return batch_obj(own_batch(p, bid))

    @app.post("/v1/batches/{bid}/cancel", tags=["batch"])
    def cancel_batch(bid: str, p=Depends(tenant_any)):
        b = own_batch(p, bid)
        if b["status"] in ("validating", "in_progress"):
            db.x("UPDATE batches SET status='cancelling', cancelling_at=? WHERE id=? AND status IN "
                 "('validating','in_progress')", (time.time(), bid))
            ensure_worker()
        return batch_obj(own_batch(p, bid))
