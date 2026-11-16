"""SQLite persistence: projects, training samples, prediction log, model history, events, API keys."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  plan TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
  tenant TEXT NOT NULL,
  period TEXT NOT NULL,              -- YYYY-MM
  metric TEXT NOT NULL,              -- predictions | trainings | rows_ingested | feedback
  count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (tenant, period, metric)
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  tenant TEXT,
  key_id TEXT,
  action TEXT NOT NULL,
  detail TEXT
);
CREATE TABLE IF NOT EXISTS projects (
  name TEXT PRIMARY KEY,             -- internal key: '<tenant>/<project>'
  tenant TEXT NOT NULL,
  short_name TEXT NOT NULL,
  task TEXT NOT NULL,
  target TEXT NOT NULL,
  config TEXT NOT NULL,
  champion_version INTEGER,
  last_trained_sample_id INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  features TEXT NOT NULL,
  label TEXT NOT NULL,
  source TEXT NOT NULL,              -- upload | feedback | pseudo
  prediction_id TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_samples_project ON samples(project, id);
CREATE INDEX IF NOT EXISTS ix_samples_pred ON samples(prediction_id);
CREATE TABLE IF NOT EXISTS predictions (
  id TEXT PRIMARY KEY,
  project TEXT NOT NULL,
  features TEXT NOT NULL,
  prediction TEXT NOT NULL,
  confidence REAL,
  model_version INTEGER NOT NULL,
  created_at REAL NOT NULL,
  actual TEXT,
  feedback_at REAL
);
CREATE INDEX IF NOT EXISTS ix_pred_project ON predictions(project, model_version, created_at);
CREATE TABLE IF NOT EXISTS models (
  project TEXT NOT NULL,
  version INTEGER NOT NULL,
  algorithm TEXT NOT NULL,
  status TEXT NOT NULL,              -- champion | retired | rejected
  metrics TEXT NOT NULL,
  reason TEXT,
  n_train INTEGER,
  created_at REAL NOT NULL,
  PRIMARY KEY (project, version)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_completions (
  id TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  key_id TEXT,
  model_version TEXT NOT NULL,
  kind TEXT NOT NULL,                -- chat | completion
  request TEXT NOT NULL,             -- messages (chat) or prompt (completion), JSON
  output TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_llm_completions_tenant ON llm_completions(tenant, created_at);
CREATE TABLE IF NOT EXISTS llm_feedback (
  completion_id TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  rating INTEGER NOT NULL,           -- +1 good, -1 bad
  correction TEXT,                   -- the ideal answer, if the user supplied one
  created_at REAL NOT NULL,
  used_in_version TEXT
);
CREATE TABLE IF NOT EXISTS stripe_events (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  received_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkout_reveals (
  session_id TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  api_key TEXT,                      -- shown to the buyer exactly once, then erased
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY,
  key_hash TEXT UNIQUE NOT NULL,
  prefix TEXT NOT NULL,
  tenant TEXT NOT NULL,
  name TEXT NOT NULL,
  role TEXT NOT NULL,                -- admin | client
  revoked INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  last_used_at REAL
);
"""


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def dumps(v) -> str:
    return json.dumps(v, default=_json_default)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(tenants)")}
            if "train_opt_in" not in cols:  # migration: customers must opt in to training on their data
                self.conn.execute("ALTER TABLE tenants ADD COLUMN train_opt_in INTEGER NOT NULL DEFAULT 0")
            for col in ("email", "stripe_customer_id", "stripe_subscription_id", "moderation"):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE tenants ADD COLUMN {col} TEXT")

    # -- low level -----------------------------------------------------------
    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql, args=()):
        with self.lock:
            return self.conn.execute(sql, args).lastrowid

    def many(self, sql, seq):
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(sql, seq)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    # -- projects ------------------------------------------------------------
    @staticmethod
    def _project(row):
        if row:
            row["config"] = json.loads(row["config"])
            row["state"] = json.loads(row["state"])
        return row

    def create_project(self, key, tenant, short_name, task, target, config):
        self.x("INSERT INTO projects(name, tenant, short_name, task, target, config, created_at) "
               "VALUES (?,?,?,?,?,?,?)", (key, tenant, short_name, task, target, dumps(config), time.time()))

    def get_project(self, name):
        return self._project(self.one("SELECT * FROM projects WHERE name=?", (name,)))

    def list_projects(self, tenant=None):
        if tenant is None:
            return [self._project(r) for r in self.q("SELECT * FROM projects ORDER BY name")]
        return [self._project(r) for r in self.q("SELECT * FROM projects WHERE tenant=? ORDER BY name", (tenant,))]

    def count_projects(self, tenant):
        return self.one("SELECT COUNT(*) n FROM projects WHERE tenant=?", (tenant,))["n"]

    def count_tenant_samples(self, tenant):
        return self.one("SELECT COUNT(*) n FROM samples WHERE project IN "
                        "(SELECT name FROM projects WHERE tenant=?)", (tenant,))["n"]

    def update_project(self, name, **fields):
        for k in ("config", "state"):
            if k in fields:
                fields[k] = dumps(fields[k])
        cols = ", ".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE projects SET {cols} WHERE name=?", (*fields.values(), name))

    def delete_project(self, name):
        with self.lock:
            for t in ("samples", "predictions", "models", "events"):
                self.conn.execute(f"DELETE FROM {t} WHERE project=?", (name,))
            self.conn.execute("DELETE FROM projects WHERE name=?", (name,))

    # -- samples -------------------------------------------------------------
    def add_samples(self, project, rows):
        """rows: iterable of (features: dict, label, source, prediction_id|None)."""
        now = time.time()
        self.many("INSERT INTO samples(project, features, label, source, prediction_id, created_at) VALUES (?,?,?,?,?,?)",
                  [(project, dumps(f), dumps(l), s, pid, now) for f, l, s, pid in rows])

    def labeled_samples(self, project, limit):
        rows = self.q("SELECT id, features, label, source FROM samples WHERE project=? ORDER BY id DESC LIMIT ?",
                      (project, limit))
        for r in rows:
            r["features"] = json.loads(r["features"])
            r["label"] = json.loads(r["label"])
        rows.reverse()
        return rows

    def sample_counts(self, project):
        return {r["source"]: r["n"] for r in
                self.q("SELECT source, COUNT(*) n FROM samples WHERE project=? GROUP BY source", (project,))}

    def count_real(self, project, after_id=0):
        return self.one("SELECT COUNT(*) n FROM samples WHERE project=? AND source!='pseudo' AND id>?",
                        (project, after_id))["n"]

    def max_sample_id(self, project):
        return self.one("SELECT COALESCE(MAX(id),0) m FROM samples WHERE project=?", (project,))["m"]

    # -- predictions ---------------------------------------------------------
    def add_predictions(self, rows):
        """rows: (id, project, features, prediction, confidence, model_version)."""
        now = time.time()
        self.many("INSERT INTO predictions(id, project, features, prediction, confidence, model_version, created_at) "
                  "VALUES (?,?,?,?,?,?,?)",
                  [(i, p, dumps(f), dumps(pr), c, v, now) for i, p, f, pr, c, v in rows])

    def get_prediction(self, pid):
        r = self.one("SELECT * FROM predictions WHERE id=?", (pid,))
        if r:
            r["features"] = json.loads(r["features"])
            r["prediction"] = json.loads(r["prediction"])
        return r

    def recent_inputs(self, project, version, limit):
        return [json.loads(r["features"]) for r in self.q(
            "SELECT features FROM predictions WHERE project=? AND model_version=? ORDER BY created_at DESC LIMIT ?",
            (project, version, limit))]

    def recent_feedback(self, project, version, limit):
        return [(json.loads(r["prediction"]), json.loads(r["actual"])) for r in self.q(
            "SELECT prediction, actual FROM predictions WHERE project=? AND model_version=? AND actual IS NOT NULL "
            "ORDER BY feedback_at DESC LIMIT ?", (project, version, limit))]

    def prediction_counts(self, project):
        return self.one("SELECT COUNT(*) total, COUNT(actual) with_feedback FROM predictions WHERE project=?",
                        (project,))

    def record_feedback(self, pred, actual):
        """Attach ground truth to a prediction and turn it into a training sample (replacing any pseudo-label)."""
        now = time.time()
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute("UPDATE predictions SET actual=?, feedback_at=? WHERE id=?",
                                  (dumps(actual), now, pred["id"]))
                self.conn.execute("DELETE FROM samples WHERE prediction_id=?", (pred["id"],))
                self.conn.execute(
                    "INSERT INTO samples(project, features, label, source, prediction_id, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (pred["project"], dumps(pred["features"]), dumps(actual), "feedback", pred["id"], now))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    # -- models --------------------------------------------------------------
    def next_model_version(self, project):
        return self.one("SELECT COALESCE(MAX(version),0)+1 v FROM models WHERE project=?", (project,))["v"]

    def add_model(self, project, version, algorithm, status, metrics, reason, n_train):
        self.x("INSERT INTO models VALUES (?,?,?,?,?,?,?,?)",
               (project, version, algorithm, status, dumps(metrics), reason, n_train, time.time()))

    def get_model(self, project, version):
        r = self.one("SELECT * FROM models WHERE project=? AND version=?", (project, version))
        if r:
            r["metrics"] = json.loads(r["metrics"])
        return r

    def list_models(self, project):
        rows = self.q("SELECT * FROM models WHERE project=? ORDER BY version DESC", (project,))
        for r in rows:
            r["metrics"] = json.loads(r["metrics"])
        return rows

    def promote(self, project, version):
        with self.lock:
            self.conn.execute("UPDATE models SET status='retired' WHERE project=? AND status='champion'", (project,))
            self.conn.execute("UPDATE models SET status='champion' WHERE project=? AND version=?", (project, version))
            self.conn.execute("UPDATE projects SET champion_version=? WHERE name=?", (version, project))

    # -- events --------------------------------------------------------------
    def log_event(self, project, kind, message):
        self.x("INSERT INTO events(project, ts, kind, message) VALUES (?,?,?,?)",
               (project, time.time(), kind, message))

    def events(self, project, limit=50):
        return self.q("SELECT ts, kind, message FROM events WHERE project=? ORDER BY id DESC LIMIT ?",
                      (project, limit))

    # -- tenants -------------------------------------------------------------
    def create_tenant(self, tid, name, plan):
        self.x("INSERT INTO tenants(id, name, plan, active, created_at) VALUES (?,?,?,1,?)",
               (tid, name, plan, time.time()))

    def get_tenant(self, tid):
        return self.one("SELECT * FROM tenants WHERE id=?", (tid,))

    def list_tenants(self):
        return self.q("SELECT * FROM tenants ORDER BY created_at")

    def update_tenant(self, tid, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE tenants SET {cols} WHERE id=?", (*fields.values(), tid))

    def tenant_by(self, field, value):
        assert field in ("stripe_customer_id", "stripe_subscription_id", "email")
        return self.one(f"SELECT * FROM tenants WHERE {field}=?", (value,))

    # -- billing -------------------------------------------------------------------
    def event_seen(self, event_id, etype):
        """Idempotency for webhooks: returns True if this event was already processed."""
        with self.lock:
            try:
                self.conn.execute("INSERT INTO stripe_events VALUES (?,?,?)", (event_id, etype, time.time()))
                return False
            except sqlite3.IntegrityError:
                return True

    def add_reveal(self, session_id, tenant, api_key):
        self.x("INSERT OR IGNORE INTO checkout_reveals VALUES (?,?,?,?)", (session_id, tenant, api_key, time.time()))

    def take_reveal(self, session_id):
        """Return the key once, then erase it."""
        with self.lock:
            r = self.conn.execute("SELECT tenant, api_key FROM checkout_reveals WHERE session_id=?",
                                  (session_id,)).fetchone()
            if not r:
                return None
            self.conn.execute("UPDATE checkout_reveals SET api_key=NULL WHERE session_id=?", (session_id,))
            return {"tenant": r["tenant"], "api_key": r["api_key"]}

    # -- usage metering --------------------------------------------------------
    @staticmethod
    def period():
        return time.strftime("%Y-%m", time.gmtime())

    def add_usage(self, tenant, metric, n=1):
        self.x("INSERT INTO usage VALUES (?,?,?,?) ON CONFLICT(tenant, period, metric) "
               "DO UPDATE SET count=count+excluded.count", (tenant, self.period(), metric, n))

    def usage(self, tenant, period=None):
        return {r["metric"]: r["count"] for r in self.q(
            "SELECT metric, count FROM usage WHERE tenant=? AND period=?", (tenant, period or self.period()))}

    def usage_history(self, tenant):
        out = {}
        for r in self.q("SELECT period, metric, count FROM usage WHERE tenant=? ORDER BY period", (tenant,)):
            out.setdefault(r["period"], {})[r["metric"]] = r["count"]
        return out

    # -- audit -------------------------------------------------------------------
    def audit(self, tenant, key_id, action, detail=None):
        self.x("INSERT INTO audit_log(ts, tenant, key_id, action, detail) VALUES (?,?,?,?,?)",
               (time.time(), tenant, key_id, action, dumps(detail) if detail is not None else None))

    def audit_log(self, tenant, limit=100):
        return self.q("SELECT ts, key_id, action, detail FROM audit_log WHERE tenant=? ORDER BY id DESC LIMIT ?",
                      (tenant, limit))

    # -- llm ---------------------------------------------------------------------
    def add_completion(self, cid, tenant, key_id, version, kind, request, output, pt, ct):
        self.x("INSERT INTO llm_completions VALUES (?,?,?,?,?,?,?,?,?,?)",
               (cid, tenant, key_id, version, kind, dumps(request), output, pt, ct, time.time()))

    def get_completion(self, cid):
        r = self.one("SELECT * FROM llm_completions WHERE id=?", (cid,))
        if r:
            r["request"] = json.loads(r["request"])
        return r

    def add_llm_feedback(self, cid, tenant, rating, correction):
        self.x("INSERT INTO llm_feedback(completion_id, tenant, rating, correction, created_at) VALUES (?,?,?,?,?) "
               "ON CONFLICT(completion_id) DO UPDATE SET rating=excluded.rating, correction=excluded.correction, "
               "created_at=excluded.created_at, used_in_version=NULL", (cid, tenant, rating, correction, time.time()))

    def unused_llm_feedback(self):
        """Feedback from tenants that opted in to training, not yet consumed by a fine-tune."""
        rows = self.q(
            "SELECT f.completion_id, f.rating, f.correction, c.kind, c.request, c.output FROM llm_feedback f "
            "JOIN llm_completions c ON c.id = f.completion_id JOIN tenants t ON t.id = f.tenant "
            "WHERE f.used_in_version IS NULL AND t.train_opt_in = 1 ORDER BY f.created_at")
        for r in rows:
            r["request"] = json.loads(r["request"])
        return rows

    def mark_llm_feedback_used(self, ids, version):
        self.many("UPDATE llm_feedback SET used_in_version=? WHERE completion_id=?", [(version, i) for i in ids])

    def llm_feedback_stats(self, tenant=None):
        where, args = ("WHERE tenant=?", (tenant,)) if tenant else ("", ())
        return self.one(f"SELECT COUNT(*) total, SUM(rating>0) positive, SUM(rating<0) negative, "
                        f"SUM(correction IS NOT NULL) corrections, SUM(used_in_version IS NOT NULL) used "
                        f"FROM llm_feedback {where}", args)

    def purge_llm_logs(self, older_than_days):
        """Retention policy: drop completion logs (and their unused feedback) past the retention window."""
        cutoff = time.time() - older_than_days * 86400
        with self.lock:
            self.conn.execute("DELETE FROM llm_feedback WHERE completion_id IN "
                              "(SELECT id FROM llm_completions WHERE created_at < ?)", (cutoff,))
            return self.conn.execute("DELETE FROM llm_completions WHERE created_at < ?", (cutoff,)).rowcount

    # -- api keys ------------------------------------------------------------
    def add_key(self, key_id, key, tenant, name, role):
        self.x("INSERT INTO api_keys(id, key_hash, prefix, tenant, name, role, created_at) VALUES (?,?,?,?,?,?,?)",
               (key_id, hash_key(key), key[:12], tenant, name, role, time.time()))

    def lookup_key(self, key):
        r = self.one("SELECT k.id, k.tenant, k.role, k.revoked, t.active, t.plan FROM api_keys k "
                     "JOIN tenants t ON t.id = k.tenant WHERE k.key_hash=?", (hash_key(key),))
        if r:
            self.x("UPDATE api_keys SET last_used_at=? WHERE id=?", (time.time(), r["id"]))
        return r

    def list_keys(self, tenant):
        return self.q("SELECT id, prefix, name, role, revoked, created_at, last_used_at FROM api_keys "
                      "WHERE tenant=? ORDER BY created_at", (tenant,))

    def revoke_key(self, tenant, key_id):
        with self.lock:
            return self.conn.execute("UPDATE api_keys SET revoked=1 WHERE tenant=? AND id=?",
                                     (tenant, key_id)).rowcount
