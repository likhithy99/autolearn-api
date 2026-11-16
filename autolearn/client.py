"""Python SDK for AutoLearn customers.

    from autolearn.client import AutoLearn
    al = AutoLearn("https://api.example.com", api_key="al_cli_...")
    out = al.predict("churn", {"tenure": 3, "plan": "basic"})
    al.feedback("churn", out["predictions"][0]["prediction_id"], actual="yes")

    reply = al.chat("What is our refund policy?", knowledge_base="policies")
    al.rate(reply["id"], -1, correction="Enterprise refunds are available within 60 days.")
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class AutoLearnError(Exception):
    def __init__(self, status: int, detail: Any, request_id: str | None = None):
        super().__init__(f"HTTP {status}: {detail} (request_id={request_id})")
        self.status, self.detail, self.request_id = status, detail, request_id


class AutoLearn:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0, max_retries: int = 3):
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                  headers={"X-API-Key": api_key, "User-Agent": "autolearn-python/1.0"})
        self.max_retries = max_retries

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _req(self, method: str, path: str, **kw):
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.request(method, path, **kw)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            if r.status_code >= 400:
                try:
                    detail = r.json().get("detail")
                except Exception:
                    detail = r.text
                raise AutoLearnError(r.status_code, detail, r.headers.get("X-Request-ID"))
            return r.json()

    # account
    def me(self):
        return self._req("GET", "/v1/me")

    def create_key(self, name: str, role: str = "client"):
        return self._req("POST", "/v1/keys", json={"name": name, "role": role})

    # projects
    def create_project(self, name: str, task: str, target: str, **config):
        return self._req("POST", "/v1/projects", json={"name": name, "task": task, "target": target,
                                                       "config": config})

    def project(self, name: str):
        return self._req("GET", f"/v1/projects/{name}")

    def update_config(self, name: str, **config):
        return self._req("PATCH", f"/v1/projects/{name}", json={"config": config})

    def delete_project(self, name: str):
        return self._req("DELETE", f"/v1/projects/{name}")

    # data & training
    def upload(self, name: str, rows: list[dict], batch_size: int = 5000):
        total = 0
        for i in range(0, len(rows), batch_size):
            total += self._req("POST", f"/v1/projects/{name}/data", json={"rows": rows[i:i + batch_size]})["accepted"]
        return {"accepted": total}

    def upload_csv(self, name: str, path: str):
        with open(path, "rb") as f:
            return self._req("POST", f"/v1/projects/{name}/data/csv", files={"file": (path, f, "text/csv")})

    def train(self, name: str, wait: bool = True):
        return self._req("POST", f"/v1/projects/{name}/train", params={"wait": str(wait).lower()})

    def models(self, name: str):
        return self._req("GET", f"/v1/projects/{name}/models")

    def promote(self, name: str, version: int):
        return self._req("POST", f"/v1/projects/{name}/models/{version}/promote")

    # inference
    def predict(self, name: str, rows: dict | list[dict], explain: bool = False):
        body = {"features": rows} if isinstance(rows, dict) else {"rows": rows}
        body["explain"] = explain
        return self._req("POST", f"/v1/projects/{name}/predict", json=body)

    def feedback(self, name: str, prediction_id: str | list[tuple[str, Any]], actual: Any = None):
        items = ([{"prediction_id": prediction_id, "actual": actual}] if isinstance(prediction_id, str)
                 else [{"prediction_id": p, "actual": a} for p, a in prediction_id])
        return self._req("POST", f"/v1/projects/{name}/feedback", json={"items": items})

    # LLM (OpenAI-compatible; the official `openai` SDK also works with base_url=<host>/v1)
    def chat(self, messages: list[dict] | str, knowledge_base: str | None = None, **params):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"messages": messages, **params}
        if knowledge_base:
            body["knowledge_base"] = knowledge_base
        return self._req("POST", "/v1/chat/completions", json=body)

    def rate(self, completion_id: str, rating: int, correction: str | None = None):
        """Thumbs up (+1) / down (-1), optionally with the ideal answer; drives continual learning."""
        return self._req("POST", "/v1/feedback", json={"completion_id": completion_id, "rating": rating,
                                                       "correction": correction})

    # knowledge bases (RAG)
    def add_documents(self, kb: str, documents: list[dict]):
        """documents: [{"id": optional, "title": optional, "text": "..."}]"""
        return self._req("POST", f"/v1/knowledge/{kb}/documents", json={"documents": documents})

    def search(self, kb: str, query: str, top_k: int = 5):
        return self._req("POST", f"/v1/knowledge/{kb}/search", json={"query": query, "top_k": top_k})["results"]

    def delete_document(self, kb: str, doc_id: str):
        return self._req("DELETE", f"/v1/knowledge/{kb}/documents/{doc_id}")

    # monitoring
    def monitor(self, name: str):
        return self._req("GET", f"/v1/projects/{name}/monitor")

    def events(self, name: str, limit: int = 50):
        return self._req("GET", f"/v1/projects/{name}/events", params={"limit": limit})
