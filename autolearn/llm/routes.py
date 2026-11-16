"""OpenAI-compatible LLM endpoints, sharing tenants / keys / quotas / metering with the rest of the API.

Customers can use the official `openai` SDK unchanged:
    from openai import OpenAI
    client = OpenAI(base_url="https://your-host/v1", api_key="al_cli_...")
    client.chat.completions.create(model="autolearn-llm", messages=[{"role": "user", "content": "Hi"}])
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..config import PLANS

log = logging.getLogger("autolearn.llm")
MODEL_ID = "autolearn-llm"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str | list[dict[str, Any]] | None = None


class _GenParams(BaseModel):
    model_config = ConfigDict(extra="allow")  # accept (and ignore) other OpenAI params
    model: str | None = MODEL_ID
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, ge=0, le=1000)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stream_options: dict | None = None


class ChatRequest(_GenParams):
    messages: list[ChatMessage] = Field(min_length=1, max_length=500)
    knowledge_base: str | None = Field(default=None, description="ground the answer in this knowledge base")
    knowledge_top_k: int = Field(default=4, ge=1, le=20)


class CompletionRequest(_GenParams):
    prompt: str = Field(max_length=200_000)


class LLMFeedbackIn(BaseModel):
    completion_id: str
    rating: Literal[1, -1]
    correction: str | None = Field(default=None, max_length=20_000)


class TenantSettings(BaseModel):
    train_opt_in: bool


class RollbackIn(BaseModel):
    version: str


class EngineHolder:
    """Loads the CURRENT published model and hot-swaps when a new version is promoted."""

    def __init__(self, llm_dir: Path | None, device: str | None):
        self.llm_dir, self.device = llm_dir, device
        self._engine = None
        self._lock = threading.Lock()

    def registry(self):
        if not self.llm_dir:
            raise HTTPException(503, "LLM is not configured on this server (set AUTOLEARN_LLM_DIR)")
        from .engine import LLMRegistry
        return LLMRegistry(self.llm_dir)

    def get(self):
        reg = self.registry()
        current = reg.current()
        if not current:
            raise HTTPException(503, "no LLM version has been published yet")
        with self._lock:
            if self._engine is None or self._engine.version != current:
                from .engine import Engine
                log.info("loading LLM %s", current)
                self._engine = Engine(reg.path(current), self.device)
            return self._engine


def register_llm_routes(app, settings, db, metrics, tenant_any, tenant_admin, platform, knowledge=None,
                        moderate=None):
    holder = EngineHolder(settings.llm_dir, settings.llm_device)
    app.state.llm = holder

    def check_tokens(p, needed: int):
        limit = PLANS[p.plan].get("llm_tokens_per_month")
        if limit is not None and db.usage(p.tenant).get("llm_tokens", 0) + needed > limit:
            raise HTTPException(402, f"monthly LLM token quota of {limit:,} exceeded for plan '{p.plan}'")

    def gen_kwargs(req: _GenParams):
        stop = [req.stop] if isinstance(req.stop, str) else (req.stop or None)
        return dict(max_tokens=req.max_completion_tokens or req.max_tokens or 256, temperature=req.temperature,
                    top_p=req.top_p, top_k=req.top_k, repetition_penalty=req.repetition_penalty, stop=stop,
                    seed=req.seed)

    def record(p, cid, engine, kind, request, text, final):
        pt, ct = final.get("prompt_tokens", 0), final.get("completion_tokens", 0)
        db.add_completion(cid, p.tenant, p.key_id, engine.version, kind, request, text, pt, ct)
        db.add_usage(p.tenant, "llm_tokens", pt + ct)
        db.add_usage(p.tenant, "llm_requests", 1)
        metrics.inc("autolearn_llm_tokens_total", {"type": "prompt"}, pt)
        metrics.inc("autolearn_llm_tokens_total", {"type": "completion"}, ct)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

    def run(p, kind: str, req, stream_factory, request_payload, chunk_builder, full_builder, extra=None):
        extra = extra or {}
        engine = holder.get()
        kw = gen_kwargs(req)
        check_tokens(p, kw["max_tokens"])
        cid = ("chatcmpl-" if kind == "chat" else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        t0 = time.perf_counter()

        if not req.stream:
            text, final = "", {}
            for ev in stream_factory(engine, kw):
                if "delta" in ev:
                    text += ev["delta"]
                else:
                    final = ev
            finish = final["finish_reason"]
            if moderate and text:
                try:
                    out_mod = moderate(p.tenant, [text], "output")
                except HTTPException:  # tenant blocks flagged output: withhold it
                    out_mod = {"flagged": True, "where": "output", "blocked": True}
                    text, finish = "", "content_filter"
                if out_mod:
                    extra = {**extra, "moderation": {**extra.get("moderation", {}), "output": out_mod}}
            usage = record(p, cid, engine, kind, request_payload, text, final)
            metrics.observe("autolearn_llm_request_seconds", time.perf_counter() - t0)
            return {**full_builder(cid, created, engine.version, text, finish, usage), **extra}

        include_usage = bool((req.stream_options or {}).get("include_usage"))

        def sse():
            text, final = "", {"finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0}
            try:
                if kind == "chat":
                    yield _sse({**chunk_builder(cid, created, engine.version, {"role": "assistant", "content": ""},
                                                None), **extra})
                for ev in stream_factory(engine, kw):
                    if "delta" in ev:
                        text += ev["delta"]
                        yield _sse(chunk_builder(cid, created, engine.version, ev["delta"], None))
                    else:
                        final = ev
                yield _sse(chunk_builder(cid, created, engine.version, None, final["finish_reason"]))
            finally:  # metered even if the client disconnects mid-stream
                usage = record(p, cid, engine, kind, request_payload, text, final)
                metrics.observe("autolearn_llm_request_seconds", time.perf_counter() - t0)
            if include_usage:
                yield _sse({"id": cid, "object": "chat.completion.chunk" if kind == "chat" else "text_completion",
                            "created": created, "model": f"{MODEL_ID}:{engine.version}", "choices": [],
                            "usage": usage})
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ routes --
    @app.get("/v1/models", tags=["llm"])
    def list_models(p=Depends(tenant_any)):
        cur = holder.registry().current() if settings.llm_dir else None
        data = []
        if cur:
            data.append({"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "autolearn",
                         "version": cur})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions", tags=["llm"])
    def chat_completions(req: ChatRequest, p=Depends(tenant_any)):
        messages = [{"role": "system" if m.role == "developer" else m.role, "content": m.content}
                    for m in req.messages if m.role != "tool"]
        extra = {}
        if moderate:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
            if isinstance(last_user, str) and last_user:
                in_mod = moderate(p.tenant, [last_user], "input")  # raises 400 in block mode
                if in_mod:
                    extra["moderation"] = {"input": in_mod}
        if req.knowledge_base:
            if knowledge is None:
                raise HTTPException(503, "knowledge bases are not enabled")
            from ..knowledge import grounding_message
            question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "") or ""
            if isinstance(question, list):
                question = " ".join(x.get("text", "") for x in question if isinstance(x, dict))
            passages = knowledge.search(p.tenant, req.knowledge_base, question, req.knowledge_top_k)
            db.add_usage(p.tenant, "kb_searches", 1)
            ground = grounding_message(passages) if passages else \
                "No relevant sources were found. Say you don't know if the question needs specific facts."
            sys_msgs = [m for m in messages if m["role"] == "system"]
            if sys_msgs:
                sys_msgs[0]["content"] = f"{sys_msgs[0]['content']}\n\n{ground}"
            else:
                messages = [{"role": "system", "content": ground}] + messages
            extra["citations"] = [{k: v for k, v in x.items() if k != "score"} | {"score": x["score"]}
                                  for x in passages]

        def chunk(cid, created, ver, delta, finish):
            d = {} if delta is None else (delta if isinstance(delta, dict) else {"content": delta})
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": f"{MODEL_ID}:{ver}", "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}

        def full(cid, created, ver, text, finish, usage):
            return {"id": cid, "object": "chat.completion", "created": created, "model": f"{MODEL_ID}:{ver}",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": finish}], "usage": usage}

        return run(p, "chat", req, lambda e, kw: e.chat_stream(messages, **kw), messages, chunk, full, extra)

    @app.post("/v1/completions", tags=["llm"])
    def completions(req: CompletionRequest, p=Depends(tenant_any)):
        def chunk(cid, created, ver, delta, finish):
            return {"id": cid, "object": "text_completion", "created": created, "model": f"{MODEL_ID}:{ver}",
                    "choices": [{"index": 0, "text": delta or "", "finish_reason": finish}]}

        def full(cid, created, ver, text, finish, usage):
            return {"id": cid, "object": "text_completion", "created": created, "model": f"{MODEL_ID}:{ver}",
                    "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage}

        extra = {}
        if moderate:
            in_mod = moderate(p.tenant, [req.prompt], "input")
            if in_mod:
                extra["moderation"] = {"input": in_mod}
        return run(p, "completion", req, lambda e, kw: e.complete_stream(req.prompt, **kw), req.prompt, chunk, full,
                   extra)

    @app.post("/v1/feedback", tags=["llm"])
    def llm_feedback(body: LLMFeedbackIn, p=Depends(tenant_any)):
        c = db.get_completion(body.completion_id)
        if not c or c["tenant"] != p.tenant:
            raise HTTPException(404, "completion not found")
        db.add_llm_feedback(body.completion_id, p.tenant, body.rating, body.correction)
        db.add_usage(p.tenant, "llm_feedback", 1)
        t = db.get_tenant(p.tenant)
        return {"accepted": True, "used_for_training": bool(t["train_opt_in"]),
                "note": None if t["train_opt_in"] else
                "Your organisation has not opted in to training; feedback is stored for analytics only."}

    @app.get("/v1/settings", tags=["account"])
    def get_settings(p=Depends(tenant_any)):
        return {"train_opt_in": bool(db.get_tenant(p.tenant)["train_opt_in"])}

    @app.patch("/v1/settings", tags=["account"])
    def patch_settings(body: TenantSettings, p=Depends(tenant_admin)):
        db.update_tenant(p.tenant, train_opt_in=int(body.train_opt_in))
        db.audit(p.tenant, p.key_id, "settings.update", body.model_dump())
        return body

    @app.get("/v1/admin/llm", tags=["platform admin"])
    def llm_status(_=Depends(platform)):
        reg = holder.registry()
        return {"current": reg.current(), "versions": reg.versions(), "feedback": db.llm_feedback_stats(),
                "events": db.events("__llm__", 30)}

    @app.post("/v1/admin/llm/rollback", tags=["platform admin"])
    def llm_rollback(body: RollbackIn, _=Depends(platform)):
        reg = holder.registry()
        try:
            reg.set_current(body.version)
        except FileNotFoundError:
            raise HTTPException(404, "version not found")
        db.log_event("__llm__", "promote", f"{body.version} manually set as current (rollback/override)")
        return {"current": body.version}


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"
