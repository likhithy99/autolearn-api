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
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..config import PLANS

log = logging.getLogger("autolearn.llm")
MODEL_ID = "autolearn-llm"


# Measured on autolearn-pro (6 should-call / 5 should-not requests x 2 tool specs): with no hint the model often
# asks for an order number the user already gave. This hint: mean P(call | needed) 0.95, max P(call | not
# needed) 0.01; single-tool "where is my order A-7731?" 0.30 -> 0.84 vs the hint without the example.
TOOL_HINT = ("When one of the functions can answer the request with the information the user already gave, call it "
             "right away instead of asking. If a required argument is missing, ask the user for it. Otherwise answer "
             "normally. Example: if the user writes \"where is order X-1?\" and a function looks up orders by "
             "order_id, call it with {\"order_id\": \"X-1\"}.")


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
    tools: list[dict[str, Any]] | None = Field(default=None, max_length=128)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = Field(
        default=None, description='{"type": "json_object"} or {"type": "json_schema", "json_schema": {"schema": ...}}')
    knowledge_base: str | None = Field(default=None, description="ground the answer in this knowledge base")
    knowledge_top_k: int = Field(default=4, ge=1, le=20)


class CompletionRequest(_GenParams):
    prompt: str = Field(max_length=200_000)


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: str | list[str] = Field(description="text or list of texts (max 2048)")
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"


class LLMFeedbackIn(BaseModel):
    completion_id: str
    rating: Literal[1, -1]
    correction: str | None = Field(default=None, max_length=20_000)


class TenantSettings(BaseModel):
    train_opt_in: bool


class RollbackIn(BaseModel):
    version: str


class TuningExample(BaseModel):
    messages: list[ChatMessage] = Field(min_length=2, max_length=100)


class TuningExamplesIn(BaseModel):
    examples: list[TuningExample] = Field(min_length=1, max_length=1000)


class CustomRollbackIn(BaseModel):
    version: str | None = Field(default=None, description="adapter version to serve; null = the base model")


class EngineHolder:
    """Serves AutoLearn's own model (CURRENT published version, hot-swapped on promotion) plus any open-weight
    models registered under open_models_dir. Engines load lazily on first use."""

    def __init__(self, llm_dir: Path | None, device: str | None, open_dir: Path | None = None,
                 batch_size: int | None = None):
        self.llm_dir, self.device, self.open_dir, self.batch_size = llm_dir, device, open_dir, batch_size
        self._engine = None
        self._open: dict = {}
        self._lock = threading.Lock()

    def open_catalog(self) -> dict:
        from .open_models import catalog
        return catalog(self.open_dir)

    def resolve(self, model: str | None):
        """Engine for a requested model id ('autolearn-llm', 'autolearn-llm:v3', 'autolearn-pro', ...)."""
        name = (model or MODEL_ID).split(":")[0]
        if name == MODEL_ID:
            return self.get()
        cat = self.open_catalog()
        if name not in cat:
            avail = ([MODEL_ID] if self.llm_dir else []) + sorted(cat)
            raise HTTPException(404, {"error": {"type": "invalid_request_error", "code": "model_not_found",
                                                "message": f"The model '{model}' does not exist. "
                                                           f"Available: {', '.join(avail)}"}})
        with self._lock:
            if name not in self._open:
                from .engine import Engine
                log.info("loading open model %s from %s", name, cat[name]["path"])
                eng = Engine.open_model(cat[name]["path"], self.device)
                n = self.batch_size if self.batch_size is not None else (8 if eng.device.type == "cuda" else 0)
                if n > 1:  # continuous batching: concurrent requests share forward passes
                    eng.enable_batching(max_batch=n, max_len=min(4096, eng.config.block_size))
                self._open[name] = eng
            return self._open[name]

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
    open_dir = getattr(settings, "open_models_dir", None) or settings.data_dir.parent / "models" / "open"
    holder = EngineHolder(settings.llm_dir, settings.llm_device, open_dir, getattr(settings, "llm_batch_size", None))
    app.state.llm = holder

    def json_format(req) -> str | None:
        rf = getattr(req, "response_format", None) or {}
        kind = rf.get("type", "text")
        if kind not in ("text", "json_object", "json_schema"):
            raise HTTPException(422, "response_format.type must be text, json_object or json_schema")
        return None if kind == "text" else kind

    def check_tokens(p, needed: int):
        limit = PLANS[p.plan].get("llm_tokens_per_month")
        if limit is not None and db.usage(p.tenant).get("llm_tokens", 0) + needed > limit:
            raise HTTPException(402, f"monthly LLM token quota of {limit:,} exceeded for plan '{p.plan}'")

    def gen_kwargs(req: _GenParams):
        stop = [req.stop] if isinstance(req.stop, str) else (req.stop or None)
        kw = dict(max_tokens=req.max_completion_tokens or req.max_tokens or 256, temperature=req.temperature,
                  top_p=req.top_p, top_k=req.top_k, repetition_penalty=req.repetition_penalty, stop=stop,
                  seed=req.seed)
        if json_format(req):
            kw["json_mode"] = True
        return kw

    store = None
    if holder.open_dir is not None:
        from .custom import AdapterStore
        store = AdapterStore(settings.data_dir / "adapters")
    app.state.adapters = store

    def adapter_for(p, engine):
        """(label, adapter state) for this tenant's custom version of an open model, if it has one."""
        if store is None or not engine.supports_adapters or not PLANS[p.plan].get("self_training"):
            return engine.version, None
        version, state = store.active(p.tenant, engine.model_id)
        return (f"custom-{version}", state) if state is not None else (engine.version, None)

    def record(p, cid, engine, kind, request, text, final):
        pt, ct = final.get("prompt_tokens", 0), final.get("completion_tokens", 0)
        db.add_completion(cid, p.tenant, p.key_id, engine.version, kind, request, text, pt, ct, engine.model_id)
        db.add_usage(p.tenant, "llm_tokens", pt + ct)
        db.add_usage(p.tenant, "llm_requests", 1)
        metrics.inc("autolearn_llm_tokens_total", {"type": "prompt"}, pt)
        metrics.inc("autolearn_llm_tokens_total", {"type": "completion"}, ct)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

    def run(p, kind: str, req, stream_factory, request_payload, chunk_builder, full_builder, extra=None,
            tool_chunks=None):
        extra = extra or {}
        served = holder.resolve(req.model)
        label, adapter = adapter_for(p, served)
        kw = gen_kwargs(req)
        if served.supports_adapters:
            kw["adapter"] = adapter
        engine = SimpleNamespace(model_id=served.model_id, version=label)  # what the response reports
        _stream = stream_factory
        stream_factory = lambda _e, kw: _stream(served, kw)  # noqa: E731
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
            return {**full_builder(cid, created, engine, text, finish, usage), **extra}

        include_usage = bool((req.stream_options or {}).get("include_usage"))

        def sse():
            text, final = "", {"finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0}
            try:
                if kind == "chat":
                    yield _sse({**chunk_builder(cid, created, engine, {"role": "assistant", "content": ""},
                                                None), **extra})
                for ev in stream_factory(engine, kw):
                    if "delta" in ev:
                        text += ev["delta"]
                        if tool_chunks is None:  # with tools the reply is parsed first, then sent
                            yield _sse(chunk_builder(cid, created, engine, ev["delta"], None))
                    else:
                        final = ev
                finish = final["finish_reason"]
                if tool_chunks is not None:
                    deltas, finish = tool_chunks(text, finish)
                    for d in deltas:
                        yield _sse(chunk_builder(cid, created, engine, d, None))
                yield _sse(chunk_builder(cid, created, engine, None, finish))
            finally:  # metered even if the client disconnects mid-stream
                usage = record(p, cid, engine, kind, request_payload, text, final)
                metrics.observe("autolearn_llm_request_seconds", time.perf_counter() - t0)
            if include_usage:
                yield _sse({"id": cid, "object": "chat.completion.chunk" if kind == "chat" else "text_completion",
                            "created": created, "model": f"{engine.model_id}:{engine.version}", "choices": [],
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
                         "version": cur, "description": "AutoLearn's own model, trained from scratch and "
                                                        "continually improved from customer feedback"})
        emb = app.state.get_embedder() if hasattr(app.state, "get_embedder") else None
        if emb is not None:
            data.append({"id": emb.model_id, "object": "model", "created": int(time.time()), "owned_by": "autolearn",
                         "type": "embedding", "dimensions": emb.dim,
                         "description": "Text embeddings for search, clustering and recommendations"})
        for mid, m in holder.open_catalog().items():
            data.append({"id": mid, "object": "model", "created": int(time.time()), "owned_by": "autolearn",
                         "version": m.get("repo", "").split("/")[-1].lower(), "description": m.get("description"),
                         "context_window": m.get("context")})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions", tags=["llm"])
    def chat_completions(req: ChatRequest, p=Depends(tenant_any)):
        served = holder.resolve(req.model)
        tools = req.tools if req.tools and req.tool_choice != "none" else None
        uses_tools = bool(tools) or any(m.role == "tool" or getattr(m, "tool_calls", None) for m in req.messages)
        if uses_tools and not served.supports_tools:
            raise HTTPException(400, f"model '{served.model_id}' does not support tools; use one of: "
                                     + ", ".join(m for m, e in holder.open_catalog().items()))
        if tools:
            for t in tools:
                f = t.get("function") if isinstance(t, dict) else None
                if t.get("type") != "function" or not isinstance(f, dict) or not f.get("name"):
                    raise HTTPException(422, "each tool must be {'type': 'function', 'function': {'name': ...}}")
        fmt = json_format(req)
        if fmt and tools:
            raise HTTPException(400, "response_format JSON cannot be combined with tools")
        messages = []
        for m in req.messages:
            if m.role == "tool" and not uses_tools:
                continue
            msg = {"role": "system" if m.role == "developer" else m.role, "content": m.content}
            if getattr(m, "tool_calls", None):
                calls = m.tool_calls
                if not isinstance(calls, list) or not all(
                        isinstance(c, dict) and isinstance(c.get("function", c), dict)
                        and isinstance(c.get("function", c).get("name"), str)
                        and isinstance(c.get("function", c).get("arguments", "{}"), (str, dict)) for c in calls):
                    raise HTTPException(422, "tool_calls must be a list of {'function': {'name': str, 'arguments': ...}}")
                msg["tool_calls"] = calls
            messages.append(msg)
        if tools:  # steer the model in the system prompt (measurably raises correct tool use on small models)
            want = req.tool_choice["function"]["name"] if isinstance(req.tool_choice, dict) else None
            rule = (f"You must call the function {want}." if want else
                    "You must call one of the functions." if req.tool_choice == "required" else TOOL_HINT)
            if messages[0]["role"] == "system":
                messages[0] = {**messages[0], "content": f"{messages[0]['content']}\n\n{rule}"}
            else:
                messages.insert(0, {"role": "system", "content": f"{served.tok.system_prompt or ''}\n\n{rule}".strip()})
        if fmt:  # output is constrained to valid JSON; tell the model what to put in it
            rule = "Respond only with a single JSON object."
            if fmt == "json_schema":
                schema = (req.response_format.get("json_schema") or {}).get("schema")
                if not isinstance(schema, dict):
                    raise HTTPException(422, "json_schema.schema must be a JSON Schema object")
                rule += " It must match this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
            if messages[0]["role"] == "system":
                messages[0] = {**messages[0], "content": f"{messages[0]['content']}\n\n{rule}"}
            else:
                sysp = getattr(served.tok, "system_prompt", None) or ""
                messages.insert(0, {"role": "system", "content": f"{sysp}\n\n{rule}".strip()})
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
            # Sources go right next to the question (small models follow them far more reliably than when they
            # sit in the system prompt), and sampling is near-greedy unless the client chose a temperature.
            last = max(i for i, m in enumerate(messages) if m["role"] == "user") if any(
                m["role"] == "user" for m in messages) else None
            if last is not None and isinstance(messages[last]["content"], str):
                messages[last] = {**messages[last], "content": f"{ground}\n\nQuestion: {messages[last]['content']}"}
            else:
                messages = [{"role": "system", "content": ground}] + messages
            if "temperature" not in req.model_fields_set:
                req.temperature = 0.2
            extra["citations"] = [{k: v for k, v in x.items() if k != "score"} | {"score": x["score"]}
                                  for x in passages]

        def chunk(cid, created, eng, delta, finish):
            d = {} if delta is None else (delta if isinstance(delta, dict) else {"content": delta})
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": f"{eng.model_id}:{eng.version}", "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}

        def split_tools(text, finish):
            from .open_models import parse_tool_calls
            content, calls = parse_tool_calls(text, tools)
            return content, calls, ("tool_calls" if calls else finish)

        def full(cid, created, eng, text, finish, usage):
            msg = {"role": "assistant", "content": text}
            if tools:
                content, calls, finish = split_tools(text, finish)
                msg = {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}
            return {"id": cid, "object": "chat.completion", "created": created, "model": f"{eng.model_id}:{eng.version}",
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}

        def tool_chunks(text, finish):
            content, calls, finish = split_tools(text, finish)
            deltas = [{"content": content}] if content else []
            if calls:
                deltas.append({"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]})
            return deltas, finish

        stream = (lambda e, kw: e.chat_stream(messages, tools=tools, **kw)) if uses_tools else \
            (lambda e, kw: e.chat_stream(messages, **kw))
        return run(p, "chat", req, stream, messages, chunk, full, extra, tool_chunks if tools else None)

    @app.post("/v1/embeddings", tags=["llm"])
    def embeddings(req: EmbeddingRequest, p=Depends(tenant_any)):
        emb = app.state.get_embedder() if hasattr(app.state, "get_embedder") else None
        if emb is None:
            raise HTTPException(503, "no embedding model is installed on this server")
        if req.model and req.model.split(":")[0] != emb.model_id:
            raise HTTPException(404, {"error": {"type": "invalid_request_error", "code": "model_not_found",
                                                "message": f"The model '{req.model}' does not exist. "
                                                           f"Available: {emb.model_id}"}})
        texts = [req.input] if isinstance(req.input, str) else req.input
        if not texts or len(texts) > 2048 or any(not isinstance(t, str) or not t or len(t) > 50_000 for t in texts):
            raise HTTPException(422, "input must be 1-2048 non-empty strings of at most 50,000 characters")
        n = emb.count_tokens(texts)
        check_tokens(p, n)
        t0 = time.perf_counter()
        vecs = emb.encode(texts)
        db.add_usage(p.tenant, "llm_tokens", n)
        db.add_usage(p.tenant, "embedding_requests", 1)
        metrics.inc("autolearn_llm_tokens_total", {"type": "embedding"}, n)
        metrics.observe("autolearn_llm_request_seconds", time.perf_counter() - t0)
        if req.encoding_format == "base64":
            import base64
            items = [base64.b64encode(v.astype("<f4").tobytes()).decode() for v in vecs]
        else:
            items = [[round(float(x), 7) for x in v] for v in vecs]
        return {"object": "list", "model": emb.model_id, "usage": {"prompt_tokens": n, "total_tokens": n},
                "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(items)]}

    @app.post("/v1/completions", tags=["llm"])
    def completions(req: CompletionRequest, p=Depends(tenant_any)):
        def chunk(cid, created, eng, delta, finish):
            return {"id": cid, "object": "text_completion", "created": created, "model": f"{eng.model_id}:{eng.version}",
                    "choices": [{"index": 0, "text": delta or "", "finish_reason": finish}]}

        def full(cid, created, eng, text, finish, usage):
            return {"id": cid, "object": "text_completion", "created": created, "model": f"{eng.model_id}:{eng.version}",
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

    # ------------------------------------------------------------ custom models --
    def custom_model(p, model: str):
        if store is None or model not in holder.open_catalog():
            raise HTTPException(404, f"custom models are available for: {', '.join(holder.open_catalog()) or 'none'}")
        return model

    def need_plan(p):
        if not PLANS[p.plan].get("self_training"):
            raise HTTPException(402, f"custom models are not included in the '{p.plan}' plan; upgrade to pro")

    def custom_status(p, model):
        versions = store.versions(p.tenant, model)
        fb = db.unused_llm_feedback(model, p.tenant) if db.get_tenant(p.tenant)["train_opt_in"] else []
        stats = db.tuning_stats(p.tenant, model)
        return {"model": model, "current_version": store.current(p.tenant, model),
                "pending_examples": stats["pending"] or 0, "total_examples": stats["total"],
                "pending_feedback": len(fb), "training_requested": store.training_requested(p.tenant, model),
                "versions": [{k: v.get(k) for k in ("version", "promoted", "eval", "data", "created_at")}
                             for v in versions]}

    @app.get("/v1/custom-models", tags=["custom models"])
    def list_custom(p=Depends(tenant_any)):
        if store is None:
            return []
        return [custom_status(p, m) for m in holder.open_catalog()]

    @app.get("/v1/custom-models/{model}", tags=["custom models"])
    def get_custom(model: str, p=Depends(tenant_any)):
        return custom_status(p, custom_model(p, model))

    @app.post("/v1/custom-models/{model}/examples", tags=["custom models"], status_code=201)
    def add_examples(model: str, body: TuningExamplesIn, p=Depends(tenant_admin)):
        custom_model(p, model)
        need_plan(p)
        from .custom import validate_conversation
        convs = []
        for i, ex in enumerate(body.examples):
            msgs = [{"role": m.role, "content": m.content} for m in ex.messages]
            err = validate_conversation(msgs)
            if err:
                raise HTTPException(422, f"example {i}: {err}")
            convs.append(msgs)
        ids = db.add_tuning_examples(p.tenant, model, convs)
        db.audit(p.tenant, p.key_id, "custom.examples", {"model": model, "count": len(ids)})
        return {"model": model, "added": len(ids), **{k: v for k, v in custom_status(p, model).items()
                                                      if k in ("pending_examples", "total_examples")}}

    @app.post("/v1/custom-models/{model}/train", tags=["custom models"], status_code=202)
    def train_custom_now(model: str, p=Depends(tenant_admin)):
        custom_model(p, model)
        need_plan(p)
        st = custom_status(p, model)
        if not st["pending_examples"] and not st["pending_feedback"]:
            raise HTTPException(409, "no new examples or feedback to train on")
        store.request_training(p.tenant, model)
        db.audit(p.tenant, p.key_id, "custom.train", {"model": model})
        return {"model": model, "status": "queued",
                "note": "The new version goes live only if it beats the current one on your held-out examples."}

    @app.post("/v1/custom-models/{model}/rollback", tags=["custom models"])
    def rollback_custom(model: str, body: CustomRollbackIn, p=Depends(tenant_admin)):
        custom_model(p, model)
        try:
            store.set_current(p.tenant, model, body.version)
        except FileNotFoundError:
            raise HTTPException(404, "version not found")
        db.audit(p.tenant, p.key_id, "custom.rollback", {"model": model, "version": body.version})
        return {"model": model, "current_version": body.version}

    @app.delete("/v1/custom-models/{model}", tags=["custom models"])
    def delete_custom(model: str, p=Depends(tenant_admin)):
        custom_model(p, model)
        store.delete(p.tenant, model)
        n = db.delete_tuning_examples(p.tenant, model)
        db.audit(p.tenant, p.key_id, "custom.delete", {"model": model, "examples": n})
        return {"model": model, "deleted": True, "examples_deleted": n}

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
