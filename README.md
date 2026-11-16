# AutoLearn

A self-hosted AI platform you can sell as an API: an **OpenAI-compatible** API (the official `openai`
SDKs work unchanged) plus AutoML, behind one set of API keys, billing plans and usage meters.

| Capability | Endpoint | Notes |
|---|---|---|
| Chat and completions | `/v1/chat/completions`, `/v1/completions` | streaming, stop/seed/temperature/top-p, per-tenant metering |
| Function calling | `tools`, `tool_choice` | OpenAI `tool_calls` format, also streamed |
| Guaranteed-valid JSON | `response_format` | `json_object` / `json_schema` via constrained decoding |
| Embeddings | `/v1/embeddings` | 384-dim, normalised; float or base64 |
| Answers from your documents (RAG) | `/v1/knowledge/*`, `knowledge_base` on chat | hybrid keyword + semantic search, citations |
| **Custom model per customer** | `/v1/custom-models/*` | LoRA adapter trained on the customer's examples and feedback; promoted only if it beats the current one |
| Learning from feedback | `/v1/feedback` | opt-in; feeds the custom models and AutoLearn's own model |
| Content moderation | `/v1/moderations` | own classifier (Civil Comments), off/flag/block per tenant |
| AutoML predictions | `/v1/projects/*` | trains, monitors drift, **retrains itself** from feedback |
| Accounts and billing | `/v1/keys`, `/v1/me`, Stripe checkout and portal, `/dashboard` | roles, quotas, rate limits, audit log |

Models: **`autolearn-pro`** (Qwen2.5-1.5B-Instruct, Apache-2.0, for production), **`autolearn-embed`**, and
**`autolearn-llm`**, a GPT-style model built entirely from scratch here (data pipeline, tokenizer, transformer,
pretraining on free GPUs, chat tuning) that keeps learning from feedback. All of them run on one inference
engine written in this repo (`autolearn/llm/`), with no `transformers` dependency.

```
                       ┌──────────────────────────── FastAPI (one service) ───────────────────────────┐
 customers ──HTTPS──▶  │ auth (API keys, roles) · tenants · plans/quotas · rate limits · usage metering │
  (X-API-Key or        │ audit log · request IDs · JSON logs · Prometheus /metrics                     │
   Bearer token)       ├───────────────────────────────┬──────────────────────────────────────────────┤
                       │ /v1/projects/* (AutoML)        │ /v1/chat (+tools, JSON) · /v1/embeddings      │
                       │  ingest → AutoML → registry    │  Engine (KV cache, streaming SSE)             │
                       │  predict → feedback            │  feedback (opt-in) → continual worker         │
                       │  background learner:           │   SFT + DPO + replay → eval gate → publish    │
                       │   drift / degradation /        │   → API hot-swaps to the new version          │
                       │   schedule → retrain → promote │                                               │
                       └───────────────────────────────┴──────────────────────────────────────────────┘
                              SQLite (WAL) · model files on disk · versioned LLM registry
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
python scripts/fetch_datasets.py                 # Adult Census + California Housing demo data
.venv/bin/pytest -q                              # 32 tests

export AUTOLEARN_ADMIN_KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
export AUTOLEARN_LLM_DIR=models/llm              # optional: enables the LLM endpoints
export AUTOLEARN_OPEN_MODELS_DIR=models/open     # optional: open-weight models (see "Model lineup")
.venv/bin/uvicorn autolearn.main:app --port 8000
# interactive API docs: http://localhost:8000/docs
```

Docker: `AUTOLEARN_ADMIN_KEY=... docker compose up --build` starts the API and the LLM
continual-learning worker.

## Onboarding a paying customer

```bash
# platform operator (you): create a tenant and choose its plan
curl -X POST localhost:8000/v1/admin/tenants -H "X-API-Key: $AUTOLEARN_ADMIN_KEY" \
     -H 'Content-Type: application/json' -d '{"name":"Acme Corp","plan":"pro"}'
# → returns tenant_id and the customer's first admin key (shown once)
```

The customer then creates their own keys (`POST /v1/keys`, role `admin` or `client`), revokes them
(`DELETE /v1/keys/{id}`), and checks `GET /v1/me`, `GET /v1/usage` and `GET /v1/audit`.

| Plan | Predictions/mo | LLM tokens/mo | Projects | Stored rows | Rate limit/key | Self-training |
|---|---|---|---|---|---|---|
| free | 10k | 200k | 2 | 50k | 60/min | no |
| pro | 1M | 50M | 20 | 2M | 1,200/min | yes |
| enterprise | ∞ | ∞ | ∞ | ∞ | 12,000/min | yes |

Edit the `PLANS` table in `autolearn/config.py` to change limits. Usage is metered per tenant per
month (`predictions`, `rows_ingested`, `trainings`, `feedback`, `llm_tokens`, `llm_requests`). Feed
it to your billing system (for example Stripe metered billing) from `GET /v1/admin/tenants`.

## Selling access (self-serve, Stripe)

`autolearn/billing.py` turns the API into a store. Card data never touches your server.

* **Pricing page:** `GET /` shows the plans with "Buy" buttons. Buying goes to Stripe Checkout.
* **Automatic onboarding:** Stripe calls `/stripe/webhook`, the server creates the tenant and its
  admin API key, and `/billing/success` shows the key to the buyer **once**.
* **Lifecycle:** plan upgrades and downgrades update limits. `past_due`/`unpaid`/`canceled` suspends
  the account, and paying again reactivates it. Webhooks are signature-verified and idempotent.
* **Self-service:** `POST /v1/billing/portal` gives the customer a Stripe portal link (update card,
  invoices, cancel).

Setup: create a Stripe account. Make one Product with a monthly Price per paid plan. Add a webhook
for `checkout.session.completed`, `customer.subscription.updated` and
`customer.subscription.deleted`. Then set:

```bash
STRIPE_SECRET_KEY=sk_live_...  STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_PRO=price_...     STRIPE_PRICE_ENTERPRISE=price_...
AUTOLEARN_PUBLIC_URL=https://api.yourdomain.com
AUTOLEARN_PRICE_LABELS="free=$0,pro=$49/mo,enterprise=Contact us"
```

Use `sk_test_...` keys and the Stripe CLI (`stripe listen --forward-to localhost:8000/stripe/webhook`)
to test end to end before going live.

## Product 1: AutoML that learns on its own

```python
from autolearn.client import AutoLearn
al = AutoLearn("http://localhost:8000", api_key="al_adm_...")
al.create_project("churn", task="classification", target="churned")
al.upload_csv("churn", "customers.csv")        # the learner trains automatically, or:
al.train("churn")
out = al.predict("churn", {"tenure": 3, "plan": "basic", "note": "billing errors"}, explain=True)
al.feedback("churn", out["predictions"][0]["prediction_id"], actual="yes")   # closes the loop
```

**How it learns by itself** (`autolearn/learner.py`)

* **Feature typing:** numeric, categorical and free-text columns are detected automatically, and
  ID-like columns are dropped.
* **Model search:** logistic/ridge regression, random forest and gradient boosting are
  cross-validated, and the winner is chosen.
* **Retrain triggers:** a background loop retrains on any of these:
  * *initial*: enough labels exist and there is no model yet
  * *scheduled*: N new labels have arrived
  * *adaptive*: **live accuracy from feedback has degraded**, or **input drift** was detected
    (numeric mean shift, unseen categories)
* **Data-window selection:** each retrain compares *all history* against *recent* data and against
  *only data since the last training*. After the world changes, stale history is dropped automatically.
* **Champion/challenger:** the challenger is scored against the current model on a **fresh holdout**
  (data neither has seen). It is promoted only if it is at least as good. Every version is kept,
  with one-call rollback (`POST /models/{v}/promote`).
* **Explanations:** permutation feature importance is reported per model (`explain=true`).
* **Self-training (optional):** high-confidence predictions become capped pseudo-labels. Real
  feedback replaces them.

**Demo on real data** (UCI Adult Census, 48,842 people). The model is trained only on people aged
35 or younger, then serves older people. Nobody calls `/train` after launch:

```
python examples/simulate_production.py --url http://localhost:8000 --admin-key $AUTOLEARN_ADMIN_KEY
round  model  live acc  monitor / learner activity
    1     v1     0.730  LEARNER: v2 ... PROMOTED
    2     v2     0.777  LEARNER: v3 gradient_boosting [recent_half] ... PROMOTED
    3     v3     0.782  drift in ['age']; LEARNER: v4 ... rejected, champion kept
    4     v3     0.815  LEARNER: v5 ... PROMOTED
```

## Product 2: an LLM built from scratch

| Stage | Module | What it does |
|---|---|---|
| Data | `autolearn/llm/data.py` | Streams **FineWeb-Edu** (1.3T tokens), filters quality, removes exact duplicates, scrubs PII (emails/phones), and writes memory-mapped token shards. Disk budget: `--tokens` |
| Tokenizer | `autolearn/llm/tokenizer.py` | Byte-level BPE (32k) trained on our corpus. Chat template; protected against control-token injection |
| Model | `autolearn/llm/model.py` | Decoder transformer: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, KV cache. Presets from 10M to 7B parameters |
| Pretraining | `autolearn/llm/train.py` | AdamW, cosine schedule, gradient accumulation and clipping, bf16 + `torch.compile` on CUDA, **multi-GPU DDP via torchrun**, time budgets, resumable checkpoints |
| Chat tuning | `train.py --stage sft` | Fine-tunes on **SmolTalk** conversations, with loss on assistant tokens only |
| Serving | `autolearn/llm/routes.py`, `engine.py` | **OpenAI-compatible** `/v1/chat/completions` (+ SSE streaming), `/v1/completions`, `/v1/models`; temperature/top-p/top-k/stop/seed; per-tenant token metering and quotas |
| Self-improvement | `autolearn/llm/continual.py` | Uses feedback (thumbs up/down + corrections) from **opted-in** customers: SFT on good answers and corrections, **DPO** on (correction vs. bad answer) pairs, and replay to prevent forgetting. Promotes a new version only if it beats the current one on held-out feedback without regressing on general chat. The API hot-swaps to it; rollback is available |

```bash
scripts/llm_pipeline.sh                    # laptop defaults: 300M tokens, 22M params, ~2 h
PRESET=small TOKENS=5B MINUTES=360 NPROC=8 scripts/llm_pipeline.sh    # 8 GPUs
```

**Free multi-week GPU training on Kaggle** (about 30 GPU-hours per week on 2× T4):

```bash
# one-time: phone-verify your Kaggle account, create an API token at https://www.kaggle.com/settings,
# save it to ~/.kaggle/access_token
nohup .venv/bin/python scripts/kaggle/campaign.py --user <kaggle-username> > logs/kaggle_campaign.log 2>&1 &
caffeinate -i -w $! &      # keep the Mac awake while the campaign runs
```

The campaign trains the ~101M-parameter `small` model on 4B fresh FineWeb-Edu tokens. It runs 12-hour
sessions that resume from checkpoints, waits out the weekly quota, runs chat fine-tuning at the end,
and publishes the result to `models/llm`. Check `logs/kaggle_campaign.log` for progress.

Customers use the official OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="al_cli_...")
r = client.chat.completions.create(model="autolearn-llm",
                                   messages=[{"role": "user", "content": "Explain photosynthesis"}])
# rate the answer so the model learns from it:
#   POST /v1/feedback {"completion_id": r.id, "rating": -1, "correction": "..."}
```

**Knowledge bases (RAG).** Customers upload their own documents (`POST /v1/knowledge/{kb}/documents`).
Retrieval is hybrid: BM25 keyword ranking via SQLite FTS5 plus embedding similarity, merged with
reciprocal rank fusion, with no extra service needed. On paraphrased questions with no shared keywords,
keyword-only search found 0/4 right answers and hybrid found 3/4. Passing
`"knowledge_base": "<kb>"` to chat completions grounds the answer in the top passages and returns
`citations`. This is how the model answers about company-specific facts it was never trained on.

**Benchmarks.** `python -m autolearn.llm.benchmarks --ckpt <path>.pt` (or `--version vN`) scores
HellaSwag, ARC-Easy and ARC-Challenge by answer log-likelihood. Random guessing is 25%; GPT-2 small
(124M) gets about 29% on HellaSwag and 44% on ARC-Easy.

> **Reality check.** Quality depends on scale, not code. A 22M-parameter model trained on a laptop
> writes loosely fluent text but is **not** a useful assistant. Matching GPT-4 or Claude needs
> roughly 100B+ parameters, 10T+ tokens and thousands of GPUs. See
> [docs/SCALING.md](docs/SCALING.md) for the size / data / compute / cost table and how to run the
> same code on a GPU cluster.

## Model lineup

| Model id | What it is | Best for |
|---|---|---|
| `autolearn-pro` | Qwen2.5-1.5B-Instruct (Apache-2.0), served on AutoLearn's own transformer engine | Production chat, drafting, code, answers from your knowledge bases |
| `autolearn-embed` | bge-small-en-v1.5 embeddings (MIT, 384 dims) on our own BERT encoder (`autolearn/llm/embeddings.py`) | `/v1/embeddings`, semantic search; powers hybrid knowledge-base retrieval |
| `autolearn-llm` | AutoLearn's own model, trained from scratch here and continually improved from opted-in feedback | Experimentation; grows with the training campaigns |

Open-weight models run on the same engine as ours: Qwen2 and Llama-family weights are mapped straight onto
`autolearn/llm/model.py` (`autolearn/llm/open_models.py`). A test checks the result against an independent
reference implementation, and the chat format against the model's official template. Add another model:

```bash
python -m autolearn.llm.open_models add Qwen/Qwen2.5-7B-Instruct --name autolearn-max   # needs ~16GB GPU
python -m autolearn.llm.open_models list
python -m autolearn.llm.embeddings add BAAI/bge-small-en-v1.5 --name autolearn-embed   # embeddings
```

**Custom models per customer** (`autolearn/llm/custom.py`, `lora.py`): each company gets its own LoRA
adapter on `autolearn-pro`, trained from the examples it uploads (`POST /v1/custom-models/{model}/examples`)
and its opted-in feedback. A new version is promoted only if it beats the current one on the company's
held-out examples without regressing on general prompts, so the model learns on its own. One base model in
memory serves every company: the right adapter is applied per request. Worker:
`python -m autolearn.llm.custom --data-dir data --open-dir models/open` (also a docker-compose service).

Only add models whose licence allows commercial use, and keep their `LICENSE` file (it is downloaded with
the weights). Customers pick a model with the standard `model` field; `GET /v1/models` lists them.

## API reference (summary)

Full interactive docs are at `/docs`. Authenticate with `X-API-Key: <key>` or
`Authorization: Bearer <key>`.

| Area | Endpoints |
|---|---|
| Ops | `GET /health`, `GET /ready`, `GET /metrics` (platform key) |
| Platform admin | `POST/GET /v1/admin/tenants`, `PATCH /v1/admin/tenants/{id}` (plan, suspend), `GET /v1/admin/tenants/{id}/usage`, `GET /v1/admin/llm`, `POST /v1/admin/llm/rollback` |
| Account | `GET /v1/me`, `GET /v1/usage`, `GET/POST /v1/keys`, `DELETE /v1/keys/{id}`, `GET /v1/audit`, `GET/PATCH /v1/settings` (training opt-in) |
| AutoML | `POST/GET /v1/projects`, `GET/PATCH/DELETE /v1/projects/{p}`, `POST /data`, `POST /data/csv`, `POST /train`, `GET /models`, `GET /models/{v}`, `POST /models/{v}/promote`, `POST /predict`, `POST /feedback`, `GET /monitor`, `GET /events` |
| LLM | `GET /v1/models`, `POST /v1/chat/completions` (+ `knowledge_base` for grounded answers with citations), `POST /v1/completions`, `POST /v1/feedback` |
| Knowledge bases (RAG) | `POST/GET /v1/knowledge/{kb}/documents`, `DELETE /v1/knowledge/{kb}/documents/{id}`, `DELETE /v1/knowledge/{kb}`, `POST /v1/knowledge/{kb}/search` |
| Billing | `GET /` (pricing), `GET /signup?plan=`, `POST /stripe/webhook`, `GET /billing/success`, `POST /v1/billing/portal` |

Errors: `401` bad key · `402` plan quota/limit · `403` wrong role or suspended · `404` · `409`
not trained or busy · `413` body too large · `422` validation · `429` rate limited (`Retry-After`).

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `AUTOLEARN_ADMIN_KEY` | auto-generated into `data/admin_key.txt` | platform operator key |
| `AUTOLEARN_DATA_DIR` | `./data` | SQLite DB + AutoML models |
| `AUTOLEARN_LEARN_INTERVAL` | `15` | seconds between autonomous learner checks |
| `AUTOLEARN_LLM_DIR` | unset (LLM off) | versioned LLM registry |
| `AUTOLEARN_OPEN_MODELS_DIR` | `<data dir>/../models/open` | open-weight models served next to ours |
| `AUTOLEARN_LLM_BATCH` | 8 on CUDA, off elsewhere | continuous-batching slots per model (concurrent requests share each forward pass) |
| `AUTOLEARN_LLM_DEVICE` | auto (cuda > mps > cpu) | inference device |
| `AUTOLEARN_LLM_LOG_RETENTION_DAYS` | `30` | completion-log retention, enforced by the continual worker |
| `AUTOLEARN_MAX_BODY_BYTES` | 20 MB | request size limit |
| `AUTOLEARN_CORS_ORIGINS` | none | comma-separated origins for browser clients |

Per-project AutoML settings (`PATCH /v1/projects/{p}` with `{"config": {...}}`): `min_rows`,
`retrain_every`, `adaptive_min_new`, `min_improvement`, `degrade_tolerance`, `drift_threshold`,
`cv_folds`, `max_train_rows`, `self_training`, `pseudo_label_threshold`, `max_pseudo_fraction`.

## Going live

See [docs/DEPLOY.md](docs/DEPLOY.md) for free hosting (Oracle Cloud Always Free VM with automatic
HTTPS, or a Cloudflare Tunnel from your Mac) and the launch checklist.

## Production checklist (things code can't do for you)

* Put it behind TLS (a load balancer or Caddy/nginx). Never expose plain HTTP.
* Run **one worker per container** (models and the learner live in-process). Scale by running more
  containers per tenant shard. For high traffic, move SQLite to Postgres and model files to object storage.
* Back up `AUTOLEARN_DATA_DIR` and the LLM registry.
* Connect `/metrics` to Prometheus/Grafana and alert on 5xx rate, latency and failed trainings.
* For the LLM: add a moderation layer, run benchmark evals in the promotion gate, and scale up
  model, data and compute (see docs/SCALING.md).
* Legal: terms of service, privacy policy, DPA, dataset attribution (FineWeb-Edu ODC-By, SmolTalk
  Apache-2.0, UCI Adult CC BY 4.0).
