# AutoLearn API: Customer Guide

Base URL: `https://api.yourdomain.com` (replace with your provider's URL).
Every request needs your API key, in either `X-API-Key: <key>` or `Authorization: Bearer <key>`.
Interactive reference: `/docs`. Web console: `/dashboard`.

## 1. Keys

You receive an **admin key** when you sign up. Use it to create **client keys** for your apps. Client
keys can predict, chat and send feedback, but cannot change settings or data.

```bash
curl -X POST $BASE/v1/keys -H "X-API-Key: $ADMIN_KEY" -H 'Content-Type: application/json' \
     -d '{"name": "production web app", "role": "client"}'
```

Revoke a key with `DELETE /v1/keys/{id}`. Check usage and limits with `GET /v1/me`.

## 2. Chat (OpenAI-compatible)

Python, using the official OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.yourdomain.com/v1", api_key="al_cli_...")
r = client.chat.completions.create(
    model="autolearn-llm",
    messages=[{"role": "user", "content": "Summarise our refund policy"}],
    extra_body={"knowledge_base": "policies"},      # optional: answer from your documents
)
print(r.choices[0].message.content)
```

JavaScript:

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "https://api.yourdomain.com/v1", apiKey: "al_cli_..." });
const r = await client.chat.completions.create({
  model: "autolearn-llm", messages: [{ role: "user", content: "Hello" }], stream: true });
for await (const chunk of r) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
```

Supported parameters: `messages`, `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `stream`,
`stream_options.include_usage`, plus the AutoLearn extras `knowledge_base`, `knowledge_top_k`,
`top_k` and `repetition_penalty`.

### Make the model better: feedback

```bash
curl -X POST $BASE/v1/feedback -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"completion_id": "chatcmpl-...", "rating": -1, "correction": "The refund window is 60 days."}'
```

If your organisation opts in (`PATCH /v1/settings {"train_opt_in": true}`), ratings and corrections
are used to fine-tune the model. A new version goes live only if it beats the current one on
held-out examples. **Your data is never used for training unless you opt in.**

## 3. Knowledge bases (answers from your documents)

```bash
curl -X POST $BASE/v1/knowledge/policies/documents -H "X-API-Key: $ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"documents": [{"id": "refunds", "title": "Refund policy", "text": "Enterprise customers ..."}]}'

curl -X POST $BASE/v1/knowledge/policies/search -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"query": "refund window", "top_k": 3}'
```

Re-uploading a document with the same `id` replaces it. Delete one with
`DELETE /v1/knowledge/{kb}/documents/{id}`, or a whole base with `DELETE /v1/knowledge/{kb}`.
Grounded chat answers include `citations`.

## 4. Predictions on your own data (AutoML)

```python
from autolearn.client import AutoLearn
al = AutoLearn("https://api.yourdomain.com", api_key="al_adm_...")
al.create_project("churn", task="classification", target="churned")
al.upload_csv("churn", "customers.csv")        # a model is trained automatically
out = al.predict("churn", {"tenure": 3, "plan": "basic"}, explain=True)
al.feedback("churn", out["predictions"][0]["prediction_id"], actual="yes")
```

* `task` is `classification` (categories) or `regression` (numbers).
* Columns can be numbers, categories or free text; they're detected automatically.
* Send real outcomes back with `/feedback`. The service watches its live accuracy and data drift,
  retrains on its own, and switches models only when the new one is better. Check
  `GET /v1/projects/{p}/events` to see what it did and why.
* Roll back any time with `POST /v1/projects/{p}/models/{version}/promote`.

## 5. Content moderation

```bash
curl -X POST $BASE/v1/moderations -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"input": ["text to check"]}'
```

Apply it automatically to chat with `PUT /v1/settings/moderation {"moderation": "flag" | "block" | "off"}`.

## 6. Errors and limits

| Code | Meaning | What to do |
|---|---|---|
| 401 | Missing, invalid or revoked key | Check the key |
| 402 | Plan quota or limit reached | Upgrade (`POST /v1/billing/portal`) or wait for the monthly reset |
| 403 | Wrong role, or account suspended | Use an admin key; check billing |
| 409 | Model not trained yet / training in progress | Upload data or retry shortly |
| 413 | Request too large | Split into smaller batches |
| 422 | Invalid request | See `detail` in the response |
| 429 | Rate limited | Retry after the `Retry-After` seconds |
| 400 `content_policy_violation` | Blocked by your moderation setting | Rephrase, or change the setting |

Every response carries an `X-Request-ID` header. Include it when contacting support.

## 7. Data and privacy

* Your data is isolated from other customers at every layer: keys, projects, documents, models and logs.
* Deleting a project or document erases its data and models.
* Chat logs are kept for a limited retention period (30 days by default) and then deleted automatically.
* Training on your data happens only with explicit opt-in.
