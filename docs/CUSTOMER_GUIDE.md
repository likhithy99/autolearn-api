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
    model="autolearn-pro",
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
  model: "autolearn-pro", messages: [{ role: "user", content: "Hello" }], stream: true });
for await (const chunk of r) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
```

Models (`GET /v1/models`): **`autolearn-pro`** for production use, and **`autolearn-llm`**, our own model,
which is retrained from customer feedback.

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

### Embeddings

```python
v = client.embeddings.create(model="autolearn-embed", input=["refund policy", "shipping times"])
print(len(v.data[0].embedding))   # 384; vectors are normalised, so cosine similarity = dot product
```

## 3. Your own custom model (pro and enterprise plans)

Teach `autolearn-pro` your company's facts, tone and formats. Your training data shapes **only your
model**; no other customer's model is affected.

```bash
# 1. upload example conversations, each ending with the ideal answer
curl -X POST $BASE/v1/custom-models/autolearn-pro/examples -H "X-API-Key: $ADMIN_KEY" \
  -H 'Content-Type: application/json' -d '{"examples": [
    {"messages": [{"role": "user", "content": "What is your refund window?"},
                  {"role": "assistant", "content": "You can return unworn shoes within 45 days of delivery."}]}]}'

# 2. train now (it also trains automatically as examples and feedback accumulate)
curl -X POST $BASE/v1/custom-models/autolearn-pro/train -H "X-API-Key: $ADMIN_KEY"

# 3. check versions and their evaluation results
curl $BASE/v1/custom-models/autolearn-pro -H "X-API-Key: $ADMIN_KEY"
```

Keep using `model="autolearn-pro"` in your requests: once a version passes evaluation, your requests are
served by it automatically and the response's `model` shows e.g. `autolearn-pro:custom-v2`.

* **Learns from feedback too:** if you opt in (`PATCH /v1/settings {"train_opt_in": true}`), your 👍/👎 ratings
  and corrections on `autolearn-pro` answers become training data for your custom model.
* **Quality gate:** a new version goes live only if it beats the current one on held-out examples and
  doesn't lose general ability (checked on a fixed set of everyday questions).
* **Undo:** `POST /v1/custom-models/autolearn-pro/rollback {"version": "v1"}`, or `{"version": null}` to go back
  to the standard model. `DELETE /v1/custom-models/autolearn-pro` erases your adapters and examples.
* Tip: 20–200 varied examples work best; phrase questions the way your users do.

## 4. Knowledge bases (answers from your documents)

```bash
curl -X POST $BASE/v1/knowledge/policies/documents -H "X-API-Key: $ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"documents": [{"id": "refunds", "title": "Refund policy", "text": "Enterprise customers ..."}]}'

curl -X POST $BASE/v1/knowledge/policies/search -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"query": "refund window", "top_k": 3}'
```

Search understands meaning as well as keywords, so "how do I get my money back?" finds your refund policy.
Each result has `match`: `keyword`, `semantic` or `both`. Re-uploading a document with the same `id` replaces it. Delete one with
`DELETE /v1/knowledge/{kb}/documents/{id}`, or a whole base with `DELETE /v1/knowledge/{kb}`.
Grounded chat answers include `citations`. Unless you set `temperature`, grounded answers use 0.2 so they stick
closely to your documents.

## 5. Predictions on your own data (AutoML)

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

## 6. Content moderation

```bash
curl -X POST $BASE/v1/moderations -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"input": ["text to check"]}'
```

Apply it automatically to chat with `PUT /v1/settings/moderation {"moderation": "flag" | "block" | "off"}`.

## 7. Errors and limits

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

## 8. Data and privacy

* Your data is isolated from other customers at every layer: keys, projects, documents, models and logs.
* Deleting a project or document erases its data and models.
* Chat logs are kept for a limited retention period (30 days by default) and then deleted automatically.
* Training on your data happens only with explicit opt-in (or when you upload examples for your custom
  model), and only ever changes your own custom model.
