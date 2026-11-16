"""Live product demo against a running AutoLearn server, using the official OpenAI SDK.

    # terminal 1
    AUTOLEARN_ADMIN_KEY=demo-admin AUTOLEARN_LLM_DIR=models/llm AUTOLEARN_OPEN_MODELS_DIR=models/open \\
        .venv/bin/uvicorn autolearn.main:app --port 8000
    # terminal 2
    .venv/bin/python examples/demo.py --base http://localhost:8000 --admin-key demo-admin

Creates a demo customer (Acme Shoes), then shows chat, answers from company documents with citations,
function calling, guaranteed-JSON extraction, embeddings, moderation and usage metering.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx
from openai import OpenAI

POLICIES = [
    {"id": "returns", "title": "Returns policy",
     "text": "Acme Shoes accepts returns within 45 days of delivery. Items must be unworn. Refunds go to the "
             "original payment method within 5 business days. Sale items are final sale and cannot be returned."},
    {"id": "shipping", "title": "Shipping",
     "text": "Standard shipping is free on orders over $60 and takes 3-5 business days. Express shipping costs $15 "
             "and arrives in 1-2 business days."},
    {"id": "care", "title": "Shoe care",
     "text": "Clean leather with a damp cloth and condition it monthly. Keep suede away from water; if it gets wet, "
             "let it dry naturally away from heat, then brush it."},
]


def title(t):
    print(f"\n\033[1m── {t} " + "─" * max(0, 70 - len(t)) + "\033[0m")


def timed(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--admin-key", required=True, help="platform admin key (AUTOLEARN_ADMIN_KEY)")
    ap.add_argument("--model", default="autolearn-pro")
    a = ap.parse_args()

    t = httpx.post(f"{a.base}/v1/admin/tenants", headers={"X-API-Key": a.admin_key},
                   json={"name": "Acme Shoes (demo)", "plan": "pro"}, timeout=30).raise_for_status().json()
    key = t["admin_key"]["api_key"]
    h = {"X-API-Key": key}
    client = OpenAI(base_url=f"{a.base}/v1", api_key=key, timeout=900)
    print(f"Demo customer created: {t['name']} ({t['tenant_id']}), plan {t['plan']}")
    print("Models:", ", ".join(m.id for m in client.models.list().data))

    title("1. Chat (official OpenAI SDK, streaming)")
    print("Q: Write a two-sentence welcome message for new customers of a shoe store.\nA: ", end="", flush=True)
    for ch in client.chat.completions.create(model=a.model, stream=True, max_tokens=80, messages=[
            {"role": "user", "content": "Write a two-sentence welcome message for new customers of a shoe store."}]):
        if ch.choices:
            print(ch.choices[0].delta.content or "", end="", flush=True)
    print()

    title("2. Answers from the company's own documents, with citations")
    httpx.post(f"{a.base}/v1/knowledge/policies/documents", headers=h, json={"documents": POLICIES},
               timeout=120).raise_for_status()
    for q in ["How do I get my money back?", "My suede shoes got wet, what should I do?"]:
        r, dt = timed(lambda: client.chat.completions.create(
            model=a.model, max_tokens=90, messages=[{"role": "user", "content": q}],
            extra_body={"knowledge_base": "policies"}))
        cites = ", ".join(c["title"] for c in r.model_extra.get("citations", [])[:2])
        print(f"Q: {q}\nA: {r.choices[0].message.content.strip()}\n   sources: {cites}  ({dt:.0f}s)")

    title("3. Function calling: the model decides to call your API")
    tools = [{"type": "function", "function": {
        "name": "get_order_status", "description": "Get the shipping status of a customer order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]
    msgs = [{"role": "user", "content": "Hi, where is my order A-7731?"}]
    r = client.chat.completions.create(model=a.model, messages=msgs, tools=tools, temperature=0, max_tokens=80)
    m = r.choices[0].message
    if m.tool_calls:
        call = m.tool_calls[0]
        print(f"Model calls: {call.function.name}({call.function.arguments})")
        result = {"order_id": "A-7731", "status": "shipped", "carrier": "UPS", "eta": "Thursday"}
        print(f"Your system returns: {json.dumps(result)}")
        msgs += [m.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}]
        r2 = client.chat.completions.create(model=a.model, messages=msgs, tools=tools, temperature=0, max_tokens=80)
        print(f"Model answers the customer: {r2.choices[0].message.content}")
    else:
        print("Model answered directly:", m.content)

    title("4. Structured extraction with guaranteed-valid JSON")
    email = ("Hi, this is Maria Lopez. I'd like 2 pairs of the Trail Runner in size 8 and one Classic Loafer "
             "size 9, shipped to 12 Elm St, Denver.")
    schema = {"type": "object", "required": ["customer_name", "items", "shipping_address"], "properties": {
        "customer_name": {"type": "string"}, "shipping_address": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "product": {"type": "string"}, "size": {"type": "number"}, "quantity": {"type": "integer"}}}}}}
    r = client.chat.completions.create(model=a.model, temperature=0, max_tokens=200,
                                       messages=[{"role": "user", "content": "Extract the order:\n\n" + email}],
                                       response_format={"type": "json_schema",
                                                        "json_schema": {"name": "order", "schema": schema}})
    print(f"Email: {email}\nJSON:  {json.dumps(json.loads(r.choices[0].message.content))}")

    title("5. Embeddings (semantic similarity)")
    texts = ["refund policy", "how do I get my money back", "express delivery price"]
    e = client.embeddings.create(model="autolearn-embed", input=texts)
    v = [d.embedding for d in e.data]
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))  # noqa: E731 (vectors are normalised)
    print(f"similarity('{texts[0]}', '{texts[1]}') = {dot(v[0], v[1]):.2f}")
    print(f"similarity('{texts[0]}', '{texts[2]}') = {dot(v[0], v[2]):.2f}")

    title("6. Moderation")
    res = httpx.post(f"{a.base}/v1/moderations", headers=h, json={"input": [
        "Thanks for the quick delivery!", "I will find you and hurt you"]}, timeout=30)
    if res.status_code == 200:
        for txt, out in zip(["Thanks for the quick delivery!", "I will find you and hurt you"], res.json()["results"]):
            cats = [c for c, f in out["categories"].items() if f]
            print(f"{'FLAGGED' if out['flagged'] else 'ok':8s} {txt!r} {cats if cats else ''}")
    else:
        print("moderation model not installed on this server")

    title("7. Usage metering (what the customer is billed for)")
    me = httpx.get(f"{a.base}/v1/me", headers=h, timeout=30).json()
    print(json.dumps(me.get("usage", me), indent=1)[:600])


if __name__ == "__main__":
    main()
