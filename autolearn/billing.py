"""Self-serve sales: pricing page -> Stripe Checkout -> automatic tenant + API key -> billing portal.

Card data never touches this server (Stripe-hosted checkout and portal).

Flow
  GET  /                      pricing page with "Buy" buttons
  GET  /signup?plan=pro       creates a Stripe Checkout session and redirects to it
  POST /stripe/webhook        Stripe -> us: provision on payment, change plan, suspend on cancel/non-payment
  GET  /billing/success       shows the customer's API key exactly once
  POST /v1/billing/portal     tenant admin -> Stripe customer portal (update card, invoices, cancel)

Setup (Stripe dashboard):
  1. Create a Product per paid plan with a recurring Price; set STRIPE_PRICE_PRO / STRIPE_PRICE_ENTERPRISE.
  2. Add a webhook endpoint https://<your-domain>/stripe/webhook for events:
       checkout.session.completed, customer.subscription.updated, customer.subscription.deleted
     and set STRIPE_WEBHOOK_SECRET. Set STRIPE_SECRET_KEY and AUTOLEARN_PUBLIC_URL.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import logging
import secrets
import time

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .config import PLANS

log = logging.getLogger("autolearn.billing")
STRIPE_API = "https://api.stripe.com/v1"
ACTIVE_STATUSES = {"active", "trialing"}


def verify_stripe_signature(payload: bytes, header: str, secret: str, tolerance: int = 300) -> dict:
    """Stripe webhook signature scheme: HMAC-SHA256 over '<timestamp>.<raw body>'."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
        ts = int(parts["t"])
        sigs = [v for k, v in (p.split("=", 1) for p in header.split(",") if "=" in p) if k == "v1"]
    except (KeyError, ValueError):
        raise HTTPException(400, "malformed Stripe-Signature header")
    if abs(time.time() - ts) > tolerance:
        raise HTTPException(400, "webhook timestamp outside tolerance")
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in sigs):
        raise HTTPException(400, "invalid webhook signature")
    return json.loads(payload)


class Stripe:
    def __init__(self, secret_key: str):
        self.key = secret_key

    def _req(self, method, path, data=None):
        r = httpx.request(method, f"{STRIPE_API}{path}", data=data, auth=(self.key, ""), timeout=20)
        if r.status_code >= 400:
            log.error("stripe %s %s -> %s %s", method, path, r.status_code, r.text[:500])
            raise HTTPException(502, "payment provider error")
        return r.json()

    def create_checkout(self, price_id, plan, success_url, cancel_url):
        return self._req("POST", "/checkout/sessions", {
            "mode": "subscription",
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "allow_promotion_codes": "true",
            "billing_address_collection": "required",
            "metadata[plan]": plan,
            "subscription_data[metadata][plan]": plan,
        })

    def get_checkout(self, session_id):
        return self._req("GET", f"/checkout/sessions/{session_id}")

    def portal(self, customer_id, return_url):
        return self._req("POST", "/billing_portal/sessions", {"customer": customer_id, "return_url": return_url})


def register_billing_routes(app, settings, db, new_key, tenant_admin):
    stripe = Stripe(settings.stripe_secret_key) if settings.stripe_secret_key else None
    prices = settings.stripe_prices or {}
    price_to_plan = {v: k for k, v in prices.items()}
    labels = settings.price_labels or {}

    def provision(session: dict) -> str | None:
        """Idempotently create the tenant + first admin key for a paid checkout session."""
        if session.get("payment_status") not in ("paid", "no_payment_required"):
            return None
        sub = session.get("subscription")
        existing = db.tenant_by("stripe_subscription_id", sub) if sub else None
        if existing:
            return existing["id"]
        details = session.get("customer_details") or {}
        plan = (session.get("metadata") or {}).get("plan", "pro")
        plan = plan if plan in PLANS else "pro"
        tid = "t_" + secrets.token_hex(6)
        db.create_tenant(tid, details.get("name") or details.get("email") or "Customer", plan)
        db.update_tenant(tid, email=details.get("email"), stripe_customer_id=session.get("customer"),
                         stripe_subscription_id=sub)
        key = new_key(tid, "initial admin key", "admin")
        db.add_reveal(session["id"], tid, key["api_key"])
        db.audit(tid, "stripe", "tenant.provisioned", {"plan": plan, "session": session["id"]})
        log.info("provisioned tenant %s on plan %s", tid, plan)
        return tid

    # ------------------------------------------------------------ pricing --
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def pricing():
        cards = []
        for plan, lim in PLANS.items():
            fmt = lambda v: "Unlimited" if v is None else f"{v:,}"
            if plan in prices:
                cta = f'<a class="btn" href="/signup?plan={plan}">Get {plan.title()}</a>'
            elif plan == "enterprise":
                cta = '<a class="btn ghost" href="mailto:sales@example.com">Contact sales</a>'
            else:
                cta = '<span class="muted">Ask us for a key</span>'
            cards.append(f"""
            <div class="card"><h2>{plan.title()}</h2><p class="price">{html.escape(labels.get(plan, ""))}</p>
            <ul><li>{fmt(lim['predictions_per_month'])} predictions / month</li>
            <li>{fmt(lim.get('llm_tokens_per_month'))} LLM tokens / month</li>
            <li>{fmt(lim['max_projects'])} projects</li><li>{fmt(lim['max_rows'])} stored rows</li>
            <li>{lim['rate_limit_per_min']:,} requests / min per key</li>
            <li>{'Self-training models' if lim['self_training'] else 'Standard retraining'}</li></ul>{cta}</div>""")
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AutoLearn API</title><style>
:root{{--bg:#fafafa;--fg:#111;--card:#fff;--muted:#666;--accent:#2952e3;--line:#e5e5e5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--card:#1b1b1b;--muted:#999;--line:#333}}}}
body{{margin:0;font:16px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1000px;margin:0 auto;padding:48px 16px}}h1{{font-size:2rem;margin:0 0 8px}}
.sub{{color:var(--muted);margin:0 0 32px}}.grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px}}
.card h2{{margin:0}}.price{{font-size:1.6rem;font-weight:600;margin:8px 0}}ul{{padding-left:18px;color:var(--muted)}}
.btn{{display:inline-block;background:var(--accent);color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none}}
.btn.ghost{{background:transparent;color:var(--accent);border:1px solid var(--accent)}}.muted{{color:var(--muted)}}
</style></head><body><main><h1>AutoLearn API</h1>
<p class="sub">Self-improving machine learning and language models over a simple API.
<a href="/docs">API docs</a></p><div class="grid">{''.join(cards)}</div></main></body></html>"""

    # ------------------------------------------------------------- checkout --
    @app.get("/signup", include_in_schema=False)
    def signup(plan: str = "pro"):
        if not stripe:
            raise HTTPException(503, "online payments are not configured")
        if plan not in prices:
            raise HTTPException(404, "unknown or non-purchasable plan")
        s = stripe.create_checkout(prices[plan], plan,
                                   f"{settings.public_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
                                   f"{settings.public_url}/")
        return RedirectResponse(s["url"], status_code=303)

    @app.get("/billing/success", response_class=HTMLResponse, include_in_schema=False)
    def success(session_id: str):
        if not stripe:
            raise HTTPException(503, "online payments are not configured")
        reveal = db.take_reveal(session_id)
        if reveal is None:  # webhook not arrived yet: verify with Stripe directly and provision
            session = stripe.get_checkout(session_id)
            if not provision(session):
                return HTMLResponse("<p>Payment not completed yet. Refresh this page in a few seconds.</p>", 402)
            reveal = db.take_reveal(session_id)
        if not reveal or not reveal["api_key"]:
            return HTMLResponse("<p>This key was already shown. Create new keys with <code>POST /v1/keys</code> "
                                "using an admin key, or contact support.</p>")
        key = html.escape(reveal["api_key"])
        return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Your API key</title><style>body{{font:16px/1.5 system-ui;
max-width:720px;margin:48px auto;padding:0 16px}}code,pre{{background:#f2f2f2;color:#111;padding:2px 6px;border-radius:6px;word-break:break-all}}
pre{{padding:12px;overflow:auto}}</style></head><body><h1>Welcome to AutoLearn</h1>
<p><strong>Copy your admin API key now. It will not be shown again.</strong></p><pre>{key}</pre>
<p>Quick start:</p><pre>curl {settings.public_url}/v1/me -H "X-API-Key: {key}"</pre>
<p>Create app keys with <code>POST /v1/keys</code>. Full reference: <a href="/docs">API docs</a>.</p></body></html>"""

    # -------------------------------------------------------------- webhook --
    @app.post("/stripe/webhook", include_in_schema=False)
    async def webhook(request: Request):
        if not settings.stripe_webhook_secret:
            raise HTTPException(503, "webhooks not configured")
        event = verify_stripe_signature(await request.body(), request.headers.get("stripe-signature", ""),
                                        settings.stripe_webhook_secret)
        if db.event_seen(event["id"], event["type"]):
            return {"received": True, "duplicate": True}
        obj = event["data"]["object"]
        etype = event["type"]
        if etype == "checkout.session.completed":
            provision(obj)
        elif etype in ("customer.subscription.updated", "customer.subscription.deleted"):
            t = db.tenant_by("stripe_subscription_id", obj["id"])
            if t:
                fields = {}
                items = (obj.get("items") or {}).get("data") or []
                plan = price_to_plan.get(items[0]["price"]["id"]) if items else None
                if plan and plan != t["plan"]:
                    fields["plan"] = plan
                active = etype != "customer.subscription.deleted" and obj.get("status") in ACTIVE_STATUSES
                if bool(t["active"]) != active:
                    fields["active"] = int(active)
                if fields:
                    db.update_tenant(t["id"], **fields)
                    db.audit(t["id"], "stripe", "tenant.billing_update", {**fields, "status": obj.get("status")})
        return {"received": True}

    # --------------------------------------------------------------- portal --
    @app.post("/v1/billing/portal", tags=["account"])
    def billing_portal(p=Depends(tenant_admin)):
        if not stripe:
            raise HTTPException(503, "online payments are not configured")
        t = db.get_tenant(p.tenant)
        if not t.get("stripe_customer_id"):
            raise HTTPException(404, "this account is not billed through Stripe")
        return {"url": stripe.portal(t["stripe_customer_id"], f"{settings.public_url}/")["url"]}
