"""Self-serve billing: checkout -> webhook provisioning -> one-time key reveal -> plan change / suspension."""

import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from autolearn import billing
from autolearn.api import create_app
from autolearn.config import Settings

SECRET = "whsec_test"


def signed(payload: dict, secret=SECRET, ts=None):
    body = json.dumps(payload).encode()
    ts = ts or int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return body, {"Stripe-Signature": f"t={ts},v1={sig}", "Content-Type": "application/json"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    calls = []

    def fake_req(self, method, path, data=None):
        calls.append((method, path, data))
        if path == "/checkout/sessions":
            return {"id": "cs_1", "url": "https://checkout.stripe.test/cs_1"}
        if path.startswith("/checkout/sessions/"):
            return {"id": path.rsplit("/", 1)[1], "payment_status": "unpaid"}
        if path == "/billing_portal/sessions":
            return {"url": "https://billing.stripe.test/p"}
        raise AssertionError(path)

    monkeypatch.setattr(billing.Stripe, "_req", fake_req)
    s = Settings(data_dir=tmp_path, admin_key="adm", learn_interval=3600, start_learner=False,
                 stripe_secret_key="sk_test", stripe_webhook_secret=SECRET,
                 stripe_prices={"pro": "price_pro", "enterprise": "price_ent"}, price_labels={"pro": "$49/mo"})
    with TestClient(create_app(s)) as c:
        c.calls = calls
        yield c


def completed_event(eid="evt_1"):
    return {"id": eid, "type": "checkout.session.completed", "data": {"object": {
        "id": "cs_1", "payment_status": "paid", "customer": "cus_1", "subscription": "sub_1",
        "customer_details": {"email": "buyer@example.com", "name": "Buyer Inc"}, "metadata": {"plan": "pro"}}}}


def test_pricing_page_and_checkout_redirect(client):
    page = client.get("/").text
    assert "$49/mo" in page and "/signup?plan=pro" in page
    r = client.get("/signup?plan=pro", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("https://checkout.stripe.test")
    assert client.calls[0][2]["line_items[0][price]"] == "price_pro"
    assert client.get("/signup?plan=free", follow_redirects=False).status_code == 404


def provision_and_get_key(client):
    body, h = signed(completed_event())
    assert client.post("/stripe/webhook", content=body, headers=h).json() == {"received": True}
    # replayed event is ignored (idempotent)
    assert client.post("/stripe/webhook", content=body, headers=h).json()["duplicate"] is True
    page = client.get("/billing/success?session_id=cs_1").text
    key = page.split("<pre>")[1].split("</pre>")[0]
    assert key.startswith("al_adm_")
    me = client.get("/v1/me", headers={"X-API-Key": key}).json()
    assert me["plan"] == "pro" and me["name"] == "Buyer Inc"
    assert "already shown" in client.get("/billing/success?session_id=cs_1").text
    assert client.post("/v1/billing/portal", headers={"X-API-Key": key}).json()["url"].startswith("https://billing")
    return key


def test_webhook_provisions_and_reveals_key_once(client):
    provision_and_get_key(client)


def test_bad_signature_rejected(client):
    body, h = signed(completed_event(), secret="wrong")
    assert client.post("/stripe/webhook", content=body, headers=h).status_code == 400
    body, h = signed(completed_event(), ts=int(time.time()) - 3600)
    assert client.post("/stripe/webhook", content=body, headers=h).status_code == 400


def test_subscription_upgrade_and_cancellation(client):
    key = provision_and_get_key(client)
    upd = {"id": "evt_2", "type": "customer.subscription.updated", "data": {"object": {
        "id": "sub_1", "status": "active", "items": {"data": [{"price": {"id": "price_ent"}}]}}}}
    body, h = signed(upd)
    client.post("/stripe/webhook", content=body, headers=h)
    assert client.get("/v1/me", headers={"X-API-Key": key}).json()["plan"] == "enterprise"
    unpaid = {"id": "evt_3", "type": "customer.subscription.updated", "data": {"object": {
        "id": "sub_1", "status": "past_due", "items": {"data": [{"price": {"id": "price_ent"}}]}}}}
    body, h = signed(unpaid)
    client.post("/stripe/webhook", content=body, headers=h)
    assert client.get("/v1/me", headers={"X-API-Key": key}).status_code == 403  # suspended
    paid = {"id": "evt_4", "type": "customer.subscription.updated", "data": {"object": {
        "id": "sub_1", "status": "active", "items": {"data": [{"price": {"id": "price_ent"}}]}}}}
    body, h = signed(paid)
    client.post("/stripe/webhook", content=body, headers=h)
    assert client.get("/v1/me", headers={"X-API-Key": key}).status_code == 200  # reactivated
    gone = {"id": "evt_5", "type": "customer.subscription.deleted", "data": {"object": {"id": "sub_1",
                                                                                     "status": "canceled"}}}
    body, h = signed(gone)
    client.post("/stripe/webhook", content=body, headers=h)
    assert client.get("/v1/me", headers={"X-API-Key": key}).status_code == 403


def test_success_page_before_payment(client):
    r = client.get("/billing/success?session_id=cs_unpaid")
    assert r.status_code == 402
