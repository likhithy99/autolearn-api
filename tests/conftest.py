import numpy as np
import pytest
from fastapi.testclient import TestClient

from autolearn.api import create_app
from autolearn.config import Settings

ADMIN = "test-platform-key"


def make_rows(n, seed=0, flip=False):
    """Synthetic 'will the customer churn?' data: numeric + categorical + free text."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        tenure = float(rng.integers(1, 72))
        spend = float(rng.normal(70, 25))
        plan = str(rng.choice(["basic", "plus", "premium"]))
        complaint = bool(rng.random() < 0.3)
        note = "customer reported slow service and billing errors" if complaint else "happy with product no issues"
        score = (-0.06 * tenure + 0.02 * spend + (1.5 if complaint else 0) + (0.8 if plan == "basic" else 0)
                 + rng.normal(0, 0.5))
        label = "yes" if score > 0.5 else "no"
        if flip:  # concept drift: the world changed
            label = "no" if label == "yes" else "yes"
        rows.append({"tenure": tenure, "monthly_spend": spend, "plan": plan, "note": note, "churned": label})
    return rows


class Ctx:
    def __init__(self, client, app):
        self.client, self.app = client, app
        self.db, self.learner = app.state.db, app.state.learner

    def tenant(self, plan="pro", name="Acme"):
        r = self.client.post("/v1/admin/tenants", json={"name": name, "plan": plan}, headers={"X-API-Key": ADMIN})
        assert r.status_code == 201, r.text
        return r.json()["tenant_id"], {"X-API-Key": r.json()["admin_key"]["api_key"]}


@pytest.fixture
def ctx(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, admin_key=ADMIN, learn_interval=3600, start_learner=False))
    with TestClient(app) as client:
        yield Ctx(client, app)
