"""End-to-end production simulation on the real UCI Adult Census dataset.

Story: a customer company onboards, trains a model on the data it has (only people aged <= 35),
then puts it in production where real traffic comes from an older population it has never seen.
Its app sends ground-truth outcomes back as feedback. Nobody calls /train during the live phase:
the service must detect the problem and fix itself.

    python examples/simulate_production.py --url http://localhost:8000 --admin-key <platform key>
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autolearn.client import AutoLearn  # noqa: E402

import httpx  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "datasets" / "adult.csv"


def records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--admin-key", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=400)
    args = ap.parse_args()

    df = pd.read_csv(DATA, skipinitialspace=True).sample(frac=1, random_state=0)
    young, older = df[df.age <= 35], df[df.age > 35]
    initial, stream = young.iloc[:3000], older.iloc[: args.rounds * args.batch]

    # 1. Platform operator onboards a customer.
    r = httpx.post(f"{args.url}/v1/admin/tenants", headers={"X-API-Key": args.admin_key},
                   json={"name": "Demo Bank", "plan": "pro"}).json()
    print(f"[platform] onboarded tenant {r['tenant_id']} (plan=pro)")
    admin = AutoLearn(args.url, r["admin_key"]["api_key"])
    app_key = admin.create_key("loan-scoring web app", role="client")["api_key"]
    app = AutoLearn(args.url, app_key)

    # 2. Customer creates a project and uploads the data it has.
    name = f"income-{int(time.time())}"
    admin.create_project(name, "classification", "income", retrain_every=5000, adaptive_min_new=300)
    admin.upload(name, records(initial))
    res = admin.train(name, wait=True)
    print(f"[customer] trained v{res['version']} {res['algorithm']} on {len(initial)} rows (age<=35): "
          f"offline accuracy={res['metrics']['accuracy']:.3f}  f1={res['metrics']['f1_macro']:.3f}")

    # 3. Live traffic from a population the model never saw; outcomes come back as feedback.
    print("\n[live] older population traffic; no manual retraining from here on")
    print(f"{'round':>5} {'model':>6} {'live acc':>9}  monitor / learner activity")
    last_seen = time.time()
    for i in range(args.rounds):
        batch = stream.iloc[i * args.batch:(i + 1) * args.batch]
        truth = batch["income"].tolist()
        out = app.predict(name, records(batch.drop(columns=["income"])))
        preds = out["predictions"]
        acc = sum(p["prediction"] == t for p, t in zip(preds, truth)) / len(truth)
        app.feedback(name, [(p["prediction_id"], t) for p, t in zip(preds, truth)])

        time.sleep(4)  # give the autonomous learner a cycle
        while admin.project(name)["status"] == "training":
            time.sleep(1)
        mon = admin.monitor(name)
        notes = []
        if mon.get("drift_detected"):
            notes.append(f"drift in {mon['drift']['drifted_features']}")
        if mon.get("degraded"):
            notes.append("live accuracy degraded")
        for e in reversed(admin.events(name, 10)):
            if e["kind"] in ("promote", "reject") and e["ts"] > last_seen:
                notes.append("LEARNER: " + e["message"])
                last_seen = e["ts"]
        print(f"{i + 1:>5} {'v' + str(out['model_version']):>6} {acc:>9.3f}  {'; '.join(notes)}")

    print("\n[history]")
    for m in reversed(admin.models(name)):
        print(f"  v{m['version']} {m['status']:<9} {m['algorithm']:<20} acc={m['test_metrics']['accuracy']:.3f}"
              f"  ({m['reason']})")
    imp = admin.predict(name, records(stream.drop(columns=["income"]).iloc[:1]), explain=True)
    print("\n[explain] top features:", ", ".join(f["feature"] for f in imp["feature_importance"][:5]))
    print("[usage]", admin.me()["usage_this_month"])


if __name__ == "__main__":
    main()
