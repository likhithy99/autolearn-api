"""Content moderation, trained from scratch on Civil Comments (Google/Jigsaw, ~1.8M human-rated comments, CC0).

    python -m autolearn.moderation train --rows 400000 --out models/moderation

Model: TF-IDF (word 1-2 grams + char 3-5 grams) -> one logistic regression per category, with a per-category
decision threshold tuned for F1 on a validation split. Served at POST /v1/moderations (OpenAI-style), and
optionally applied to chat input/output per tenant (`moderation` setting: off | flag | block).
"""

from __future__ import annotations

import argparse
import re
import json
import os
import time
import uuid
from pathlib import Path

import joblib
import numpy as np

# High-precision rules for explicit violent intent, which is rare in the training data (the ML model misses some).
THREAT_RULES = re.compile(
    r"\b(i(?:'| a)?m|i(?:'ll| will| am going to| am gonna|'m gonna|'m going to)|we(?:'ll| will| are going to)|"
    r"gonna|going to)\s+(?:\w+\s+){0,4}(kill|murder|hurt|shoot|stab|beat|strangle|rape|bomb|attack)\s+"
    r"(you|u|him|her|them|your\s+(?:family|kids|children|wife|husband|mother|father))\b"
    r"(?!\s+(?:at|in|on)\s)", re.I)  # "beat you at chess" is not a threat

CATEGORIES = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]


class Moderator:
    def __init__(self, path: str | Path):
        bundle = joblib.load(Path(path) / "model.joblib")
        self.vectorizer = bundle["vectorizer"]
        self.models = bundle["models"]
        self.thresholds = bundle["thresholds"]

    def score(self, texts: list[str]) -> list[dict]:
        X = self.vectorizer.transform([t or "" for t in texts])
        probs = {c: m.predict_proba(X)[:, 1] for c, m in self.models.items()}
        out = []
        for i in range(len(texts)):
            scores = {c: round(float(probs[c][i]), 5) for c in self.models}
            cats = {c: bool(scores[c] >= self.thresholds[c]) for c in self.models}
            if THREAT_RULES.search(texts[i] or ""):
                scores["threat"] = max(scores.get("threat", 0.0), 0.99)
                cats["threat"] = True
            out.append({"flagged": any(cats.values()), "categories": cats, "category_scores": scores})
        return out


def train(a):
    os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import FeatureUnion

    t0 = time.time()
    ds = load_dataset("google/civil_comments", split="train", streaming=True).shuffle(seed=0, buffer_size=50_000)
    texts, labels = [], {c: [] for c in CATEGORIES}
    for ex in ds:
        if not ex["text"]:
            continue
        texts.append(ex["text"])
        for c in CATEGORIES:
            labels[c].append(int(ex[c] >= 0.5))
        if len(texts) >= a.rows:
            break
    Y = {c: np.array(v) for c, v in labels.items()}
    print(f"loaded {len(texts):,} comments in {time.time() - t0:.0f}s; positives: "
          + ", ".join(f"{c}={int(Y[c].sum())}" for c in CATEGORIES), flush=True)

    idx = np.arange(len(texts))
    tr, te = train_test_split(idx, test_size=0.15, random_state=0, stratify=Y["toxicity"])
    tr, va = train_test_split(tr, test_size=0.15, random_state=0, stratify=Y["toxicity"][tr])
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=200_000, min_df=3, sublinear_tf=True,
                                 strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200_000, min_df=5,
                                 sublinear_tf=True)),
    ])
    tx = [texts[i] for i in tr]
    Xtr = vec.fit_transform(tx)
    Xva = vec.transform([texts[i] for i in va])
    Xte = vec.transform([texts[i] for i in te])
    print(f"vectorized: {Xtr.shape[1]:,} features ({time.time() - t0:.0f}s)", flush=True)

    models, thresholds, report = {}, {}, {}
    for c in CATEGORIES:
        if Y[c][tr].sum() < 20:
            print(f"skip {c}: too few positives", flush=True)
            continue
        m = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")
        m.fit(Xtr, Y[c][tr])
        pv = m.predict_proba(Xva)[:, 1]
        grid = np.linspace(0.05, 0.95, 91)
        f1s = [f1_score(Y[c][va], pv >= t, zero_division=0) for t in grid]
        thr = float(grid[int(np.argmax(f1s))])
        pt = m.predict_proba(Xte)[:, 1]
        yt = Y[c][te]
        report[c] = {"positives_test": int(yt.sum()), "roc_auc": round(roc_auc_score(yt, pt), 4),
                     "avg_precision": round(average_precision_score(yt, pt), 4), "threshold": round(thr, 2),
                     "precision": round(precision_score(yt, pt >= thr, zero_division=0), 4),
                     "recall": round(recall_score(yt, pt >= thr, zero_division=0), 4),
                     "f1": round(f1_score(yt, pt >= thr, zero_division=0), 4)}
        models[c], thresholds[c] = m, thr
        print(c, report[c], flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "models": models, "thresholds": thresholds}, out / "model.joblib", compress=3)
    meta = {"dataset": "google/civil_comments (CC0)", "rows": len(texts), "test_rows": len(te),
            "metrics": report, "trained_at": time.time(), "seconds": round(time.time() - t0)}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# --------------------------------------------------------------------- API ----
from pydantic import BaseModel, Field  # noqa: E402


class ModIn(BaseModel):
    input: str | list[str] = Field(max_length=1000)
    model: str | None = None


class ModSetting(BaseModel):
    moderation: str = Field(pattern="^(off|flag|block)$")


def register_moderation_routes(app, settings, db, tenant_any, tenant_admin):
    from fastapi import Depends, HTTPException

    path = Path(os.environ.get("AUTOLEARN_MODERATION_DIR", settings.data_dir.parent / "models" / "moderation"))
    state = {"mod": None, "mtime": None}

    def get():
        f = path / "model.joblib"
        if not f.exists():
            return None
        mtime = f.stat().st_mtime
        if state["mtime"] != mtime:
            state["mod"], state["mtime"] = Moderator(path), mtime
        return state["mod"]

    @app.post("/v1/moderations", tags=["llm"])
    def moderations(body: ModIn, p=Depends(tenant_any)):
        mod = get()
        if mod is None:
            raise HTTPException(503, "moderation model not installed")
        texts = [body.input] if isinstance(body.input, str) else body.input
        db.add_usage(p.tenant, "moderations", len(texts))
        return {"id": "modr-" + uuid.uuid4().hex, "model": "autolearn-moderation", "results": mod.score(texts)}

    @app.put("/v1/settings/moderation", tags=["account"])
    def set_mode(body: ModSetting, p=Depends(tenant_admin)):
        db.update_tenant(p.tenant, moderation=body.moderation)
        db.audit(p.tenant, p.key_id, "settings.moderation", body.model_dump())
        return body

    def check(tenant: str, texts: list[str], where: str) -> dict | None:
        """Returns a moderation summary, raises 400 if the tenant blocks flagged content."""
        t = db.get_tenant(tenant)
        mode = (t or {}).get("moderation") or "off"
        mod = get() if mode != "off" else None
        if mod is None:
            return None
        res = mod.score(texts)
        flagged = [r for r in res if r["flagged"]]
        summary = {"flagged": bool(flagged), "where": where,
                   "categories": sorted({c for r in flagged for c, v in r["categories"].items() if v})}
        if flagged and mode == "block":
            raise HTTPException(400, {"error": "content_policy_violation", "where": where,
                                      "categories": summary["categories"]})
        return summary

    app.state.moderate = check
    return check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--rows", type=int, default=400_000)
    t.add_argument("--out", default="models/moderation")
    t.set_defaults(fn=train)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
