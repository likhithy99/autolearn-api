"""Production monitoring: input drift and live accuracy from user feedback."""

from __future__ import annotations

import pandas as pd

from .automl import LIVE_METRIC, compute_metrics, prep_target
from .features import prepare_frame


def drift_report(bundle, recent_rows: list[dict], threshold: float) -> dict:
    """Compare recent API inputs against the champion's training distribution."""
    X = prepare_frame(recent_rows, bundle.schema)
    features = {}
    for col, st in bundle.ref_stats["numeric"].items():
        m = X[col].mean()
        if pd.isna(m):
            continue
        scale = st["std"] if st["std"] > 1e-12 else 1.0
        shift = abs(m - st["mean"]) / scale
        features[col] = {"mean_shift_std": round(float(shift), 4), "drifted": bool(shift > threshold)}
    for col, cats in bundle.ref_stats["categorical"].items():
        vals = X[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        unseen = float((~vals.isin(set(cats))).mean())
        features[col] = {"unseen_rate": round(unseen, 4), "drifted": bool(unseen > 0.2)}
    drifted = sorted(c for c, f in features.items() if f["drifted"])
    return {"n": len(recent_rows), "drift_detected": bool(drifted), "drifted_features": drifted,
            "features": features}


def live_performance(task: str, pairs: list[tuple]) -> dict:
    preds = [p for p, _ in pairs]
    actual = prep_target(pd.Series([a for _, a in pairs]), task)
    if task == "classification":
        # align label types with predictions (e.g. 1 vs "1")
        if preds and isinstance(preds[0], str):
            actual = actual.astype(str)
    return compute_metrics(task, actual, preds)


def is_degraded(task: str, live: dict | None, offline: dict | None, tolerance: float) -> bool:
    if not live or not offline:
        return False
    key = LIVE_METRIC[task]
    if live.get(key) is None or offline.get(key) is None:
        return False
    return live[key] < offline[key] - tolerance
