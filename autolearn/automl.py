"""Model search: try several algorithms with cross-validation and keep the best."""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, infer_schema, prepare_frame, reference_stats

PRIMARY_METRIC = {"classification": "f1_macro", "regression": "r2"}
LIVE_METRIC = {"classification": "accuracy", "regression": "r2"}
_SCORING = {"classification": "f1_macro", "regression": "r2"}


def py(v):
    """numpy scalar -> plain Python (JSON-safe)."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def candidates(task: str, has_text: bool) -> dict[str, tuple]:
    """algorithm name -> (estimator, needs_dense_input)."""
    if task == "classification":
        c = {
            "logistic_regression": (LogisticRegression(max_iter=3000, class_weight="balanced"), False),
            "random_forest": (RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                n_jobs=-1, random_state=0), False),
        }
        if not has_text:
            c["gradient_boosting"] = (HistGradientBoostingClassifier(random_state=0), True)
    else:
        c = {
            "ridge": (Ridge(alpha=1.0), False),
            "random_forest": (RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0), False),
        }
        if not has_text:
            c["gradient_boosting"] = (HistGradientBoostingRegressor(random_state=0), True)
    return c


def prep_target(y: pd.Series, task: str) -> pd.Series:
    if task == "regression":
        return pd.to_numeric(y, errors="raise").astype(float)
    if y.map(type).nunique() > 1:  # mixed label types -> normalise to strings
        return y.astype(str)
    return y.astype(object) if not pd.api.types.is_numeric_dtype(y) else y


def compute_metrics(task: str, y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    if task == "classification":
        return {
            "f1_macro": py(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": py(accuracy_score(y_true, y_pred)),
            "n": len(y_true),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    return {
        "r2": py(r2),
        "mae": py(mean_absolute_error(y_true, y_pred)),
        "rmse": py(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": len(y_true),
    }


class ModelBundle:
    """A fitted pipeline plus everything needed to serve and monitor it."""

    def __init__(self, pipeline, schema, task, target, algorithm, ref_stats, n_train):
        self.pipeline = pipeline
        self.schema = schema
        self.task = task
        self.target = target
        self.algorithm = algorithm
        self.ref_stats = ref_stats
        self.n_train = n_train
        self.trained_at = time.time()

    def predict(self, rows) -> list[dict]:
        X = prepare_frame(rows, self.schema)
        if self.task == "classification":
            proba = self.pipeline.predict_proba(X)
            classes = [py(c) for c in self.pipeline.classes_]
            out = []
            for p in proba:
                i = int(np.argmax(p))
                out.append({
                    "prediction": classes[i],
                    "confidence": float(p[i]),
                    "probabilities": {str(c): round(float(pp), 6) for c, pp in zip(classes, p)},
                })
            return out
        return [{"prediction": float(v), "confidence": None} for v in self.pipeline.predict(X)]

    def score(self, df: pd.DataFrame) -> dict:
        y = prep_target(df[self.target], self.task)
        preds = [r["prediction"] for r in self.predict(df.drop(columns=[self.target]))]
        return compute_metrics(self.task, y, preds)


def _has_text(schema):
    return any(k == "text" for k in schema.values())


def _make_cv(y: pd.Series, task: str, folds: int):
    if task == "classification":
        k = min(folds, int(y.value_counts().min()))
        if k >= 2:
            return StratifiedKFold(k, shuffle=True, random_state=0)
    return KFold(max(2, min(folds, len(y))), shuffle=True, random_state=0)


def fit_bundle(df: pd.DataFrame, target: str, task: str, algorithm: str) -> ModelBundle:
    y = prep_target(df[target], task)
    X = df.drop(columns=[target])
    schema = infer_schema(X)
    if not schema:
        raise ValueError("No usable feature columns found")
    Xp = prepare_frame(X, schema)
    cands = candidates(task, _has_text(schema))
    if algorithm not in cands:
        algorithm = "random_forest"
    est, dense = cands[algorithm]
    pipe = Pipeline([("features", build_preprocessor(schema, dense)), ("model", clone(est))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(Xp, y)
    return ModelBundle(pipe, schema, task, target, algorithm, reference_stats(Xp, schema), len(df))


def run_automl(df: pd.DataFrame, target: str, task: str, cv_folds: int = 3) -> tuple[ModelBundle, list[dict]]:
    """Cross-validate every candidate algorithm; refit the winner on all of `df`."""
    y = prep_target(df[target], task)
    if task == "classification" and y.nunique() < 2:
        raise ValueError("Classification needs at least 2 distinct labels")
    X = df.drop(columns=[target])
    schema = infer_schema(X)
    if not schema:
        raise ValueError("No usable feature columns found")
    Xp = prepare_frame(X, schema)
    cv = _make_cv(y, task, cv_folds)

    board = []
    for name, (est, dense) in candidates(task, _has_text(schema)).items():
        pipe = Pipeline([("features", build_preprocessor(schema, dense)), ("model", clone(est))])
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, Xp, y, cv=cv, scoring=_SCORING[task], error_score="raise")
            board.append({
                "algorithm": name,
                "cv_score": py(np.nanmean(scores)),
                "cv_std": py(np.nanstd(scores)),
                "seconds": round(time.time() - t0, 2),
            })
        except Exception as e:  # one bad candidate must not sink the search
            board.append({"algorithm": name, "error": str(e)[:300]})

    ok = [b for b in board if b.get("cv_score") is not None]
    if not ok:
        raise RuntimeError(f"All candidate models failed: {board}")
    best = max(ok, key=lambda b: b["cv_score"])
    board.sort(key=lambda b: -(b.get("cv_score") if b.get("cv_score") is not None else -1e9))
    return fit_bundle(df, target, task, best["algorithm"]), board
