"""The autonomous learning loop.

Every `interval` seconds, for each project, the learner decides by itself whether to retrain:
  * initial training   - enough labeled rows exist and there is no model yet
  * scheduled          - `retrain_every` new labeled rows have arrived
  * adaptive           - live accuracy (from feedback) degraded, or the input distribution drifted,
                         and at least `adaptive_min_new` fresh labels are available to learn from

Every retrain is a champion/challenger contest: the challenger is only promoted if it scores
at least as well as the current champion on held-out data (preferably data neither has seen).
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .automl import PRIMARY_METRIC, fit_bundle, prep_target, run_automl
from .features import prepare_frame
from .monitor import drift_report, is_degraded, live_performance

log = logging.getLogger("autolearn.learner")


class Learner:
    def __init__(self, db, registry, interval: float = 15.0, metrics=None):
        self.db = db
        self.registry = registry
        self.interval = interval
        self.metrics = metrics
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="autolearn-learner", daemon=True)
        self._thread.start()
        log.info("learner started (interval=%ss)", self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.wait(self.interval):
            for p in self.db.list_projects():
                if self._stop.is_set():
                    return
                try:
                    self.check(p["name"])
                except Exception:
                    log.exception("learner check failed for %s", p["name"])

    def is_training(self, project: str) -> bool:
        return self._locks[project].locked()

    # -- monitoring ----------------------------------------------------------
    def health(self, project: str) -> dict:
        p = self.db.get_project(project)
        out = {"drift": None, "live": None, "offline": None, "degraded": False, "drift_detected": False}
        if not p or p["champion_version"] is None:
            return out
        v = p["champion_version"]
        bundle = self.registry.load(project, v)
        model = self.db.get_model(project, v)
        out["offline"] = model["metrics"].get("test") if model else None
        recent = self.db.recent_inputs(project, v, 500)
        if len(recent) >= 30:
            out["drift"] = drift_report(bundle, recent, p["config"]["drift_threshold"])
            out["drift_detected"] = out["drift"]["drift_detected"]
        pairs = self.db.recent_feedback(project, v, 300)
        if len(pairs) >= 20:
            out["live"] = live_performance(p["task"], pairs)
            out["degraded"] = is_degraded(p["task"], out["live"], out["offline"], p["config"]["degrade_tolerance"])
        return out

    def _track_state(self, project, p, h):
        """Log an event whenever drift/degradation status flips, and persist it."""
        state = dict(p["state"])
        for flag, msg in (("drift_detected", "input drift"), ("degraded", "live performance degradation")):
            if bool(h[flag]) != bool(state.get(flag)):
                detail = ""
                if flag == "drift_detected" and h["drift"]:
                    detail = f" in {', '.join(h['drift']['drifted_features'][:8])}"
                if flag == "degraded" and h["live"]:
                    detail = f" (live={h['live']}, offline={h['offline']})"
                self.db.log_event(project, "monitor", f"{msg} {'detected' + detail if h[flag] else 'cleared'}")
                state[flag] = bool(h[flag])
        if state != p["state"]:
            self.db.update_project(project, state=state)

    # -- decision ------------------------------------------------------------
    def check(self, project: str):
        p = self.db.get_project(project)
        if not p:
            return None
        cfg = p["config"]
        new = self.db.count_real(project, p["last_trained_sample_id"])
        if p["champion_version"] is None:
            if new > 0 and self.db.count_real(project) >= cfg["min_rows"]:
                return self.train(project, "initial training")
            return None
        if new >= cfg["retrain_every"]:
            return self.train(project, f"scheduled: {new} new labeled rows")
        h = self.health(project)
        self._track_state(project, p, h)
        if new >= cfg["adaptive_min_new"]:
            if h["degraded"]:
                return self.train(project, f"adaptive: live performance degraded ({new} new labels)")
            if h["drift_detected"]:
                return self.train(project, f"adaptive: input drift in {h['drift']['drifted_features'][:5]} "
                                           f"({new} new labels)")
        return None

    # -- training ------------------------------------------------------------
    def train(self, project: str, reason: str) -> dict:
        lock = self._locks[project]
        if not lock.acquire(blocking=False):
            return {"status": "busy", "message": "training already in progress"}
        t0 = time.time()
        try:
            result = self._train(project, reason)
            result["seconds"] = round(time.time() - t0, 2)
            if self.metrics:
                self.metrics.inc("autolearn_trainings_total", {"outcome": result["status"]})
            return result
        except Exception as e:
            log.exception("training failed for %s", project)
            p = self.db.get_project(project)
            if p:  # don't retry the same data in a tight loop
                self.db.update_project(project, last_trained_sample_id=self.db.max_sample_id(project))
            self.db.log_event(project, "error", f"training failed ({reason}): {e}")
            if self.metrics:
                self.metrics.inc("autolearn_trainings_total", {"outcome": "error"})
            return {"status": "error", "message": str(e)}
        finally:
            lock.release()

    def _load_frame(self, p) -> pd.DataFrame:
        cfg, target = p["config"], p["target"]
        rows = self.db.labeled_samples(p["name"], cfg["max_train_rows"])
        recs = []
        for r in rows:
            rec = dict(r["features"])
            rec.pop(target, None)
            rec[target] = r["label"]
            rec["__id"], rec["__source"] = r["id"], r["source"]
            recs.append(rec)
        df = pd.DataFrame(recs)
        real = df[df["__source"] != "pseudo"]
        pseudo = df[df["__source"] == "pseudo"]
        cap = int(cfg["max_pseudo_fraction"] * len(real))
        if len(pseudo) > cap:
            pseudo = pseudo.tail(cap)  # most recent pseudo-labels only
        return pd.concat([real, pseudo]).sort_values("__id")

    def _split(self, p, df):
        """Prefer a 'fresh' test set: real rows newer than the champion's training data."""
        real = df[df["__source"] != "pseudo"]
        n_test = max(10, int(0.2 * len(real)))
        fresh = real[real["__id"] > p["last_trained_sample_id"]]
        if p["champion_version"] is not None and len(fresh) >= 30:
            test = fresh.sample(min(len(fresh), max(n_test, len(fresh) // 2)), random_state=0)
            protocol = "fresh_holdout"
        else:
            strat = None
            if p["task"] == "classification":
                y = prep_target(real[p["target"]], "classification")
                if y.value_counts().min() >= 2 and n_test >= y.nunique():
                    strat = y
            _, test = train_test_split(real, test_size=n_test, random_state=0, stratify=strat)
            protocol = "random_holdout"
        train = df[~df["__id"].isin(test["__id"])]
        return train, test, protocol

    def _train(self, project: str, reason: str) -> dict:
        p = self.db.get_project(project)
        cfg, task, target = p["config"], p["task"], p["target"]
        self.db.log_event(project, "train", f"training started: {reason}")
        max_id = self.db.max_sample_id(project)
        df = self._load_frame(p)
        n_real = int((df["__source"] != "pseudo").sum())
        if n_real < cfg["min_rows"]:
            raise ValueError(f"need at least {cfg['min_rows']} labeled rows, have {n_real}")

        train, test, protocol = self._split(p, df)
        drop = ["__id", "__source"]
        test_x = test.drop(columns=drop)
        key = PRIMARY_METRIC[task]

        # Candidate data windows: the learner decides how much history is still relevant.
        # After a concept shift, old rows contradict new ones and a recent window wins.
        windows = {"all_history": train}
        fresh_train = train[train["__id"] > p["last_trained_sample_id"]]
        if protocol == "fresh_holdout" and len(fresh_train) >= max(cfg["min_rows"], 30):
            windows["since_last_training"] = fresh_train
        half = train[train["__id"] >= train["__id"].quantile(0.5)]
        if len(train) >= 400 and len(half) < len(train):
            windows["recent_half"] = half

        best = None
        window_scores = {}
        for wname, wdf in windows.items():
            try:
                bundle, board = run_automl(wdf.drop(columns=drop), target, task, cfg["cv_folds"])
            except Exception as e:
                window_scores[wname] = {"error": str(e)[:200]}
                continue
            m = bundle.score(test_x)
            window_scores[wname] = {"rows": len(wdf), "algorithm": bundle.algorithm, key: m[key]}
            if best is None or (m[key] if m[key] is not None else -1e9) > (best[2][key] if best[2][key] is not None else -1e9):
                best = (wname, bundle, m, board)
        if best is None:
            raise RuntimeError(f"no data window produced a model: {window_scores}")
        window, challenger, ch_metrics, board = best

        champ_metrics = None
        if p["champion_version"] is not None:
            try:
                champ_metrics = self.registry.load(project, p["champion_version"]).score(test_x)
            except Exception as e:  # e.g. schema incompatible -> challenger wins by default
                log.warning("could not score champion: %s", e)

        promote = (champ_metrics is None or champ_metrics.get(key) is None or
                   (ch_metrics[key] or -1e9) >= champ_metrics[key] + cfg["min_improvement"])

        # Final refit uses the winning window plus the (same-window) holdout rows.
        if window == "all_history":
            full = df
        else:
            full = df[df["__id"] >= windows[window]["__id"].min()]
        full_x = full.drop(columns=drop)
        train_x = windows[window].drop(columns=drop)

        importance = self._importance(challenger, test_x, target, task)
        version = self.db.next_model_version(project)
        if promote:
            # Winner is refit on its whole data window (incl. holdout rows) before going live.
            final = fit_bundle(full_x, target, task, challenger.algorithm)
        else:
            final = challenger
        self.registry.save(project, version, final)
        metrics = {
            "test": ch_metrics,
            "champion_on_same_test": champ_metrics,
            "primary_metric": key,
            "eval_protocol": protocol,
            "data_window": window,
            "data_windows_evaluated": window_scores,
            "n_train": len(train_x),
            "n_test": len(test_x),
            "n_pseudo": int((df["__source"] == "pseudo").sum()),
            "leaderboard": board,
            "feature_importance": importance,
            "schema": final.schema,
        }
        status = "champion" if promote else "rejected"
        self.db.add_model(project, version, final.algorithm, status, metrics, reason, final.n_train)
        if promote:
            self.db.promote(project, version)
        state = dict(p["state"], drift_detected=False, degraded=False) if promote else p["state"]
        self.db.update_project(project, last_trained_sample_id=max_id, state=state)

        cmp = f" vs champion v{p['champion_version']} {key}={champ_metrics[key]:.4f}" if champ_metrics else ""
        msg = (f"v{version} {final.algorithm} [{window}]: {key}={ch_metrics[key]:.4f}{cmp} on {len(test_x)} "
               f"{protocol.replace('_', ' ')} rows -> {'PROMOTED' if promote else 'rejected, champion kept'}")
        self.db.log_event(project, "promote" if promote else "reject", msg)
        log.info("[%s] %s", project, msg)
        return {"status": "promoted" if promote else "rejected", "version": version,
                "algorithm": final.algorithm, "data_window": window, "metrics": ch_metrics, "champion_metrics": champ_metrics,
                "reason": reason}

    @staticmethod
    def _importance(bundle, test_x, target, task, max_rows=500):
        """Model-agnostic permutation importance on raw input columns (explainability for customers)."""
        try:
            sample = test_x.sample(min(len(test_x), max_rows), random_state=0)
            X = prepare_frame(sample.drop(columns=[target]), bundle.schema)
            y = prep_target(sample[target], task)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = permutation_importance(bundle.pipeline, X, y, n_repeats=3, random_state=0, n_jobs=1,
                                           scoring="f1_macro" if task == "classification" else "r2")
            pairs = sorted(zip(X.columns, r.importances_mean), key=lambda t: -t[1])
            return [{"feature": c, "importance": round(float(v), 5)} for c, v in pairs if np.isfinite(v)]
        except Exception as e:
            log.warning("feature importance failed: %s", e)
            return []
