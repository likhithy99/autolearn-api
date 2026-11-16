"""Runtime settings and per-project learning defaults."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME_RE = r"^[A-Za-z0-9_-]{1,64}$"
_NAME = re.compile(PROJECT_NAME_RE)

# Every project gets these defaults; any of them can be overridden per project.
DEFAULT_PROJECT_CONFIG: dict = {
    "min_rows": 20,                # labeled rows required before the first model is trained
    "retrain_every": 200,          # retrain after this many new labeled rows arrive
    "adaptive_min_new": 30,        # retrain early on drift/degradation once this many new labels exist
    "min_improvement": 0.0,        # challenger must beat champion by at least this (primary metric)
    "degrade_tolerance": 0.05,     # live accuracy/R2 this far below offline score => "degraded"
    "drift_threshold": 0.5,        # numeric mean shift (in training std units) that counts as drift
    "cv_folds": 3,
    "max_train_rows": 50000,       # most recent rows used for training (bounds training time)
    "self_training": False,        # learn from its own high-confidence predictions (classification)
    "pseudo_label_threshold": 0.97,
    "max_pseudo_fraction": 0.2,    # pseudo-labels never exceed this fraction of real labels
}


# Commercial plans. None = unlimited. rate_limit_per_min applies per API key.
PLANS: dict[str, dict] = {
    "free":       {"predictions_per_month": 10_000,    "max_projects": 2,    "max_rows": 50_000,
                   "rate_limit_per_min": 60,    "self_training": False,
                   "llm_tokens_per_month": 200_000},
    "pro":        {"predictions_per_month": 1_000_000, "max_projects": 20,   "max_rows": 2_000_000,
                   "rate_limit_per_min": 1_200, "self_training": True,
                   "llm_tokens_per_month": 50_000_000},
    "enterprise": {"predictions_per_month": None,      "max_projects": None, "max_rows": None,
                   "rate_limit_per_min": 12_000, "self_training": True,
                   "llm_tokens_per_month": None},
}


def validate_project_name(name: str) -> bool:
    return bool(_NAME.match(name))


def merge_config(overrides: dict | None, base: dict | None = None) -> dict:
    cfg = dict(base or DEFAULT_PROJECT_CONFIG)
    for key, value in (overrides or {}).items():
        if key not in DEFAULT_PROJECT_CONFIG:
            raise ValueError(f"Unknown config key: {key!r}. Allowed: {sorted(DEFAULT_PROJECT_CONFIG)}")
        expected = type(DEFAULT_PROJECT_CONFIG[key])
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ValueError(f"Config {key!r} must be {expected.__name__}")
        cfg[key] = value
    return cfg


@dataclass
class Settings:
    data_dir: Path
    admin_key: str | None
    learn_interval: float
    start_learner: bool = True
    max_body_bytes: int = 20 * 1024 * 1024
    cors_origins: tuple[str, ...] = ()
    llm_dir: Path | None = None
    llm_device: str | None = None
    open_models_dir: Path | None = None  # open-weight models served alongside ours (see llm/open_models.py)
    llm_log_retention_days: int = 30
    public_url: str = "http://localhost:8000"
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_prices: dict | None = None         # plan -> Stripe price id
    price_labels: dict | None = None          # plan -> text shown on the pricing page, e.g. "$49/mo"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("AUTOLEARN_DATA_DIR", "./data")).resolve(),
            admin_key=os.environ.get("AUTOLEARN_ADMIN_KEY") or None,
            learn_interval=float(os.environ.get("AUTOLEARN_LEARN_INTERVAL", "15")),
            start_learner=os.environ.get("AUTOLEARN_DISABLE_LEARNER", "") != "1",
            max_body_bytes=int(os.environ.get("AUTOLEARN_MAX_BODY_BYTES", str(20 * 1024 * 1024))),
            cors_origins=tuple(o for o in os.environ.get("AUTOLEARN_CORS_ORIGINS", "").split(",") if o),
            llm_dir=Path(os.environ["AUTOLEARN_LLM_DIR"]).resolve() if os.environ.get("AUTOLEARN_LLM_DIR") else None,
            llm_device=os.environ.get("AUTOLEARN_LLM_DEVICE") or None,
            open_models_dir=Path(os.environ["AUTOLEARN_OPEN_MODELS_DIR"]).resolve()
            if os.environ.get("AUTOLEARN_OPEN_MODELS_DIR") else None,
            llm_log_retention_days=int(os.environ.get("AUTOLEARN_LLM_LOG_RETENTION_DAYS", "30")),
            public_url=os.environ.get("AUTOLEARN_PUBLIC_URL", "http://localhost:8000").rstrip("/"),
            stripe_secret_key=os.environ.get("STRIPE_SECRET_KEY") or None,
            stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET") or None,
            stripe_prices={p: os.environ[f"STRIPE_PRICE_{p.upper()}"] for p in PLANS
                           if os.environ.get(f"STRIPE_PRICE_{p.upper()}")},
            price_labels=dict(kv.split("=", 1) for kv in os.environ.get(
                "AUTOLEARN_PRICE_LABELS", "free=$0,pro=$49/mo,enterprise=Contact us").split(",") if "=" in kv),
        )
