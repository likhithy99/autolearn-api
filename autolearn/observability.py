"""Dependency-free Prometheus metrics, per-key rate limiting, and structured logging."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict


class Metrics:
    """Minimal Prometheus text-format registry (counters + latency summaries)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[tuple, float] = defaultdict(float)
        self._sums: dict[tuple, list] = defaultdict(lambda: [0.0, 0])

    @staticmethod
    def _key(name, labels):
        return name, tuple(sorted((labels or {}).items()))

    def inc(self, name, labels=None, value=1.0):
        with self._lock:
            self._counters[self._key(name, labels)] += value

    def observe(self, name, value, labels=None):
        with self._lock:
            s = self._sums[self._key(name, labels)]
            s[0] += value
            s[1] += 1

    @staticmethod
    def _fmt(name, labels, suffix=""):
        lab = ",".join(f'{k}="{v}"' for k, v in labels)
        return f"{name}{suffix}{{{lab}}}" if lab else f"{name}{suffix}"

    def render(self, gauges: dict | None = None) -> str:
        lines = []
        with self._lock:
            for (name, labels), v in sorted(self._counters.items()):
                lines.append(f"{self._fmt(name, labels)} {v}")
            for (name, labels), (s, n) in sorted(self._sums.items()):
                lines.append(f"{self._fmt(name, labels, '_sum')} {s}")
                lines.append(f"{self._fmt(name, labels, '_count')} {n}")
        for name, v in (gauges or {}).items():
            lines.append(f"{name} {v}")
        return "\n".join(lines) + "\n"


class RateLimiter:
    """Token bucket per API key: `per_minute` requests, bursting up to the same amount."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, per_minute: int) -> tuple[bool, float]:
        now = time.monotonic()
        rate = per_minute / 60.0
        with self._lock:
            tokens, last = self._buckets.get(key, (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - last) * rate)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, (1 - tokens) / rate


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
                   "msg": record.getMessage()}
        for k in ("request_id", "tenant", "path", "status", "ms"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO", json_logs: bool = True):
    root = logging.getLogger()
    if getattr(root, "_autolearn_configured", False):
        return
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.handlers = [h]
    root.setLevel(level)
    root._autolearn_configured = True
