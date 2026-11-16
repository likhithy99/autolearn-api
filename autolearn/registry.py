"""On-disk model store with a small in-memory cache."""

from __future__ import annotations

import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import joblib


class Registry:
    def __init__(self, root: Path, cache_size: int = 16):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict = OrderedDict()
        self._size = cache_size
        self._lock = threading.Lock()

    def _path(self, project: str, version: int) -> Path:
        return self.root / project / f"v{version}.joblib"

    def save(self, project: str, version: int, bundle) -> None:
        path = self._path(project, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        joblib.dump(bundle, tmp, compress=3)
        tmp.replace(path)  # atomic: readers never see a half-written model

    def load(self, project: str, version: int):
        key = (project, version)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        bundle = joblib.load(self._path(project, version))
        with self._lock:
            self._cache[key] = bundle
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return bundle

    def delete_project(self, project: str) -> None:
        with self._lock:
            for k in [k for k in self._cache if k[0] == project]:
                del self._cache[k]
        shutil.rmtree(self.root / project, ignore_errors=True)
