# cache_utils.py
# Lightweight per-minute rolling cache used by collectors and feature extractors.

from __future__ import annotations
import threading

from typing import Dict, Optional
import pandas as pd
from features_helpers import utc_floor_minute


class MinuteCache:
    """
    Stores one minute-indexed Series per key.
    - upsert(key, ts, value): add/replace the value at minute ts
    - set_series(key, series): replace entire history (expects UTC index)
    - as_mapping(): dict[str, Series] for passing into feature extractors
    """
    def __init__(self, max_minutes: int = 7 * 24 * 60):
        self.max_minutes = max_minutes
        self._store: Dict[str, pd.Series] = {}
        self._lock = threading.RLock()

    def upsert(self, key: str, ts, value: float):
        with self._lock:
            ts = utc_floor_minute(ts)
            s = self._store.get(key, pd.Series(dtype=float))
            s.loc[ts] = float(value)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            cutoff = ts - pd.Timedelta(minutes=self.max_minutes)
            self._store[key] = s.loc[s.index >= cutoff]

    def set_series(self, key: str, series: pd.Series):
        with self._lock:
            if series is None or series.empty:
                return
            s = series.copy()
            s.index = pd.to_datetime(s.index, utc=True)
            s = s.sort_index().astype(float)
            cutoff = s.index[-1] - pd.Timedelta(minutes=self.max_minutes)
            self._store[key] = s.loc[s.index >= cutoff]

    def get(self, key: str) -> pd.Series:
        with self._lock:
            return self._store.get(key, pd.Series(dtype=float))

    def as_mapping(self) -> Dict[str, pd.Series]:
        with self._lock:
            return dict(self._store)

    def prune(self, ts=None):
        """Drop points older than max_minutes relative to ts (or latest)."""
        with self._lock:
            if not self._store:
                return
            if ts is None:
                # use the newest timestamp across keys
                latest = max((s.index[-1] for s in self._store.values() if not s.empty), default=None)
                if latest is None:
                    return
                ts = latest
            ts = utc_floor_minute(ts)
            cutoff = ts - pd.Timedelta(minutes=self.max_minutes)
            for k, s in list(self._store.items()):
                self._store[k] = s.loc[s.index >= cutoff]
