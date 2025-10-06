# mempool_features.py
# Mempool feature extraction + live per-minute collector → CSV.
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd

from features_helpers import (
    utc_floor_minute,
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _delta_last, _clip01,
)
from cache_utils import MinuteCache


# ---------------- existing feature extractor ---------------- #

@dataclass(frozen=True)
class MempoolConfig:
    z_fast: int = 6 * 60   # ~6h
    d15:    int = 15
    freq:   str = "1min"

def extract_mempool_features_at(
    caches: Mapping[str, pd.Series | pd.DataFrame],
    ts,
    cfg: MempoolConfig = MempoolConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Expected optional keys in caches:
      - "txps"           : transactions per second (if you compute it)
      - "mempool_vsize"  : total vsize (MB-equivalent)
      - "fee_median"     : robust fee proxy (e.g., mempool.space 'hourFee')
    """
    ts = utc_floor_minute(ts)

    def S(key: str) -> pd.Series:
        s = _prep_series(caches.get(key, pd.Series(dtype=float)))
        s = _leq(s, ts)
        return _resample_1m_ffill(s, ts, cfg.freq)

    txps  = S("txps")
    vsize = S("mempool_vsize")
    fee   = S("fee_median")

    names = ["txps_z_fast", "txps_d15", "mempool_vsize_z_fast", "fee_median_z_fast"]
    vals  = [
        _z_last(txps,  cfg.z_fast),
        _delta_last(txps, cfg.d15),
        _z_last(vsize, cfg.z_fast),
        _z_last(fee,   cfg.z_fast),
    ]
    vals = [_clip01(float(v)) for v in vals]
    return vals, names


# ---------------- live collector ---------------- #

def collect_mempool_features_once(ts: pd.Timestamp, cache: MinuteCache) -> Tuple[List[float], List[str]]:
    """
    Snapshot mempool 'now', update rolling cache, compute features at ts.
    Requires mempool_data.mempool_snapshot_now().
    """
    from mempool_data import mempool_snapshot_now

    snap = pd.DataFrame()
    try:
        snap = mempool_snapshot_now()
    except Exception:
        pass

    if not snap.empty:
        r = snap.iloc[-1]
        cache.upsert("mempool_vsize", ts, float(r.get("vsize", 0.0)))
        fee_proxy = float(r.get("hourFee", r.get("economyFee", 0.0)))
        cache.upsert("fee_median", ts, fee_proxy)
        if "txps" in r.index:
            cache.upsert("txps", ts, float(r["txps"]))

    vals, names = extract_mempool_features_at(cache.as_mapping(), ts, MempoolConfig())
    return vals, names

def _append_csv(path: str, row: Dict[str, Any]):
    import os
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser(description="Collect mempool features each minute → CSV.")
    parser.add_argument("--csv", default="data/mempool_features_1m.csv")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    cache = MinuteCache(max_minutes=24*60)
    print(f"[mempool] writing features to {args.csv}")
    last = None
    while True:
        ts = utc_floor_minute(pd.Timestamp.utcnow())
        if last is None or ts > last:
            vals, names = collect_mempool_features_once(ts, cache)
            row = {"timestamp": ts.isoformat()}
            row.update({n: v for n, v in zip(names, vals)})
            _append_csv(args.csv, row)
            print(f"[mempool] {ts.isoformat()} → wrote {len(vals)} feats")
            last = ts
            if args.once: break
            # sleep to next minute boundary
            now = pd.Timestamp.utcnow()
            time.sleep(max(0.0, 60 - (now.second + now.microsecond/1e6)) + 0.25)
        else:
            time.sleep(1.0)
