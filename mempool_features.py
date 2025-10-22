# mempool_features.py
# Mempool feature extraction from snapshot CSVs + optional live per-minute collector.
# No derived series. Optional tx_count_z_fast added only if tx_count exists.
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from features_helpers import (
    utc_floor_minute,
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _delta_last, _clip01,
)
from cache_utils import MinuteCache


# ---------------- Feature extractor ---------------- #

@dataclass(frozen=True)
class MempoolConfig:
    z_fast: int = 6 * 60   # ~6h window for z-scores
    d15:    int = 15       # kept for compatibility; not used by default now
    freq:   str = "1min"   # resample grid


def extract_mempool_features_at(
    caches: Mapping[str, pd.Series | pd.DataFrame],
    ts,
    cfg: MempoolConfig = MempoolConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Expected keys in caches (optional, non-derived):
      - "mempool_vsize" : total vsize (vMB-equivalent)
      - "fee_median"    : chosen single fee column (e.g., 'hourFee' OR 'economyFee' OR 'minimumFee')
      - "tx_count"      : mempool transaction count (optional)
    """
    ts = utc_floor_minute(ts)

    def S(key: str) -> pd.Series:
        s = _prep_series(caches.get(key, pd.Series(dtype=float)))
        s = _leq(s, ts)
        return _resample_1m_ffill(s, ts, cfg.freq)

    vsize = S("mempool_vsize")
    fee   = S("fee_median")
    txc   = S("tx_count")  # may be empty

    names: List[str] = ["mempool_vsize_z_fast", "fee_median_z_fast"]
    vals:  List[float] = [
        _z_last(vsize, cfg.z_fast),
        _z_last(fee,   cfg.z_fast),
    ]

    # Add tx_count_z_fast only if present (no derivation)
    if not txc.empty:
        names.append("tx_count_z_fast")
        vals.append(_z_last(txc, cfg.z_fast))

    vals = [_clip01(float(v)) for v in vals]
    return vals, names


# ---------------- CSV loaders (snapshots → caches) ---------------- #

def _read_mempool_snapshot_csv(path: str) -> pd.DataFrame:
    """
    Read a mempool snapshot CSV written by mempool_data.py (timestamp index or 'timestamp' column).
    Returns a UTC-indexed, minute-granularity DataFrame.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # locate timestamp column
    ts_col = None
    for c in df.columns:
        if str(c).lower() == "timestamp":
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]

    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(__ts=idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()

    # coerce known numeric columns if present (no derivations)
    for c in ("vsize", "tx_count", "total_fee",
              "fastestFee", "halfHourFee", "hourFee", "economyFee", "minimumFee"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_caches_from_snapshot_df(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    From a snapshot DataFrame, build the caches expected by extract_mempool_features_at().
    """
    caches: Dict[str, pd.Series] = {}
    if df is None or df.empty:
        return caches

    # mempool_vsize
    if "vsize" in df.columns:
        s = df["vsize"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        s = s[~s.index.duplicated(keep="last")]
        caches["mempool_vsize"] = s.astype(float)

    # fee proxy: choose a single column (non-derived). Preference order kept as before.
    fee_col = None
    for c in ("hourFee", "economyFee", "minimumFee"):
        if c in df.columns:
            fee_col = c
            break
    if fee_col:
        s = df[fee_col].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        s = s[~s.index.duplicated(keep="last")]
        caches["fee_median"] = s.astype(float)

    # tx_count (optional)
    if "tx_count" in df.columns:
        s = df["tx_count"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        s = s[~s.index.duplicated(keep="last")]
        caches["tx_count"] = s.astype(float)

    return caches


# ---------------- Save helpers ---------------- #

def _append_or_overwrite_feature_file(path: str, new_rows: pd.DataFrame, append: bool) -> None:
    """
    Append with dedup by 'timestamp' if append=True; else overwrite.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if append and os.path.exists(path):
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()

        if not old.empty:
            if "timestamp" in old.columns:
                old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
            merged = pd.concat([old, new_rows], ignore_index=True)
        else:
            merged = new_rows.copy()

        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        merged = merged.dropna(subset=["timestamp"])
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        # Zulu formatting for consistency
        merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        merged.to_csv(path, index=False)
        print(f"Appended {len(new_rows)} row(s) → {path} (now {len(merged)} total)")
    else:
        out = new_rows.copy()
        out = out.dropna(subset=["timestamp"])
        out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(path, index=False)
        print(f"Wrote {len(out)} row(s) → {path}")


# ---------------- Optional live collector (one-shot or loop) ---------------- #

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
        fee_proxy = float(r.get("hourFee", r.get("economyFee", r.get("minimumFee", 0.0))))
        cache.upsert("fee_median", ts, fee_proxy)
        if "tx_count" in r.index:
            cache.upsert("tx_count", ts, float(r["tx_count"]))

    vals, names = extract_mempool_features_at(cache.as_mapping(), ts, MempoolConfig())
    return vals, names


# ------------------------------ __main__ ------------------------------ #

if __name__ == "__main__":
    import argparse, time

    ap = argparse.ArgumentParser(description="Mempool features from CSV snapshots or live collector (no derivations).")
    # Mode A: CSV → features
    ap.add_argument("--load-from-csv", default=None,
                    help="Directory or file path for mempool snapshots CSV. "
                         "If directory, uses --file-name inside it.")
    ap.add_argument("--file-name", default="mempool_snapshot.csv",
                    help="Snapshot filename when --load-from-csv is a directory (default: mempool_snapshot.csv).")
    ap.add_argument("--ts", default="last", help="ISO8601, 'now' (UTC), or 'last' (default).")
    ap.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                    help="Directory to write mempool_features.csv (defaults to '.' if flag present with no path).")
    ap.add_argument("--append", action="store_true",
                    help="Append and de-duplicate by timestamp (else overwrite).")
    ap.add_argument("--print", action="store_true", help="Print computed features.")

    # Mode B: live minute collector (optional)
    ap.add_argument("--collect-live", action="store_true",
                    help="Use live mempool API each minute and write features continuously.")
    ap.add_argument("--once", action="store_true",
                    help="With --collect-live, collect only one iteration and exit.")
    ap.add_argument("--out-path", default="data/mempool_features_1m.csv",
                    help="Output CSV path for --collect-live mode.")

    args = ap.parse_args()

    # ---------- Mode B: live collector ----------
    if args.collect_live:
        cache = MinuteCache(max_minutes=24*60)
        out_path = args.out_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        print(f"[mempool-live] writing features to {out_path}")
        last = None
        while True:
            ts = utc_floor_minute(pd.Timestamp.utcnow())
            if last is None or ts > last:
                vals, names = collect_mempool_features_once(ts, cache)
                row = {"timestamp": ts}
                row.update({n: v for n, v in zip(names, vals)})
                _append_or_overwrite_feature_file(out_path, pd.DataFrame([row]), append=True)
                print(f"[mempool-live] {ts.isoformat()} → wrote {len(vals)} feats")
                last = ts
                if args.once:
                    break
                # sleep to next minute boundary
                now = pd.Timestamp.utcnow()
                time.sleep(max(0.0, 60 - (now.second + now.microsecond/1e6)) + 0.25)
            else:
                time.sleep(1.0)
        raise SystemExit(0)

    # ---------- Mode A: CSV → features ----------
    if not args.load_from_csv:
        print("No --load-from-csv provided. Use --collect-live for live sampling, or pass a snapshot CSV/dir.")
        raise SystemExit(0)

    # Resolve snapshot path
    if os.path.isdir(args.load_from_csv):
        snap_path = os.path.join(args.load_from_csv, args.file_name)
    else:
        snap_path = args.load_from_csv

    df_snap = _read_mempool_snapshot_csv(snap_path)
    if df_snap.empty:
        print(f"No snapshot data found at {snap_path}")
        raise SystemExit(0)

    # Choose timestamp
    if args.ts == "now":
        ts_use = utc_floor_minute(pd.Timestamp.utcnow())
    elif args.ts == "last":
        ts_use = pd.to_datetime(df_snap.index.max()).tz_convert("UTC").floor("min")
    else:
        ts_use = utc_floor_minute(pd.to_datetime(args.ts, utc=True))

    caches = _build_caches_from_snapshot_df(df_snap)
    vals, names = extract_mempool_features_at(caches, ts_use, MempoolConfig())

    if args.print or True:
        print(f"[mempool] {len(vals)} features @ {ts_use.isoformat()}")
        for n, v in zip(names, vals):
            print(f"  {n:20s}: {v:+.6f}")

    if args.save_to_csv is not None:
        save_dir = args.save_to_csv
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "mempool_features.csv")
        row = {"timestamp": ts_use}
        row.update({n: float(v) for n, v in zip(names, vals)})
        _append_or_overwrite_feature_file(out_path, pd.DataFrame([row]), append=args.append)
