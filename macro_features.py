# macro_features.py
# Build macro feature vector from macro closes (DXY, ES, XAU, XAG) at a specified timestamp.
# Matches CLI style of your ohlcv/derivs feature scripts:
#   - --load-from-csv DIR (expects <KEY>_<INTERVAL>.csv with <KEY>_close)
#   - --interval KEY=INTERVAL (so filenames match)
#   - --ts last|now|ISO
#   - --save-to-csv [DIR] (defaults to '.' when flag present without path) + --append (dedup by timestamp)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from features_helpers import (
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _ret_last, _delta_last, _clip01,
)

# ------------------------------ Config ------------------------------ #

# Logical macro keys and their default intervals (mirrors macro_data.py defaults)
DEFAULT_KEYS: List[str] = ["DXY", "ES", "XAU", "XAG"]
DEFAULT_INTERVALS: Dict[str, str] = {
    "DXY": "15m",
    "ES":  "60m",
    "XAU": "60m",
    "XAG": "60m",
}

@dataclass(frozen=True)
class MacroFeatureConfig:
    # Windows in minutes (computed on a 1-minute ffilled grid)
    z_long: int = 7 * 24 * 60       # ~7d z-score window
    ret_60: int = 60                # 1h return
    ret_240: int = 240              # 4h return
    d_60: int = 60                  # 60-minute delta (levels)
    resample_freq: str = "1min"     # we align to per-minute with ffill

# ------------------------------ Loader helpers ------------------------------ #

def _read_key_close_from_csv(path: str, key: str) -> pd.Series:
    """
    Read <KEY>_close (preferred) or 'close' (legacy) from a CSV with a 'timestamp' column.
    Return a UTC-indexed float Series, sorted and de-duplicated.
    """
    if not path or not os.path.exists(path):
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.Series(dtype=float)

    # Find timestamp column
    ts_col = None
    for c in df.columns:
        if str(c).lower() == "timestamp":
            ts_col = c; break
    if ts_col is None:
        ts_col = df.columns[0]

    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(__ts=idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()

    # Prefer <KEY>_close; fall back to 'close'
    prefer = f"{key}_close"
    if prefer in df.columns:
        s = pd.to_numeric(df[prefer], errors="coerce")
    elif "close" in df.columns:
        s = pd.to_numeric(df["close"], errors="coerce")
    else:
        # Heuristic: any *_close
        close_like = [c for c in df.columns if c.lower().endswith("_close")]
        if close_like:
            s = pd.to_numeric(df[close_like[0]], errors="coerce")
        else:
            return pd.Series(dtype=float)

    s = s.dropna()
    s.index = pd.to_datetime(s.index, utc=True)
    s = s[~s.index.duplicated(keep="last")]
    return s.astype(float)

def _paths_from_dir(load_dir: str, key: str, interval: str) -> str:
    """
    Build expected file path for a key under load_dir: <KEY>_<INTERVAL>.csv
    """
    return os.path.join(load_dir, f"{key}_{interval}.csv")

def build_macro_caches_from_dir(load_dir: str,
                                keys: List[str],
                                intervals: Dict[str, str]) -> Dict[str, pd.Series]:
    """
    Load per-key close Series into a cache dict { key: series }.
    """
    caches: Dict[str, pd.Series] = {}
    for key in keys:
        interval = intervals.get(key, DEFAULT_INTERVALS.get(key, "60m"))
        path = _paths_from_dir(load_dir, key, interval)
        s = _read_key_close_from_csv(path, key)
        if not s.empty:
            caches[key] = s
    return caches

# ------------------------------ Core extractor ------------------------------ #

def extract_macro_features_at(
    caches: Dict[str, pd.Series],
    ts,
    cfg: MacroFeatureConfig = MacroFeatureConfig(),
    # which keys to include; defaults to all keys found in caches
    keys: Optional[List[str]] = None,
) -> Tuple[List[float], List[str]]:
    """
    Compute macro features at timestamp `ts` for each available key in `caches`.
    Each key contributes 4 features (bounded to [-1, 1]):
       - {key}_z_long      : z-score of level (7d window)
       - {key}_ret_60m     : log-return over 60 minutes
       - {key}_ret_240m    : log-return over 240 minutes
       - {key}_delta_60m   : level delta over 60 minutes (tanh-scaled)
    """
    ts = pd.to_datetime(ts, utc=True)
    if keys is None:
        keys = list(caches.keys())

    feat_vals: List[float] = []
    feat_names: List[str] = []

    def S(series: pd.Series) -> pd.Series:
        return _resample_1m_ffill(_leq(_prep_series(series), ts), ts, cfg.resample_freq)

    for key in keys:
        s = caches.get(key, pd.Series(dtype=float))
        s = S(s)

        # If the series is empty after filtering, emit zeros for that key
        if s.empty:
            feat_vals.extend([0.0, 0.0, 0.0, 0.0])
            feat_names.extend([
                f"{key}_z_long",
                f"{key}_ret_60m",
                f"{key}_ret_240m",
                f"{key}_delta_60m",
            ])
            continue

        z_long    = _clip01(_z_last(s, cfg.z_long))
        ret_60m   = _clip01(_ret_last(s, cfg.ret_60))
        ret_240m  = _clip01(_ret_last(s, cfg.ret_240))
        delta_60m = _clip01(_delta_last(s, cfg.d_60))

        feat_vals.extend([float(z_long), float(ret_60m), float(ret_240m), float(delta_60m)])
        feat_names.extend([
            f"{key}_z_long",
            f"{key}_ret_60m",
            f"{key}_ret_240m",
            f"{key}_delta_60m",
        ])

    return feat_vals, feat_names

# ------------------------------ Save helpers ------------------------------ #

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
        # write as Zulu strings for consistency
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

# ------------------------------ __main__ ------------------------------ #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Macro feature extractor from macro CSVs (DXY/ES/XAU/XAG).")
    ap.add_argument("--ts", default="last", help="ISO8601, 'now' (UTC), or 'last' (default).")
    ap.add_argument("--load-from-csv", default=None,
                    help="Directory containing macro CSVs from macro_data.py (<KEY>_<INTERVAL>.csv).")
    ap.add_argument("--key", action="append", default=DEFAULT_KEYS,
                    help="Macro keys to include (repeatable). Default: DXY, ES, XAU, XAG.")
    # interval overrides used only to resolve filenames in --load-from-csv
    ap.add_argument("--interval", action="append", default=[],
                    help="Override per-key interval for filenames, e.g. DXY=30m (repeatable).")
    # manual single-path overrides (optional; mainly for debugging)
    ap.add_argument("--csv", action="append", default=[],
                    help="Manual mapping KEY=/path/to/file.csv (repeatable) to override load-from-csv behavior.")
    # save behavior matches your other scripts
    ap.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                    help="Directory to write macro_features.csv (default '.' if flag present with no path).")
    ap.add_argument("--append", action="store_true",
                    help="Append rows to macro_features.csv and de-duplicate by timestamp (else overwrite).")
    # tuning
    ap.add_argument("--z-days", type=float, default=7.0, help="Z-score window in days (default 7).")
    ap.add_argument("--ret60", type=int, default=60, help="Return horizon minutes (default 60).")
    ap.add_argument("--ret240", type=int, default=240, help="Return horizon minutes (default 240).")
    ap.add_argument("--d60", type=int, default=60, help="Delta horizon minutes (default 60).")
    ap.add_argument("--print", action="store_true", help="Print features to stdout.")
    args = ap.parse_args()

    # Build interval map for filename resolution
    intervals = dict(DEFAULT_INTERVALS)
    for ov in args.interval:
        if "=" in ov:
            k, v = ov.split("=", 1)
            intervals[k.strip()] = v.strip()

    # Resolve timestamp
    if args.ts == "now":
        ts_use = pd.Timestamp.now(tz="UTC").floor("min")
    elif args.ts == "last":
        ts_use = None  # decide after we load series
    else:
        ts_use = pd.to_datetime(args.ts, utc=True).floor("min")

    # Load caches
    caches: Dict[str, pd.Series] = {}

    # 1) If manual KEY=path overrides were provided, apply them first
    manual_map: Dict[str, str] = {}
    for spec in args.csv:
        if "=" in spec:
            k, p = spec.split("=", 1)
            manual_map[k.strip()] = p.strip()

    for k, p in manual_map.items():
        s = _read_key_close_from_csv(p, k)
        if not s.empty:
            caches[k] = s

    # 2) If a directory was provided, use expected filenames for any remaining keys
    if args.load_from_csv:
        for k in args.key:
            if k in caches:
                continue
            interval = intervals.get(k, DEFAULT_INTERVALS.get(k, "60m"))
            path = _paths_from_dir(args.load_from_csv, k, interval)
            s = _read_key_close_from_csv(path, k)
            if not s.empty:
                caches[k] = s

    # If still empty, warn and exit gracefully
    if not caches:
        print("No macro series loaded. Provide --load-from-csv DIR or --csv KEY=path.")
        raise SystemExit(0)

    # If ts_use is 'last', pick the latest timestamp across series
    if ts_use is None:
        last_candidates: List[pd.Timestamp] = []
        for s in caches.values():
            if isinstance(s, pd.Series) and not s.empty:
                last_candidates.append(pd.Timestamp(s.index[-1]).tz_convert("UTC"))
        if not last_candidates:
            print("No timestamps found in loaded macro series.")
            raise SystemExit(0)
        ts_use = max(last_candidates).floor("min")

    # Build config from CLI
    cfg = MacroFeatureConfig(
        z_long=int(round(args.z_days * 24 * 60)),
        ret_60=int(args.ret60),
        ret_240=int(args.ret240),
        d_60=int(args.d60),
        resample_freq="1min",
    )

    # Compute features
    vals, names = extract_macro_features_at(caches, ts_use, cfg, keys=args.key)

    if args.print or True:
        print(f"[macro] {len(vals)} features @ {ts_use.isoformat()}")
        for n, v in zip(names, vals):
            print(f"  {n:22s}: {v:+.6f}")

    # Save if requested
    if args.save_to_csv is not None:
        save_dir = args.save_to_csv
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "macro_features.csv")
        row = {"timestamp": ts_use}
        row.update({n: float(v) for n, v in zip(names, vals)})
        out_df = pd.DataFrame([row])
        _append_or_overwrite_feature_file(out_path, out_df, append=args.append)
