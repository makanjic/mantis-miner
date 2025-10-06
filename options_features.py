# options_features.py
# Options feature extraction (DVOL, skew, term slope) + live per-minute collector → CSV.
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
class OptionsConfig:
    z_long: int = 7 * 24 * 60   # ~7d DVOL regime
    z_med:  int = 24 * 60       # ~1d for skew
    d60:    int = 60            # 60-min change
    freq:   str = "1min"

def extract_options_features_at(
    caches: Mapping[str, pd.Series | pd.DataFrame],
    ts,
    cfg: OptionsConfig = OptionsConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Expected optional keys:
      - "dvol_30d"
      - "skew_25d"
      - "iv_7d", "iv_30d"
    """
    ts = utc_floor_minute(ts)

    def S(key: str) -> pd.Series:
        s = _prep_series(caches.get(key, pd.Series(dtype=float)))
        s = _leq(s, ts)
        return _resample_1m_ffill(s, ts, cfg.freq)

    dvol = S("dvol_30d")
    skew = S("skew_25d")
    iv7  = S("iv_7d")
    iv30 = S("iv_30d")

    term_slope = 0.0
    if not iv7.empty and not iv30.empty:
        term_slope = float(np.tanh(iv7.iloc[-1] - iv30.iloc[-1]))

    names = ["dvol_z_long", "dvol_d60", "skew25d_z_med", "iv_term_slope"]
    vals  = [
        _z_last(dvol, cfg.z_long),
        _delta_last(dvol, cfg.d60),
        _z_last(skew, cfg.z_med),
        term_slope,
    ]
    vals = [_clip01(float(v)) for v in vals]
    return vals, names


# ---------------- live collector ---------------- #

def _compute_option_terms_from_surface(surface_df: pd.DataFrame) -> dict:
    """
    Deribit surface snapshot → approx iv_7d, iv_30d (ATM) and skew_25d (30d RR).
    """
    out: dict = {}
    if surface_df is None or surface_df.empty:
        return out
    s = surface_df.copy()
    s["expiry"] = pd.to_datetime(s["expiry"], utc=True)
    now = s["snapshot_ts"].iloc[0]
    s["ttm_days"] = (s["expiry"] - now).dt.total_seconds() / 86400.0

    def atm_iv(target_days: float) -> Optional[float]:
        pool = s.loc[s["ttm_days"] > 0].copy()
        if pool.empty: return None
        exp = pool.iloc[(pool["ttm_days"] - target_days).abs().argsort()].head(1)["expiry"].iloc[0]
        slc = s.loc[s["expiry"] == exp]
        if slc.empty: return None
        und = slc["underlying"].median()
        def near(t: str) -> Optional[float]:
            sub = slc.loc[slc["type"] == t]
            if sub.empty: return None
            sub = sub.assign(dist=(sub["strike"] - und).abs()).sort_values("dist")
            iv = sub["mark_iv"].iloc[0]
            return float(iv) if pd.notna(iv) else None
        c_iv = near("C"); p_iv = near("P")
        if c_iv is None and p_iv is None: return None
        if c_iv is None: return p_iv
        if p_iv is None: return c_iv
        return 0.5 * (c_iv + p_iv)

    iv7  = atm_iv(7.0)
    iv30 = atm_iv(30.0)
    if iv7  is not None: out["iv_7d"]  = float(iv7)
    if iv30 is not None: out["iv_30d"] = float(iv30)

    pool = s.loc[s["ttm_days"] > 0]
    if not pool.empty:
        exp = pool.iloc[(pool["ttm_days"] - 30.0).abs().argsort()].head(1)["expiry"].iloc[0]
        slc = s.loc[s["expiry"] == exp]
        if not slc.empty and "delta" in slc.columns:
            calls = slc.loc[s["type"] == "C"].dropna(subset=["delta", "mark_iv"])
            puts  = slc.loc[s["type"] == "P"].dropna(subset=["delta", "mark_iv"])
            if not calls.empty and not puts.empty:
                c = calls.iloc[(calls["delta"] - 0.25).abs().argsort()].head(1)
                p = puts.iloc[(puts["delta"] + 0.25).abs().argsort()].head(1)
                if not c.empty and not p.empty:
                    out["skew_25d"] = float(c["mark_iv"].iloc[0] - p["mark_iv"].iloc[0])
    return out

def collect_options_features_once(ts: pd.Timestamp, cache: MinuteCache) -> Tuple[List[float], List[str]]:
    """
    Take DVOL + (optionally) surface snapshot, update cache, compute features.
    Requires options_data.get_dvol_now() and options_data.surface_snapshot_now().
    """
    from options_data import get_dvol_now, surface_snapshot_now

    # DVOL (30d)
    try:
        dvol = get_dvol_now("BTC")
        if not dvol.empty:
            cache.upsert("dvol_30d", ts, float(dvol["dvol_30d"].iloc[-1]))
    except Exception:
        pass

    # Surface → skew & term IVs
    try:
        surf = surface_snapshot_now("BTC", max_instruments=600)
        feats = _compute_option_terms_from_surface(surf)
        for k in ("skew_25d", "iv_7d", "iv_30d"):
            if k in feats:
                cache.upsert(k, ts, float(feats[k]))
    except Exception:
        pass

    vals, names = extract_options_features_at(cache.as_mapping(), ts, OptionsConfig())
    return vals, names

def _append_csv(path: str, row: Dict[str, Any]):
    import os
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)

if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser(description="Collect options features each minute → CSV.")
    parser.add_argument("--csv", default="data/options_features_1m.csv")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    cache = MinuteCache(max_minutes=7*24*60)
    print(f"[options] writing features to {args.csv}")
    last = None
    while True:
        ts = utc_floor_minute(pd.Timestamp.utcnow())
        if last is None or ts > last:
            vals, names = collect_options_features_once(ts, cache)
            row = {"timestamp": ts.isoformat()}
            row.update({n: v for n, v in zip(names, vals)})
            _append_csv(args.csv, row)
            print(f"[options] {ts.isoformat()} → wrote {len(vals)} feats")
            last = ts
            if args.once: break
            now = pd.Timestamp.utcnow()
            time.sleep(max(0.0, 60 - (now.second + now.microsecond/1e6)) + 0.25)
        else:
            time.sleep(1.0)
