# options_features.py
# Options (DVOL + surface term/skew) features at a specified timestamp.
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

from features_helpers import (
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _delta_last, _clip01,
)


# ------------------------------- Config -------------------------------- #

@dataclass(frozen=True)
class OptionsConfig:
    # DVOL windows
    dvol_z_long: int = 7 * 24 * 60   # ~7d z-score window
    dvol_d60:    int = 60            # 60-minute change

    # Surface selection
    atm_moneyness: float = 0.05      # within ±5% of underlying = "ATM band"
    max_expiries:  int = 4           # use up to 4 nearest expiries for term slope

    # Numerical stabilizers / scalers for tanh-bounding to [-1, 1]
    slope_scale: float = 100.0       # scales IV/day slope before tanh
    skew_scale:  float = 50.0        # scales call-put IV diff (%) before tanh


# -------------------------- CSV → caches helpers ------------------------ #

def _read_dvol_csv(path: Optional[str]) -> pd.Series:
    """
    Expect a time series file: timestamp,dvol
    Returns a float Series indexed by UTC.
    """
    if not path or not os.path.exists(path):
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.Series(dtype=float)

    tcol = None
    for c in df.columns:
        if str(c).lower() == "timestamp":
            tcol = c; break
    if tcol is None:
        # allow index saved in first column
        tcol = df.columns[0]

    if "dvol" not in df.columns:
        return pd.Series(dtype=float)

    idx = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    s = pd.Series(pd.to_numeric(df["dvol"], errors="coerce").values, index=idx)
    s = s.dropna()
    s.index = pd.to_datetime(s.index, utc=True)
    s = s[~s.index.duplicated(keep="last")]
    return s.astype(float).sort_index()


def _read_surface_csv_latest_leq(path: Optional[str], ts: pd.Timestamp) -> pd.DataFrame:
    """
    Expect a wide table: instrument, expiry, strike, mark_iv, underlying, snapshot_ts
    Return the subset of rows at the latest snapshot_ts <= ts. Empty DataFrame if none.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if "snapshot_ts" not in df.columns:
        # try to infer
        return pd.DataFrame()

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["snapshot_ts"])
    df = df.sort_values("snapshot_ts")

    ts = pd.to_datetime(ts, utc=True)
    df_leq = df[df["snapshot_ts"] <= ts]
    if df_leq.empty:
        return pd.DataFrame()

    snap = df_leq["snapshot_ts"].max()
    out = df_leq[df_leq["snapshot_ts"] == snap].copy()

    # Normalize/parse types
    if "expiry" in out.columns:
        out["expiry"] = pd.to_datetime(out["expiry"], utc=True, errors="coerce")
    if "strike" in out.columns:
        out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    if "mark_iv" in out.columns:
        out["mark_iv"] = pd.to_numeric(out["mark_iv"], errors="coerce")
    if "underlying" in out.columns:
        out["underlying"] = pd.to_numeric(out["underlying"], errors="coerce")

    # derive option type if missing (from instrument suffix -C / -P)
    if "type" not in out.columns:
        out["type"] = out.get("instrument", "").astype(str).str[-1].str.upper().where(
            lambda s: s.isin(["C", "P"]), other=pd.NA
        )

    return out


# --------------------------- Feature extraction ------------------------- #

def _minutes_series(series: pd.Series, ts: pd.Timestamp, freq: str = "1min") -> pd.Series:
    """Leakage-safe: take ≤ ts, resample to minute grid with ffill."""
    s = _prep_series(series)
    s = _leq(s, ts)
    return _resample_1m_ffill(s, ts, freq)


def _iv_term_slope(surface: pd.DataFrame, cfg: OptionsConfig) -> float:
    """
    Compute ATM term-structure slope: regress median ATM IV (in %) vs time-to-expiry (days)
    over the nearest cfg.max_expiries expiries. Returns a tanh-bounded value.
    """
    if surface is None or surface.empty:
        return 0.0

    # Need underlying to form ATM band
    if "underlying" not in surface.columns or surface["underlying"].dropna().empty:
        return 0.0
    spot = float(surface["underlying"].median())
    if not np.isfinite(spot) or spot <= 0:
        return 0.0

    # Filter to ATM band
    if "strike" not in surface.columns or "mark_iv" not in surface.columns:
        return 0.0
    m = surface["strike"].between(spot * (1.0 - cfg.atm_moneyness), spot * (1.0 + cfg.atm_moneyness))
    atm = surface.loc[m].copy()
    atm = atm.dropna(subset=["mark_iv"])
    if atm.empty:
        return 0.0

    # Group by expiry, compute median IV
    if "expiry" in atm.columns and atm["expiry"].notna().any():
        # select nearest N expiries
        exps = sorted(pd.to_datetime(atm["expiry"]).dropna().unique())
        if cfg.max_expiries > 0:
            exps = exps[:cfg.max_expiries]
        atm = atm[atm["expiry"].isin(exps)]
        if atm.empty:
            return 0.0

        # time-to-expiry in days
        ref_time = atm["snapshot_ts"].iloc[0] if "snapshot_ts" in atm.columns else pd.Timestamp.utcnow().tz_localize("UTC")
        tte_days = (pd.to_datetime(atm["expiry"]) - pd.to_datetime(ref_time)).dt.total_seconds() / 86400.0
        atm = atm.assign(tte_days=tte_days.values)
        # median IV per expiry
        grp = atm.groupby("expiry", dropna=True)["mark_iv"].median().reset_index()
        if grp.empty:
            return 0.0
        # Build X (tte_days) by matching expiry order
        tte_map = atm.drop_duplicates("expiry").set_index("expiry")["tte_days"]
        grp["tte_days"] = grp["expiry"].map(tte_map)
        grp = grp.dropna(subset=["tte_days", "mark_iv"])
        if len(grp) < 2:
            return 0.0

        # simple slope d(IV%)/d(day)
        x = grp["tte_days"].astype(float).to_numpy()
        y = grp["mark_iv"].astype(float).to_numpy()  # IV is already in percent on Deribit
        # numeric stability: center x
        x0 = x - x.mean()
        denom = np.dot(x0, x0)
        if denom <= 0:
            return 0.0
        slope = float(np.dot(x0, y - y.mean()) / denom)  # % IV per day
        return _clip01(np.tanh(slope / cfg.slope_scale))
    return 0.0


def _iv_skew(surface: pd.DataFrame, cfg: OptionsConfig) -> float:
    """
    Put/Call IV skew near ATM for the nearest expiry:
      skew = median(IV_puts_ATM) - median(IV_calls_ATM)  [in % points], tanh-bounded.
    """
    if surface is None or surface.empty:
        return 0.0
    req_cols = {"instrument", "strike", "mark_iv", "type"}
    if not req_cols.issubset(set(surface.columns)):
        return 0.0
    if "underlying" not in surface.columns or surface["underlying"].dropna().empty:
        return 0.0

    spot = float(surface["underlying"].median())
    if not np.isfinite(spot) or spot <= 0:
        return 0.0

    df = surface.dropna(subset=["mark_iv"]).copy()
    # nearest expiry (if available)
    if "expiry" in df.columns and df["expiry"].notna().any():
        exps = sorted(pd.to_datetime(df["expiry"]).dropna().unique())
        if len(exps) == 0:
            return 0.0
        df = df[df["expiry"] == exps[0]]

    # ATM band
    m = df["strike"].between(spot * (1.0 - cfg.atm_moneyness), spot * (1.0 + cfg.atm_moneyness))
    atm = df.loc[m]
    if atm.empty:
        return 0.0

    calls = atm[atm["type"] == "C"]["mark_iv"].astype(float)
    puts  = atm[atm["type"] == "P"]["mark_iv"].astype(float)
    if calls.empty or puts.empty:
        return 0.0
    # Skew in percentage points
    skew = float(puts.median() - calls.median())
    return _clip01(np.tanh(skew / cfg.skew_scale))


def extract_options_features_at(
    *,
    ts,
    dvol_series: Optional[pd.Series] = None,
    surface_snapshot: Optional[pd.DataFrame] = None,
    cfg: OptionsConfig = OptionsConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Compute options features at timestamp `ts`, using:
      - DVOL minute series (<= ts)
      - One surface snapshot at or before ts (latest ≤ ts)
    Returns (values, names), all in [-1, 1].
    """
    ts = pd.to_datetime(ts, utc=True).floor("min")

    # --- DVOL series features ---
    values: List[float] = []
    names:  List[str] = []

    if dvol_series is None:
        dvol_series = pd.Series(dtype=float)

    dvol_m = _minutes_series(dvol_series, ts, "1min")
    values.extend([
        _z_last(dvol_m, cfg.dvol_z_long),
        _delta_last(dvol_m, cfg.dvol_d60),
    ])
    names.extend(["dvol_z_long", "dvol_d60"])

    # --- Surface snapshot features (only if provided) ---
    slope = _iv_term_slope(surface_snapshot, cfg) if surface_snapshot is not None else 0.0
    skew  = _iv_skew(surface_snapshot, cfg) if surface_snapshot is not None else 0.0
    values.extend([slope, skew])
    names.extend(["iv_term_slope", "iv_atm_skew"])

    # bound to [-1, 1]
    values = [_clip01(float(v)) for v in values]
    return values, names


# --------------------------- Convenience loader -------------------------- #

def build_options_inputs_from_csv(
    *,
    dvol_csv: Optional[str] = None,
    surface_csv: Optional[str] = None,
    ts=None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Load DVOL series and surface snapshot (latest ≤ ts) from CSVs.
    If ts is None, 'last' snapshot_ts is used for surface.
    """
    dvol = _read_dvol_csv(dvol_csv)
    surface = pd.DataFrame()
    if surface_csv:
        t = (pd.Timestamp.utcnow().tz_localize("UTC").floor("min") if ts is None
             else pd.to_datetime(ts, utc=True).floor("min"))
        surface = _read_surface_csv_latest_leq(surface_csv, t)
    return dvol, surface


# --------------------------------- __main__ ------------------------------ #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Options features from DVOL + surface CSVs at a timestamp.")
    ap.add_argument("--ts", default="now", help="ISO8601 or 'now' (UTC)")
    ap.add_argument("--dvol-csv", help="Path to <CUR>_dvol.csv (timestamp,dvol)")
    ap.add_argument("--surface-csv", help="Path to <CUR>_surface.csv (instrument,expiry,strike,mark_iv,underlying,snapshot_ts)")
    ap.add_argument("--atm-pct", type=float, default=0.05, help="ATM band ±pct around underlying (default 0.05 = 5%)")
    ap.add_argument("--max-expiries", type=int, default=4, help="Max expiries for term slope (default 4)")
    ap.add_argument("--slope-scale", type=float, default=100.0, help="Scale for IV/day slope before tanh")
    ap.add_argument("--skew-scale", type=float, default=50.0, help="Scale for IV skew (pp) before tanh")
    ap.add_argument("--save-features", help="If set, append a single-row CSV with features for ts")
    ap.add_argument("--print", action="store_true", help="Print features")
    args = ap.parse_args()

    ts = (pd.Timestamp.utcnow().tz_convert("UTC").floor("min")
          if args.ts == "now" else pd.to_datetime(args.ts, utc=True).floor("min"))

    cfg = OptionsConfig(
        atm_moneyness=args.atm_pct,
        max_expiries=args.max_expiries,
        slope_scale=args.slope_scale,
        skew_scale=args.skew_scale,
    )

    dvol_series, surface_df = build_options_inputs_from_csv(
        dvol_csv=args.dvol_csv,
        surface_csv=args.surface_csv,
        ts=ts,
    )

    vals, names = extract_options_features_at(
        ts=ts,
        dvol_series=dvol_series,
        surface_snapshot=surface_df,
        cfg=cfg,
    )

    if args.print or True:
        print(f"[options] {len(vals)} features at {ts.isoformat()}")
        for n, v in zip(names, vals):
            print(f"  {n:16s}: {v:+.6f}")

    if args.save_features:
        row = {"timestamp": ts.isoformat()}
        row.update({n: float(v) for n, v in zip(names, vals)})
        hdr = not os.path.exists(args.save_features)
        pd.DataFrame([row]).to_csv(args.save_features, mode="a", header=hdr, index=False)
        print(f"Appended features → {args.save_features}")
