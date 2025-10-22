# ohlcv_features.py
# Extracts universal, leakage-safe features (bounded in [-1, 1]) for any token.
# Can compute at a SPECIFIED timestamp (<= ts) OR at the DataFrame's LAST timestamp.
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Mapping

import numpy as np
import pandas as pd

from time_features import calendar_time_vector  # from your time_features.py

from features_helpers import (
    _safe_last, _logret_series, _rolling_mean_std, _zscore,
    _realized_vol_from_r1, _trend_slope, _entropy_proxy, _acf1, _rsi,
)

@dataclass(frozen=True)
class FeatureConfig:
    # rolling lookbacks in minutes
    ret_horizons: Tuple[int, ...] = (1, 2, 5, 15, 60)
    rv_windows:   Tuple[int, ...] = (15, 30, 60)
    trend_windows: Tuple[int, ...] = (240, 720)
    entropy_window: int = 240
    acf_window: int = 30
    vol_z_window: int = 360   # ~6h for volume z-score (seasonality proxy)

    # z-score standardization window for generic features (minutes)
    zscore_window: int = 1440  # ~1 day

    # end-of-month proximity width (hours)
    eom_window_hours: int = 36

# ---------------- feature list (unchanged) ---------------

OHLCV_FEATURE_ORDER: Tuple[str, ...] = (
    # Momentum / returns (z-scored then tanh)
    "ret_1m_z", "ret_2m_z", "ret_5m_ema_z", "ret_15m_ema_z", "ret_60m_ema_z",
    # Volatility / dispersion (tanh of z-scores)
    "rv_15m_z", "rv_30m_z", "rv_60m_z",
    "range_pct_z", "co_pct_z",
    # Participation
    "vol_z_6h_tanh",
    # Regime
    "trend_slope_240", "trend_slope_720",
    "entropy_240", "acf1_30m",
    # Oscillator
    "rsi14_tanh",
    # (Calendar/time block computed but not included in ORDER by design.)
)

# ---------------- core computation ---------------

def _extract_from_hist(hist: pd.DataFrame, cfg: FeatureConfig) -> Tuple[List[float], List[str]]:
    """
    hist: UTC-indexed minute bars up to the point you want features for.
    Returns (vals, names) with vals in [-1, 1].
    """
    if hist.empty:
        return [0.0] * len(OHLCV_FEATURE_ORDER), list(OHLCV_FEATURE_ORDER)

    hist = hist.sort_index()
    o = hist["open"].to_numpy(dtype=float)
    h = hist["high"].to_numpy(dtype=float)
    l = hist["low"].to_numpy(dtype=float)
    c = hist["close"].to_numpy(dtype=float)
    v = hist["volume"].to_numpy(dtype=float)

    # --- momentum & returns ---
    r1 = _logret_series(c)

    def _ema_series(x: np.ndarray, span: int) -> np.ndarray:
        if x.size == 0:
            return x
        k = 2.0 / (span + 1.0)
        out = np.zeros_like(x, dtype=float)
        acc = 0.0
        for i, xi in enumerate(x):
            acc = k * xi + (1 - k) * acc if i else xi
            out[i] = acc
        return out

    ret_1m_z = _safe_last(np.tanh(_zscore(r1, cfg.zscore_window)))
    ret_2m = np.concatenate([[0.0], r1[1:] + r1[:-1]]) if r1.size else np.array([0.0])
    ret_2m_z = _safe_last(np.tanh(_zscore(ret_2m, cfg.zscore_window)))
    ret_5m_ema_z  = _safe_last(np.tanh(_zscore(_ema_series(r1, 5),  cfg.zscore_window)))
    ret_15m_ema_z = _safe_last(np.tanh(_zscore(_ema_series(r1, 15), cfg.zscore_window)))
    ret_60m_ema_z = _safe_last(np.tanh(_zscore(_ema_series(r1, 60), cfg.zscore_window)))

    # --- realized vol (rolling) ---
    rv_15m_z = _safe_last(np.tanh(_zscore(_realized_vol_from_r1(r1, 15), cfg.zscore_window)))
    rv_30m_z = _safe_last(np.tanh(_zscore(_realized_vol_from_r1(r1, 30), cfg.zscore_window)))
    rv_60m_z = _safe_last(np.tanh(_zscore(_realized_vol_from_r1(r1, 60), cfg.zscore_window)))

    # --- dispersion / range ---
    range_pct = (h - l) / np.maximum(1e-9, c)
    co_pct = (c - o) / np.maximum(1e-9, c)
    range_pct_z = _safe_last(np.tanh(_zscore(range_pct, cfg.zscore_window)))
    co_pct_z    = _safe_last(np.tanh(_zscore(co_pct,    cfg.zscore_window)))

    # --- participation / volume ---
    vol_z = _zscore(v, cfg.vol_z_window)
    vol_z_6h_tanh = _safe_last(np.tanh(vol_z))

    # --- regime ---
    trend_slope_240 = _trend_slope(c, cfg.trend_windows[0])
    trend_slope_720 = _trend_slope(c, cfg.trend_windows[1])
    entropy_240     = _entropy_proxy(r1, cfg.entropy_window)
    acf1_30m        = _acf1(r1, cfg.acf_window)

    # --- oscillator ---
    rsi14_tanh = _rsi(c, 14)

    # --- calendar/time block (from last index used) ---
    last_ts = hist.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    cal_vals, cal_names = calendar_time_vector(pd.Timestamp(last_ts).timestamp(),
                                               window_hours=cfg.eom_window_hours)

    values_map: Dict[str, float] = {
        "ret_1m_z": ret_1m_z,
        "ret_2m_z": ret_2m_z,
        "ret_5m_ema_z": ret_5m_ema_z,
        "ret_15m_ema_z": ret_15m_ema_z,
        "ret_60m_ema_z": ret_60m_ema_z,
        "rv_15m_z": rv_15m_z,
        "rv_30m_z": rv_30m_z,
        "rv_60m_z": rv_60m_z,
        "range_pct_z": range_pct_z,
        "co_pct_z": co_pct_z,
        "vol_z_6h_tanh": vol_z_6h_tanh,
        "trend_slope_240": trend_slope_240,
        "trend_slope_720": trend_slope_720,
        "entropy_240": entropy_240,
        "acf1_30m": acf1_30m,
        "rsi14_tanh": rsi14_tanh,
    }
    # Calendar features computed for potential future use; not in ORDER by design.
    for n, v in zip(cal_names, cal_vals):
        values_map[n] = float(v)

    vals = [float(np.clip(values_map[name], -1.0, 1.0)) for name in OHLCV_FEATURE_ORDER]
    names = list(OHLCV_FEATURE_ORDER)
    return vals, names

# ---------------- public APIs (supports *at ts*) ---------------

def extract_ohlcv_features(
    df_1m: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> Tuple[List[float], List[str]]:
    """Use ALL data up to the DataFrame's LAST timestamp."""
    return _extract_from_hist(df_1m, cfg)

def extract_ohlcv_features_at(
    df_1m: pd.DataFrame,
    ts,
    cfg: FeatureConfig = FeatureConfig(),
) -> Tuple[List[float], List[str]]:
    """Compute features at a SPECIFIC timestamp `ts` (UTC), using only rows with index <= ts."""
    ts = pd.to_datetime(ts, utc=True).floor("min")
    hist = df_1m.loc[df_1m.index <= ts]
    return _extract_from_hist(hist, cfg)

# ----------- batch convenience -----------

def extract_ohlcv_features_for_assets(
    data_by_symbol: Mapping[str, pd.DataFrame],
    cfg: FeatureConfig = FeatureConfig(),
    ts: Optional[pd.Timestamp] = None,
) -> Dict[str, Tuple[List[float], List[str]]]:
    """
    If ts is None → each symbol uses its own last index.
    Else           → compute for the provided ts for every symbol (series truncated to <= ts).
    """
    out: Dict[str, Tuple[List[float], List[str]]] = {}
    if ts is None:
        for sym, df in data_by_symbol.items():
            out[sym] = extract_ohlcv_features(df, cfg)
    else:
        ts = pd.to_datetime(ts, utc=True).floor("min")
        for sym, df in data_by_symbol.items():
            out[sym] = extract_ohlcv_features_at(df, ts, cfg)
    return out

# ------------------------------ CSV I/O helpers ---------------------------- #

def _load_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    Load a 1m OHLCV CSV with columns: timestamp, open, high, low, close, volume.
    Returns a UTC-indexed, minute-sorted DataFrame.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    needed = ["timestamp", "open", "high", "low", "close", "volume"]
    for k in needed:
        if k not in cols:
            raise ValueError(f"CSV {path} missing column: {k}")
    ts = pd.to_datetime(df[cols["timestamp"]], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "open":   pd.to_numeric(df[cols["open"]],   errors="coerce").to_numpy(dtype=float),
            "high":   pd.to_numeric(df[cols["high"]],   errors="coerce").to_numpy(dtype=float),
            "low":    pd.to_numeric(df[cols["low"]],    errors="coerce").to_numpy(dtype=float),
            "close":  pd.to_numeric(df[cols["close"]],  errors="coerce").to_numpy(dtype=float),
            "volume": pd.to_numeric(df[cols["volume"]], errors="coerce").to_numpy(dtype=float),
        },
        index=ts,
    )
    out = out.sort_index().dropna()
    return out

def _find_ohlcv_path(load_dir: str, symbol: str) -> Optional[str]:
    """
    Look for <SYMBOL>_1m.csv first, then <SYMBOL>.csv in load_dir.
    """
    cand1 = os.path.join(load_dir, f"{symbol}_1m.csv")
    cand2 = os.path.join(load_dir, f"{symbol}.csv")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    return None

def _find_feature_path(save_dir: str, symbol: str) -> str:
    """
    Default feature file: <SYMBOL>_features.csv. If <SYMBOL>.csv already exists, use it for continuity.
    """
    cand_existing_plain = os.path.join(save_dir, f"{symbol}.csv")
    cand_default = os.path.join(save_dir, f"{symbol}_features.csv")
    return cand_existing_plain if os.path.exists(cand_existing_plain) else cand_default

def _append_or_overwrite_feature_file(path: str, new_rows: pd.DataFrame, append: bool) -> None:
    """
    Append with dedup (by 'timestamp') if append=True; otherwise overwrite.
    """
    if append and os.path.exists(path):
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()
        if not old.empty:
            # robust parse + concat
            if "timestamp" in old.columns:
                old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
            merged = pd.concat([old, new_rows], ignore_index=True)
        else:
            merged = new_rows.copy()

        # normalize timestamp, drop dups, sort
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        merged = merged.dropna(subset=["timestamp"])
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp")
        # write back (ISO format)
        merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        merged.to_csv(path, index=False)
        print(f"Appended {len(new_rows)} row(s) → {path} (now {len(merged)} total)")
    else:
        # overwrite
        out = new_rows.copy()
        out = out.dropna(subset=["timestamp"])
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
        out = out.sort_values("timestamp")
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(path, index=False)
        print(f"Wrote {len(out)} row(s) → {path}")

# ------------------------------ __main__ ---------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OHLCV feature extractor with dir-based CSV I/O.")
    parser.add_argument(
        "--symbol",
        action="append",
        default=["BTC","ETH","EURUSD","GBPUSD","CADUSD","NZDUSD","CHFUSD","XAUUSD","XAGUSD"],
        help="Symbol(s) to process. Repeat flag to pass multiple. Default: all 9.",
    )
    parser.add_argument(
        "--load-from-csv",
        nargs="?", const=".",
        default=None,
        help="Directory to read per-symbol OHLCV CSVs (expects <SYMBOL>_1m.csv or <SYMBOL>.csv). If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--ts",
        default="last",
        help="Timestamp to compute features at: ISO string, 'now' (UTC), or 'last' (each DF's own last index).",
    )
    parser.add_argument(
        "--eom-window-hours",
        type=int,
        default=36,
        help="EOM proximity window width in hours (default: 36).",
    )
    parser.add_argument(
        "--print-batch",
        action="store_true",
        help="Print a short preview for each symbol.",
    )
    parser.add_argument(
        "--save-to-csv",
        nargs="?", const=".",
        default=None,
        help="Directory to write per-symbol feature CSVs (<SYMBOL>_features.csv). If omitted, nothing is saved. If provided with no path, uses current directory.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="When saving, append and de-duplicate by timestamp. Otherwise overwrite.",
    )
    args = parser.parse_args()

    # Load OHLCV per symbol
    data_by_symbol: dict[str, pd.DataFrame] = {}

    if args.load_from_csv:
        if not os.path.isdir(args.load_from_csv):
            raise SystemExit(f"--load-from-csv path not a directory: {args.load_from_csv}")
        for sym in args.symbol:
            p = _find_ohlcv_path(args.load_from_csv, sym)
            if not p:
                print(f"[warn] No OHLCV CSV for {sym} under {args.load_from_csv} (looked for {sym}_1m.csv / {sym}.csv). Skipping.")
                continue
            df = _load_ohlcv_csv(p)
            data_by_symbol[sym] = df
            print(f"Loaded {sym}: {len(df):,} rows from {p}")
    else:
        # Generate synthetic data for each symbol so you can test end-to-end
        def _make_synth_minutes(n_minutes: int = 3 * 24 * 60, start_price: float = 30000.0) -> pd.DataFrame:
            rng = np.random.default_rng(42)
            minutes = np.arange(n_minutes)
            vol_intraday = 0.0007 + 0.0006 * np.sin(2 * np.pi * (minutes % 1440) / 1440.0)
            r = rng.normal(0.0, vol_intraday)
            price = np.empty(n_minutes)
            price[0] = start_price
            for i in range(1, n_minutes):
                price[i] = price[i - 1] * np.exp(r[i])
            close = price
            spread = np.maximum(1e-6, 0.0002 * close)
            high = close * (1 + 0.5 * spread / close)
            low  = close * (1 - 0.5 * spread / close)
            open_ = np.concatenate([[close[0]], close[:-1]])
            volume = (rng.lognormal(mean=8.0, sigma=0.35, size=n_minutes) *
                      (1.0 + 0.25 * np.sin(2 * np.pi * (minutes % 1440) / 1440.0)))
            idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=n_minutes, freq="min")
            df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
            return df

        for sym in args.symbol:
            data_by_symbol[sym] = _make_synth_minutes()
            print(f"Synth {sym}: {len(data_by_symbol[sym]):,} rows")

    if not data_by_symbol:
        raise SystemExit("No data loaded/generated. Nothing to do.")

    cfg = FeatureConfig(eom_window_hours=args.eom_window_hours)

    # Resolve ts selection
    ts_opt: Optional[pd.Timestamp]
    if args.ts == "last":
        ts_opt = None
    elif args.ts == "now":
        ts_opt = pd.Timestamp.now(tz="UTC").floor("min")
    else:
        ts_opt = pd.to_datetime(args.ts, utc=True).floor("min")

    # Compute & print a preview for the first symbol
    sym0 = next(iter(data_by_symbol))
    if ts_opt is None:
        vals, names = extract_ohlcv_features(data_by_symbol[sym0], cfg)
        ref_ts = pd.Timestamp(data_by_symbol[sym0].index[-1]).isoformat()
    else:
        vals, names = extract_ohlcv_features_at(data_by_symbol[sym0], ts_opt, cfg)
        ref_ts = ts_opt.isoformat()
    print(f"\n=== Features for {sym0} @ {ref_ts} ===")
    for n, v in zip(names, vals):
        print(f"{n:>18}: {v:+.6f}")
    print(f"Count: {len(vals)} features")
    vmin, vmax = min(vals), max(vals)
    print(f"Range check: min={vmin:+.6f}, max={vmax:+.6f} (should be within [-1, 1])")

    # Optional batch print
    if args.print_batch and len(data_by_symbol) > 1:
        print("\n=== Batch features (first 8 dims shown) ===")
        batch = extract_ohlcv_features_for_assets(data_by_symbol, cfg, ts=ts_opt)
        for sym, (v, n) in batch.items():
            short = " ".join(f"{x:+.4f}" for x in v[:8])
            ref = (data_by_symbol[sym].index[-1].strftime("%Y-%m-%d %H:%M")
                   if ts_opt is None else ts_opt.strftime("%Y-%m-%d %H:%M"))
            print(f"{sym:>8} @ {ref}  {short} ... ({len(v)} dims total)")

    # Save per-symbol features if requested
    if args.save_to_csv:
        os.makedirs(args.save_to_csv, exist_ok=True)
        rows_by_symbol: Dict[str, pd.DataFrame] = {}

        if ts_opt is None:
            # each symbol at its own last index
            for sym, df in data_by_symbol.items():
                v, n = extract_ohlcv_features(df, cfg)
                t_ref = pd.Timestamp(df.index[-1]).floor("min")
                row = {"timestamp": t_ref, "symbol": sym}
                row.update({name: float(val) for name, val in zip(n, v)})
                rows_by_symbol[sym] = pd.DataFrame([row])
        else:
            # same ts for all symbols
            for sym, df in data_by_symbol.items():
                v, n = extract_ohlcv_features_at(df, ts_opt, cfg)
                row = {"timestamp": ts_opt, "symbol": sym}
                row.update({name: float(val) for name, val in zip(n, v)})
                rows_by_symbol[sym] = pd.DataFrame([row])

        # write each symbol to its own file with dedup if append=True
        for sym, out_df in rows_by_symbol.items():
            path = _find_feature_path(args.save_to_csv, sym)
            _append_or_overwrite_feature_file(path, out_df, append=args.append)

    print("\nOK ✓  (ohlcv feature extraction completed)")
