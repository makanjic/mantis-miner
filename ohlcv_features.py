# ohlcv_features.py
# Extracts universal, leakage-safe features (bounded in [-1, 1]) for any token
# using the ENTIRE df up to its LAST timestamp (no external timestamp needed).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Mapping

import numpy as np
import pandas as pd

from time_features import calendar_time_vector  # from your time_features.py


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


# ---------------------------- helpers ---------------------------- #

def _safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    return float(arr[-1]) if arr.size else float(default)

def _logret_series(close: np.ndarray) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=float)
    return np.diff(np.log(close), prepend=np.log(close[0]))

def _rolling_mean_std(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    s = pd.Series(x)
    m = s.rolling(w, min_periods=max(5, w//10)).mean().to_numpy()
    v = s.rolling(w, min_periods=max(5, w//10)).var().to_numpy()
    std = np.sqrt(np.maximum(1e-12, v))
    return m, std

def _zscore(x: np.ndarray, w: int) -> np.ndarray:
    mu, sd = _rolling_mean_std(x, w)
    return (x - mu) / (sd + 1e-9)

def _realized_vol_from_r1(r1: np.ndarray, w: int) -> np.ndarray:
    # sqrt of rolling sum of r1^2 over window w
    if r1.size == 0:
        return np.array([], dtype=float)
    r2 = r1**2
    s = pd.Series(r2).rolling(w, min_periods=max(5, w//10)).sum().to_numpy()
    return np.sqrt(np.maximum(1e-12, s))

def _trend_slope(price: np.ndarray, w: int) -> float:
    # standardized OLS slope over the last w minutes, then tanh-bound
    if price.size < w + 5:
        return 0.0
    y = price[-w:]
    t = np.arange(w, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    slope = (t @ y) / (w - 1)
    return float(np.tanh(slope))  # already in [-1, 1]

def _entropy_proxy(r1: np.ndarray, w: int) -> float:
    # 1 - |ACF1| over the last w minutes → map to [-1,1]
    if r1.size < w + 2:
        return 0.0
    window = r1[-w:]
    if np.std(window) < 1e-9:
        val = 0.0
    else:
        acf1 = np.corrcoef(window[:-1], window[1:])[0, 1]
        val = 1.0 - abs(float(acf1))           # in [0,1]
    return float(2.0 * val - 1.0)              # map to [-1,1]

def _acf1(r1: np.ndarray, w: int) -> float:
    if r1.size < w + 2:
        return 0.0
    window = r1[-w:]
    if np.std(window) < 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(window[:-1], window[1:])[0, 1], -1.0, 1.0))

def _rsi(close: np.ndarray, period: int = 14) -> float:
    # map RSI ∈ [0,100] to ~[-1,1] via (RSI-50)/12 and tanh
    if close.size < period + 5:
        return 0.0
    diff = np.diff(close, prepend=close[0])
    up = np.clip(diff, 0, None)
    dn = -np.clip(diff, None, 0)
    k = 2.0 / (period + 1.0)
    def _ema(x):
        out = np.zeros_like(x, dtype=float); acc = 0.0
        for i, xi in enumerate(x):
            acc = k*xi + (1-k)*acc if i else xi
            out[i] = acc
        return out
    rs = _ema(up)[-1] / (_ema(dn)[-1] + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.tanh((rsi - 50.0) / 12.0))  # squashed to [-1,1]


# ---------------- main feature extractor (no timestamp arg) ---------------

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
    # Calendar/time block (derived internally from df's last timestamp)
    "dow_sin", "dow_cos", "dom_sin", "dom_cos", "eom_prox", "moy_sin", "moy_cos",
)

def extract_ohlcv_features(
    df_1m: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Extract universal features using ALL data up to the DataFrame's LAST timestamp.
    df_1m: UTC-indexed minute bars with columns ['open','high','low','close','volume'].
    Returns (values, names) with values[i] ∈ [-1, 1].
    """
    if df_1m.empty:
        return [0.0] * len(OHLCV_FEATURE_ORDER), list(OHLCV_FEATURE_ORDER)

    hist: pd.DataFrame = df_1m.sort_index().copy()

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
            acc = k*xi + (1-k)*acc if i else xi
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

    # --- calendar/time block (from last index)
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
    for n, v in zip(cal_names, cal_vals):
        values_map[n] = float(v)

    vals = [float(np.clip(values_map[name], -1.0, 1.0)) for name in OHLCV_FEATURE_ORDER]
    names = list(OHLCV_FEATURE_ORDER)
    return vals, names


# ----------- batch convenience: multiple tokens (each uses its own last ts) -----------

def extract_ohlcv_features_for_assets(
    data_by_symbol: Mapping[str, pd.DataFrame],
    cfg: FeatureConfig = FeatureConfig(),
) -> Dict[str, Tuple[List[float], List[str]]]:
    """
    Compute ohlcv features for a dict of {symbol: df_1m}, each at its OWN last index.
    Returns {symbol: (values, names)}; names are identical across symbols.
    """
    out: Dict[str, Tuple[List[float], List[str]]] = {}
    for sym, df in data_by_symbol.items():
        out[sym] = extract_ohlcv_features(df, cfg)
    return out


# ------------------------------ __main__ test ---------------------------- #

if __name__ == "__main__":
    import argparse
    import os

    def _load_csv(path: str) -> pd.DataFrame:
        """
        Load a 1m OHLCV CSV with columns:
          timestamp, open, high, low, close, volume
        Returns a UTC-indexed, minute-sorted DataFrame.
        """
        df = pd.read_csv(path)
        # Column normalization
        cols = {c.lower(): c for c in df.columns}
        needed = ["timestamp", "open", "high", "low", "close", "volume"]
        for k in needed:
            if k not in cols:
                raise ValueError(f"CSV {path} missing column: {k}")
        # Parse timestamp to UTC
        ts = pd.to_datetime(df[cols["timestamp"]], utc=True)
        out = pd.DataFrame(
            {
                "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
                "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
                "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
                "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
                "volume": pd.to_numeric(df[cols["volume"]], errors="coerce"),
            },
            index=ts,
        )
        out = out.sort_index().dropna()
        return out

    def _make_synth_minutes(n_minutes: int = 3 * 24 * 60, start_price: float = 30000.0) -> pd.DataFrame:
        """
        Generate synthetic 1-minute OHLCV to test the extractor without data files.
        """
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
        low = close * (1 - 0.5 * spread / close)
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = (rng.lognormal(mean=8.0, sigma=0.35, size=n_minutes) *
                  (1.0 + 0.25 * np.sin(2 * np.pi * (minutes % 1440) / 1440.0)))

        idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=n_minutes, freq="1min")
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        return df

    parser = argparse.ArgumentParser(description="Test ohlcv feature extractor (no timestamp arg)")
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Path to 1m OHLCV CSV (timestamp,open,high,low,close,volume). "
             "Repeat flag to load multiple symbols.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="Symbol name for each --csv (same order). If omitted, filename stems are used.",
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
        help="Also compute features for all loaded symbols (each at its own last timestamp).",
    )
    args = parser.parse_args()

    # Load data
    data_by_symbol: dict[str, pd.DataFrame] = {}
    if args.csv:
        if args.symbol and len(args.symbol) != len(args.csv):
            raise SystemExit("If --symbol is used, provide the same number as --csv files.")
        for i, path in enumerate(args.csv):
            sym = args.symbol[i] if i < len(args.symbol) else os.path.splitext(os.path.basename(path))[0]
            df = _load_csv(path)
            data_by_symbol[sym] = df
            print(f"Loaded {sym}: {len(df):,} rows from {path}")
    else:
        print("No CSV provided; generating synthetic BTC data...")
        data_by_symbol["BTC"] = _make_synth_minutes()
        print(f"Synth BTC rows: {len(data_by_symbol['BTC']):,}")

    cfg = FeatureConfig(eom_window_hours=args.eom_window_hours)

    # Single symbol demo (first symbol)
    sym0 = next(iter(data_by_symbol))
    vals, names = extract_ohlcv_features(data_by_symbol[sym0], cfg)
    last_ts = data_by_symbol[sym0].index[-1]
    print(f"\n=== Features for {sym0} at its LAST timestamp ({pd.Timestamp(last_ts).isoformat()}) ===")
    for n, v in zip(names, vals):
        print(f"{n:>18}: {v:+.6f}")
    print(f"Count: {len(vals)} features")

    # Range check
    vmin, vmax = min(vals), max(vals)
    print(f"\nRange check: min={vmin:+.6f}, max={vmax:+.6f} (should be within [-1, 1])")
    if vmin < -1.000001 or vmax > 1.000001:
        print("WARNING: Out-of-bounds value detected (should not happen).")

    # Batch demo
    if args.print_batch and len(data_by_symbol) > 1:
        print("\n=== Batch features (first 8 names shown; each symbol uses its own last timestamp) ===")
        batch = extract_ohlcv_features_for_assets(data_by_symbol, cfg)
        head_names = names[:8]
        for sym, (v, n) in batch.items():
            short = " ".join(f"{x:+.4f}" for x in v[:8])
            last_ts_sym = data_by_symbol[sym].index[-1]
            print(f"{sym:>8} @ {pd.Timestamp(last_ts_sym).strftime('%Y-%m-%d %H:%M')}  {short} ... ({len(v)} dims total)")

    print("\nOK ✓  (ohlcv feature extraction completed)")
