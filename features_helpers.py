# features_helpers.py
# One place for common time-series helpers used by feature extractors.

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


__all__ = [
    "utc_floor_minute",
    "_prep_series", "_leq", "_resample_1m_ffill",
    "_z_last", "_delta_last", "_ret_last", "_bp_change_last",
    "_basis_change", "_sum_window", "_clip01",
    "_safe_last", "_logret_series", "_rolling_mean_std", "_zscore",
    "_realized_vol_from_r1", "_trend_slope", "_entropy_proxy", "_acf1", "_rsi",
]


# ----------------------------- time helpers ----------------------------- #

def utc_floor_minute(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True)
    if ts.tz is None:
        ts = ts.tz_convert("UTC")
    return ts.floor("min")


# ----------------------------- series hygiene ----------------------------- #

def _prep_series(x) -> pd.Series:
    """
    Normalize inputs to a UTC, float Series sorted by index.
    Accepts Series or single-column DataFrame; returns empty Series if invalid.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame) and x.shape[1] >= 1:
        s = x.iloc[:, 0]
    else:
        return pd.Series(dtype=float)

    s.index = pd.to_datetime(s.index, utc=True)
    s = s.sort_index()
    s = pd.to_numeric(s, errors="coerce")
    return s.dropna().astype(float)


def _leq(s: pd.Series, ts) -> pd.Series:
    """Slice series to values <= ts (inclusive)."""
    if s.empty:
        return s
    return s.loc[:utc_floor_minute(ts)]


def _resample_1m_ffill(s: pd.Series, ts, freq: str = "1min") -> pd.Series:
    """
    Reindex to a continuous per-minute grid from first obs to ts and ffill.
    Keeps ≤ ts only (leakage-safe).
    """
    if s.empty:
        return s
    ts = utc_floor_minute(ts)
    start = s.index[0].floor(freq)
    idx = pd.date_range(start, ts, freq=freq, tz="UTC")
    return s.reindex(idx).ffill()


# ----------------------------- small transforms ----------------------------- #

def _z_last(s: pd.Series, w: int, min_frac: float = 0.1) -> float:
    """
    Rolling z-score of last value over window w.
    Uses min_periods ≈ max(5, w*min_frac). tanh-bounded output in [-1,1].
    """
    if s.empty or w <= 1:
        return 0.0
    mp = max(5, int(w * min_frac))
    roll = s.rolling(w, min_periods=mp)
    mu = roll.mean()
    sd = roll.std()
    z = (s - mu) / (sd + 1e-9)
    return float(np.tanh(z.iloc[-1])) if not z.empty else 0.0


def _delta_last(s: pd.Series, minutes: int) -> float:
    """Last value minus value minutes ago, tanh-bounded."""
    if s.empty or minutes <= 0:
        return 0.0
    t1 = s.index[-1]
    t0 = t1 - pd.Timedelta(minutes=minutes)
    prev = s.loc[:t0]
    if prev.empty:
        return 0.0
    return float(np.tanh(float(s.iloc[-1] - prev.iloc[-1])))


def _ret_last(s: pd.Series, minutes: int) -> float:
    """Log return over `minutes`: log(s_t / s_{t-min}), tanh-bounded."""
    if s.empty or minutes <= 0:
        return 0.0
    t1 = s.index[-1]
    t0 = t1 - pd.Timedelta(minutes=minutes)
    prev = s.loc[:t0]
    if prev.empty:
        return 0.0
    p0 = float(prev.iloc[-1])
    p1 = float(s.iloc[-1])
    if p0 <= 0 or p1 <= 0:
        return 0.0
    return float(np.tanh(np.log(p1 / p0)))


def _bp_change_last(s: pd.Series, minutes: int) -> float:
    """Basis points change over `minutes`: 10_000 * (p_t/p_{t-min} - 1), tanh-bounded/100 to keep scale."""
    if s.empty or minutes <= 0:
        return 0.0
    t1 = s.index[-1]
    t0 = t1 - pd.Timedelta(minutes=minutes)
    prev = s.loc[:t0]
    if prev.empty:
        return 0.0
    p0 = float(prev.iloc[-1])
    p1 = float(s.iloc[-1])
    if p0 == 0:
        return 0.0
    bps = 10_000.0 * (p1 / p0 - 1.0)
    # scale down before tanh to avoid saturating too often
    return float(np.tanh(bps / 100.0))


def _basis_change(perp: pd.Series, spot: pd.Series, minutes: int, mode: str = "bps") -> float:
    """
    Change in perp-spot basis over `minutes`.
    basis = (perp - spot) / spot  (ratio). If mode='bps', return basis change in bps (tanh/100).
    """
    if perp.empty or spot.empty or minutes <= 0:
        return 0.0
    # align indices
    s = pd.concat([perp.rename("perp"), spot.rename("spot")], axis=1).dropna()
    if s.empty:
        return 0.0
    s["basis"] = (s["perp"] - s["spot"]) / (s["spot"] + 1e-9)
    t1 = s.index[-1]
    t0 = t1 - pd.Timedelta(minutes=minutes)
    prev = s.loc[:t0]
    if prev.empty:
        return 0.0
    b0 = float(prev["basis"].iloc[-1])
    b1 = float(s["basis"].iloc[-1])
    if mode == "bps":
        bps = 10_000.0 * (b1 - b0)
        return float(np.tanh(bps / 100.0))
    else:
        return float(np.tanh(b1 - b0))


def _sum_window(s: pd.Series, minutes: int) -> float:
    """Sum of values over the last `minutes` (inclusive), tanh-bounded by a soft scale."""
    if s.empty or minutes <= 0:
        return 0.0
    t1 = s.index[-1]
    t0 = t1 - pd.Timedelta(minutes=minutes)
    wsum = float(s.loc[t0:].sum())
    # soft scaling to keep in [-1,1] without a hard-coded domain
    scale = 1.0 + abs(wsum) / 10_000.0
    return float(np.tanh(wsum / scale))


def _clip01(x: float) -> float:
    """Hard clip to [-1,1]."""
    return float(max(-1.0, min(1.0, x)))


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
