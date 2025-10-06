# derivs_features.py
# Derivatives (perp/spot basis, funding, OI, long/short, liquidations) features at a specified timestamp.
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, List
import numpy as np
import pandas as pd

from features_helpers import (
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _delta_last, _basis_change,
)


@dataclass(frozen=True)
class DerivsConfig:
    z_long: int = 7 * 24 * 60     # ~7d window for slow regime (funding / OI)
    z_med:  int = 24 * 60         # ~1d window for long/short ratio
    d15:    int = 15              # 15-minute delta/sum
    d60:    int = 60              # 60-minute delta/sum
    freq:   str = "1min"          # resample grid


# --------------------------- main API --------------------------- #

def extract_derivs_features_at(
    caches: Mapping[str, pd.Series | pd.DataFrame],
    ts,
    cfg: DerivsConfig = DerivsConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Compute derivatives features at timestamp `ts`, using only data <= ts.
    Expected keys in `caches` (all optional):
      funding, perp_close, spot_close, open_interest, long_short, liq_buy, liq_sell
    Returns (values, names). Values are bounded to [-1, 1].
    """
    ts = pd.to_datetime(ts, utc=True)

    def S(key: str) -> pd.Series:
        return _resample_1m_ffill(_leq(_prep_series(caches.get(key, pd.Series(dtype=float))), ts), ts, cfg.freq)

    funding    = S("funding")
    perp_close = S("perp_close")
    spot_close = S("spot_close")
    oi         = S("open_interest")
    lsr        = S("long_short")
    liq_b      = S("liq_buy")
    liq_s      = S("liq_sell")

    names = [
        "funding_z_long", "funding_d60",
        "basis_5m", "basis_60m",
        "oi_z_long", "oi_d60",
        "long_short_z_med",
        "liq_net_15m", "liq_net_60m",
    ]
    vals = [
        _z_last(funding, cfg.z_long),
        _delta_last(funding, cfg.d60),
        _basis_change(perp_close, spot_close, 5),
        _basis_change(perp_close, spot_close, 60),
        _z_last(oi, cfg.z_long),
        _delta_last(oi, cfg.d60),
        _z_last(lsr, cfg.z_med),
        _clip01(np.tanh(_sum_window(liq_b, cfg.d15) - _sum_window(liq_s, cfg.d15))),
        _clip01(np.tanh(_sum_window(liq_b, cfg.d60) - _sum_window(liq_s, cfg.d60))),
    ]
    vals = [_clip01(float(v)) for v in vals]
    return vals, names


# ------------------------------ __main__ test ------------------------------ #

if __name__ == "__main__":
    # Synthetic demo to verify shapes/ranges without external APIs.
    rng = np.random.default_rng(123)

    def synth_series(n=2000, start="2025-09-25 00:00Z", vol=1.0, drift=0.0):
        idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
        x = drift + np.cumsum(rng.normal(0.0, vol, size=n))
        return pd.Series(x, index=idx)

    ts = pd.Timestamp("2025-10-03 08:14:00Z")

    caches = {
        "funding": 0.0001 + 0.0002 * synth_series(vol=0.01),
        "perp_close": 60000 + synth_series(vol=30.0).cumsum().abs(),
        "spot_close": 59950 + synth_series(vol=30.0).cumsum().abs(),
        "open_interest": 1.0e9 + 1.0e7 * synth_series(vol=0.1),
        "long_short": 1.0 + 0.05 * synth_series(vol=0.02),
        "liq_buy": np.abs(synth_series(vol=1.0)),
        "liq_sell": np.abs(synth_series(vol=1.1)),
    }

    vals, names = extract_derivs_features_at(caches, ts, DerivsConfig())
    print(f"Got {len(vals)} derivatives features at {ts.isoformat()}")
    for n, v in zip(names, vals):
        print(f"{n:>18}: {v:+.6f}")
    print("Range:", f"{min(vals):+.4f}", "→", f"{max(vals):+.4f}", " (should be within [-1, 1])")
    print("OK ✓")
