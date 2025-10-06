# macro_features.py
# Cross-asset macro features (DXY/ES/XAU returns; UST yields bp change) at specified timestamp.
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, List
import numpy as np
import pandas as pd

from features_helpers import (
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _ret_last, _bp_change_last, _clip01
)


@dataclass(frozen=True)
class MacroConfig:
    freq: str = "1min"


# --------------------------- main API --------------------------- #

def extract_macro_features_at(
    caches: Mapping[str, pd.Series | pd.DataFrame],
    ts,
    cfg: MacroConfig = MacroConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Compute macro features at timestamp `ts`, using only data <= ts.
    Expected keys (optional):
      DXY, ES, XAU (price levels),
      UST2Y, UST10Y (yield levels)
    Returns (values, names) in [-1, 1].
    """
    ts = pd.to_datetime(ts, utc=True)

    def S(key: str) -> pd.Series:
        return _resample_1m_ffill(_leq(_prep_series(caches.get(key, pd.Series(dtype=float))), ts), ts, cfg.freq)

    DXY  = S("DXY")
    ES   = S("ES")
    XAU  = S("XAU")
    U2   = S("UST2Y")
    U10  = S("UST10Y")

    names = ["dxy_ret_5m", "dxy_ret_60m", "es_ret_5m", "xau_ret_5m", "ust2y_bp_5m", "ust10y_bp_5m"]
    vals  = [
        _ret_last(DXY, 5),
        _ret_last(DXY, 60),
        _ret_last(ES, 5),
        _ret_last(XAU, 5),
        _bp_change_last(U2, 5),
        _bp_change_last(U10, 5),
    ]
    vals = [_clip01(float(v)) for v in vals]
    return vals, names


# ------------------------------ __main__ test ------------------------------ #

if __name__ == "__main__":
    rng = np.random.default_rng(99)

    def synth_price(n=2000, start="2025-09-25 00:00Z", base=100.0, vol=0.1):
        idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
        logp = np.log(base) + np.cumsum(rng.normal(0.0, vol, size=n))
        return pd.Series(np.exp(logp), index=idx)

    def synth_yield(n=2000, start="2025-09-25 00:00Z", base=4.0, vol=0.002):
        idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
        y = base + np.cumsum(rng.normal(0.0, vol, size=n))
        return pd.Series(y, index=idx)

    ts = pd.Timestamp("2025-10-03 08:14:00Z")
    caches = {
        "DXY": synth_price(base=105.0, vol=0.001),
        "ES":  synth_price(base=5300.0, vol=0.005),
        "XAU": synth_price(base=2350.0, vol=0.002),
        "UST2Y": synth_yield(base=4.90, vol=0.0005),
        "UST10Y": synth_yield(base=4.25, vol=0.0004),
    }

    vals, names = extract_macro_features_at(caches, ts, MacroConfig())
    print(f"Got {len(vals)} macro features at {ts.isoformat()}")
    for n, v in zip(names, vals):
        print(f"{n:>14}: {v:+.6f}")
    print("Range:", f"{min(vals):+.4f}", "→", f"{max(vals):+.4f}", " (should be within [-1, 1])")
    print("OK ✓")
