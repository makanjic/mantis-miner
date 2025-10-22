# derivs_features.py
# Derivatives (perp/spot basis, funding, OI, long/short) features at a specified timestamp.
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

from features_helpers import (
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _delta_last, _basis_change,
    _sum_window, _clip01,
)

# --------------------------- config --------------------------- #

@dataclass(frozen=True)
class DerivsConfig:
    z_long: int = 7 * 24 * 60     # ~7d window for slow regime (funding / OI)
    z_med:  int = 24 * 60         # ~1d window for long/short ratio
    d15:    int = 15              # 15-minute delta/sum (kept for liqs if you re-enable)
    d60:    int = 60              # 60-minute delta/sum
    freq:   str = "1min"          # resample grid

# --------------------------- core extractor --------------------------- #

def extract_derivs_features_at(
    caches: Mapping[str, pd.Series | pd.DataFrame],
    ts,
    cfg: DerivsConfig = DerivsConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Compute derivatives features at timestamp `ts`, using only data <= ts.
    Expected keys in `caches` (all optional):
      funding, perp_close, spot_close, open_interest, long_short
      (liq_buy, liq_sell supported in helpers but disabled here)
    Returns (values, names). Values are bounded to [-1, 1].
    """
    ts = pd.to_datetime(ts, utc=True)

    def S(key: str) -> pd.Series:
        return _resample_1m_ffill(
            _leq(_prep_series(caches.get(key, pd.Series(dtype=float))), ts),
            ts,
            cfg.freq,
        )

    funding    = S("funding")
    perp_close = S("perp_close")
    spot_close = S("spot_close")
    oi         = S("open_interest")
    lsr        = S("long_short")
    # liq_b      = S("liq_buy")
    # liq_s      = S("liq_sell")

    names = [   
        "funding_z_long", "funding_d60",
        "basis_5m", "basis_60m",
        "oi_z_long", "oi_d60",
        "long_short_z_med",
        # "liq_net_15m", "liq_net_60m",
    ]
    vals = [
        _z_last(funding, cfg.z_long),
        _delta_last(funding, cfg.d60),
        _basis_change(perp_close, spot_close, 5),
        _basis_change(perp_close, spot_close, 60),
        _z_last(oi, cfg.z_long),
        _delta_last(oi, cfg.d60),
        _z_last(lsr, cfg.z_med),
        # _clip01(np.tanh(_sum_window(liq_b, cfg.d15) - _sum_window(liq_s, cfg.d15))),
        # _clip01(np.tanh(_sum_window(liq_b, cfg.d60) - _sum_window(liq_s, cfg.d60))),
    ]
    vals = [_clip01(float(v)) for v in vals]
    return vals, names

# --------------------------- CSV → caches helpers --------------------------- #

def _read_csv_series(path: Optional[str], value_cols: List[str]) -> pd.Series:
    """
    Load a CSV with a 'timestamp' column (UTC or naive), return the FIRST existing column
    in value_cols as Series. If the file/column is missing or empty, returns empty Series.
    """
    if not path or not os.path.exists(path):
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.Series(dtype=float)

    # robust timestamp handling
    ts_col = None
    for c in df.columns:
        if c.lower() == "timestamp":
            ts_col = c
            break
    if ts_col is None:
        # maybe the index was saved as unnamed column
        if df.columns and str(df.columns[0]).lower() in ("", "unnamed: 0", "index"):
            ts_col = df.columns[0]
        else:
            return pd.Series(dtype=float)

    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(__ts=idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()

    # pick the first available value column
    for vc in value_cols:
        if vc in df.columns:
            s = pd.to_numeric(df[vc], errors="coerce").dropna()
            s.index = pd.to_datetime(s.index, utc=True)
            s = s[~s.index.duplicated(keep="last")]
            return s.astype(float)

    return pd.Series(dtype=float)

def build_derivs_caches_from_csv(
    *,
    perp_csv: Optional[str] = None,
    spot_csv: Optional[str] = None,
    funding_csv: Optional[str] = None,
    lsratio_csv: Optional[str] = None,
    oi_csv: Optional[str] = None,
) -> Dict[str, pd.Series]:
    """
    Map CSV files (from derivs_data.py --save-to-csv) into the caches expected by extract_derivs_features_at().
    """
    caches: Dict[str, pd.Series] = {}

    # perp/spot close
    perp_close = _read_csv_series(perp_csv, ["close"])
    if not perp_close.empty:
        caches["perp_close"] = perp_close
    spot_close = _read_csv_series(spot_csv, ["close"])
    if not spot_close.empty:
        caches["spot_close"] = spot_close

    # funding
    funding = _read_csv_series(funding_csv, ["fundingRate", "funding_rate"])
    if not funding.empty:
        caches["funding"] = funding

    # long/short ratio (prefer provided ratio; else compute from accounts if present)
    if lsratio_csv and os.path.exists(lsratio_csv):
        df = pd.read_csv(lsratio_csv)
        tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
        idx = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.assign(__ts=idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()

        series = None
        if "longShortRatio" in df.columns:
            series = pd.to_numeric(df["longShortRatio"], errors="coerce")
        elif {"longAccount", "shortAccount"}.issubset(df.columns):
            la = pd.to_numeric(df["longAccount"], errors="coerce")
            sa = pd.to_numeric(df["shortAccount"], errors="coerce")
            ratio = la / sa.replace(0.0, np.nan)
            series = ratio
        if series is not None:
            s = series.dropna()
            s.index = pd.to_datetime(s.index, utc=True)
            s = s[~s.index.duplicated(keep="last")].astype(float)
            if not s.empty:
                caches["long_short"] = s

    # open interest
    oi = _read_csv_series(oi_csv, ["sumOpenInterest", "openInterest", "open_interest"])
    if not oi.empty:
        caches["open_interest"] = oi

    return caches

# --------------------------- dir-based loader --------------------------- #

def _paths_from_dir(load_dir: str, symbol: str, perp_interval: str, spot_interval: str,
                    ls_period: str, oi_period: str) -> Dict[str, Optional[str]]:
    """
    Build expected file paths for a symbol under load_dir.
    """
    p = lambda *parts: os.path.join(load_dir, *parts)
    paths = {
        "perp_csv":   p(f"{symbol}_perp_{perp_interval}.csv"),
        "spot_csv":   p(f"{symbol}_spot_{spot_interval}.csv"),
        "funding_csv":p(f"{symbol}_funding.csv"),
        "lsratio_csv":p(f"{symbol}_lsratio_{ls_period}.csv"),
        "oi_csv":     p(f"{symbol}_oi_{oi_period}.csv"),
    }
    # Normalize missing files to None
    for k, v in list(paths.items()):
        if not os.path.exists(v):
            paths[k] = None
    return paths

def build_caches_from_dir(load_dir: str, symbol: str,
                          perp_interval: str = "1m", spot_interval: str = "1m",
                          ls_period: str = "5m", oi_period: str = "5m") -> Dict[str, pd.Series]:
    paths = _paths_from_dir(load_dir, symbol, perp_interval, spot_interval, ls_period, oi_period)
    return build_derivs_caches_from_csv(**paths)

# --------------------------- feature file I/O --------------------------- #

def _find_feature_path(save_dir: str, symbol: str) -> str:
    """
    Default feature file: <SYMBOL>_features.csv.
    If <SYMBOL>.csv already exists, use it for continuity (same behavior as ohlcv_features.py).
    """
    cand_existing_plain = os.path.join(save_dir, f"{symbol}.csv")
    cand_default = os.path.join(save_dir, f"{symbol}_features.csv")
    return cand_existing_plain if os.path.exists(cand_existing_plain) else cand_default

def _append_or_overwrite_feature_file(path: str, new_rows: pd.DataFrame, append: bool) -> None:
    """
    Append with dedup (by 'timestamp') if append=True; otherwise overwrite.
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

# --------------------------- __main__ CLI --------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Derivatives features from CSVs (dir-based like ohlcv_features.py).")
    ap.add_argument("--symbol", action="append", default=["BTCUSDT"],
                    help="Binance symbol(s) to process (repeatable). Default: BTCUSDT.")
    ap.add_argument("--ts", default="last", help="ISO8601, 'now' (UTC), or 'last' (default).")
    ap.add_argument("--load-from-csv", default=None,
                    help="Directory containing files saved by derivs_data.py "
                         "(<SYMBOL>_perp_<interval>.csv, <SYMBOL>_spot_<interval>.csv, etc.).")
    ap.add_argument("--perp-interval", default="1m")
    ap.add_argument("--spot-interval", default="1m")
    ap.add_argument("--ls-period", default="5m")
    ap.add_argument("--oi-period", default="5m")

    # Optional explicit file overrides (single-symbol use/debug)
    ap.add_argument("--perp-csv", help="Path to <SYMBOL>_perp_<interval>.csv")
    ap.add_argument("--spot-csv", help="Path to <SYMBOL>_spot_<interval>.csv")
    ap.add_argument("--funding-csv", help="Path to <SYMBOL>_funding.csv")
    ap.add_argument("--lsratio-csv", help="Path to <SYMBOL>_lsratio_<period>.csv")
    ap.add_argument("--oi-csv", help="Path to <SYMBOL>_oi_<period>.csv")

    # Save like ohlcv_features.py
    ap.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                    help="Directory to write per-symbol feature CSVs (<SYMBOL>_features.csv "
                         "or <SYMBOL>.csv if present). If omitted, nothing is saved.")
    ap.add_argument("--append", action="store_true",
                    help="When saving, append and de-duplicate by timestamp. Otherwise overwrite.")
    ap.add_argument("--print-batch", action="store_true", help="Print a short preview for each symbol.")

    args = ap.parse_args()

    # Resolve target timestamp
    if args.ts == "last":
        ts_opt: Optional[pd.Timestamp] = None
    elif args.ts == "now":
        ts_opt = pd.Timestamp.now(tz="UTC").floor("min")
    else:
        ts_opt = pd.to_datetime(args.ts, utc=True).floor("min")

    # Process each symbol
    for sym in args.symbol:
        # 1) Load caches
        if args.load_from_csv:
            caches = build_caches_from_dir(
                args.load_from_csv, sym,
                perp_interval=args.perp_interval,
                spot_interval=args.spot_interval,
                ls_period=args.ls_period,
                oi_period=args.oi_period,
            )
        elif any([args.perp_csv, args.spot_csv, args.funding_csv, args.lsratio_csv, args.oi_csv]):
            caches = build_derivs_caches_from_csv(
                perp_csv=args.perp_csv,
                spot_csv=args.spot_csv,
                funding_csv=args.funding_csv,
                lsratio_csv=args.lsratio_csv,
                oi_csv=args.oi_csv,
            )
        else:
            # Synthetic fallback for quick testing (no files)
            print(f"[warn] No CSVs provided for {sym}; generating synthetic caches for smoke test.")
            rng = np.random.default_rng(123)
            idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=5*24*60, freq="min")
            price = 50000 + np.cumsum(rng.normal(0, 20, size=len(idx)))
            perp_close = pd.Series(price, index=idx)
            spot_close = perp_close * (1 - 0.0002)  # tiny basis
            funding    = pd.Series(rng.normal(0, 0.00002, size=len(idx))).cumsum() / 100.0
            oi         = pd.Series(1e9 + np.cumsum(rng.normal(0, 1e6, size=len(idx))), index=idx)
            lsr        = pd.Series(1.0 + rng.normal(0, 0.03, size=len(idx)), index=idx)
            caches = {
                "perp_close": perp_close,
                "spot_close": spot_close,
                "funding": funding,
                "open_interest": oi,
                "long_short": lsr,
            }

        # 2) Choose timestamp when args.ts == 'last'
        if ts_opt is None:
            # Prefer the latest minute from perp/spot if available; else the max over all series
            last_candidates: List[pd.Timestamp] = []
            for k in ("perp_close", "spot_close"):
                s = caches.get(k)
                if isinstance(s, pd.Series) and not s.empty:
                    last_candidates.append(pd.Timestamp(s.index[-1]).tz_convert("UTC"))
            if not last_candidates:
                for s in caches.values():
                    if isinstance(s, pd.Series) and not s.empty:
                        last_candidates.append(pd.Timestamp(s.index[-1]).tz_convert("UTC"))
            if not last_candidates:
                print(f"[{sym}] no data available to compute features; skipping.")
                continue
            ts_use = max(last_candidates).floor("min")
        else:
            ts_use = ts_opt

        # 3) Compute features
        vals, names = extract_derivs_features_at(caches, ts_use, DerivsConfig())
        if args.print_batch or len(args.symbol) == 1:
            print(f"\n[derivs:{sym}] {len(vals)} features @ {ts_use.isoformat()}")
            for n, v in zip(names, vals):
                print(f"  {n:18s}: {v:+.6f}")

        # 4) Save per-symbol file if requested
        if args.save_to_csv:
            save_dir = args.save_to_csv
            os.makedirs(save_dir, exist_ok=True)
            out_path = _find_feature_path(save_dir, sym)
            row = {"timestamp": ts_use, "symbol": sym}
            row.update({n: float(v) for n, v in zip(names, vals)})
            out_df = pd.DataFrame([row])
            _append_or_overwrite_feature_file(out_path, out_df, append=args.append)

    print("\nOK ✓ (derivs feature extraction completed)")
