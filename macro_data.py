# macro_data.py
# Macro proxies with per-ticker intervals (yfinance), incremental append + overlap,
# and column-safe merging (preserves <KEY>_close on append).

from __future__ import annotations

import os
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

# Optional dependency (install: pip install yfinance)
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

# ---------------- Config: map logical keys -> yfinance tickers & default intervals ---------------

TICKERS: Dict[str, str] = {
    "DXY": "DX=F",   # US Dollar Index futures
    "ES":  "ES=F",   # S&P 500 E-mini futures
    "XAU": "GC=F",   # Gold futures
    "XAG": "SI=F",   # Silver futures
    # If you later want 10Y yield via Yahoo index (note: ^TNX ~ yield * 10):
    # "UST10Y": "^TNX",
}

# Sensible per-ticker default intervals (coarser expands history & reduces noise)
DEFAULT_INTERVALS: Dict[str, str] = {
    "DXY": "15m",
    "ES":  "60m",
    "XAU": "60m",
    "XAG": "60m",
    # "UST10Y": "60m",
}

# yfinance intraday history caps by interval (rough rules of thumb; used for period selection)
INTERVAL_CAP_DAYS: Dict[str, int] = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "90m": 730,
    "1h": 730, "1d": 10000
}

def _minutes_per_bar(interval: str) -> int:
    m = {
        "1m":1, "2m":2, "5m":5, "15m":15, "30m":30, "60m":60, "90m":90,
        "1h":60, "1d":1440
    }
    return m.get(interval, 60)

# ---------------- yfinance helpers ----------------

def _yf_download(
    yf_ticker: str,
    interval: str,
    *,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    period_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Best-effort bounded download for the given interval.
    If start/end provided: request that window (Yahoo may give more; we trim later).
    Else: use period_days (capped by INTERVAL_CAP_DAYS).
    Returns UTC-indexed df with 'close' (and 'volume' if available; else 0).
    """
    if not _HAS_YF:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    kwargs = dict(
        tickers=yf_ticker,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=True,
    )
    df = None

    if start_ts is not None and end_ts is not None:
        kwargs["start"] = pd.Timestamp(start_ts).tz_convert("UTC").tz_localize(None)
        kwargs["end"]   = pd.Timestamp(end_ts).tz_convert("UTC").tz_localize(None)
        df = yf.download(**kwargs)

    if df is None or df.empty:
        if period_days is None:
            period_days = INTERVAL_CAP_DAYS.get(interval, 60)
        period_days = max(1, min(period_days, INTERVAL_CAP_DAYS.get(interval, 60)))
        kwargs.pop("start", None); kwargs.pop("end", None)
        kwargs["period"] = f"{period_days}d"
        df = yf.download(**kwargs)

    if df is None or df.empty:
        return pd.DataFrame(columns=["close", "volume"])

    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]
    colmap = {
        "Close": "close", f"Close_{yf_ticker}": "close",
        "Volume": "volume", f"Volume_{yf_ticker}": "volume",
    }
    df = df.rename(columns=colmap)

    idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.set_index(idx)

    if "close" not in df.columns:
        # Try lowercase mapping
        lower = {c.lower(): c for c in df.columns}
        if "close" in lower:
            df = df.rename(columns={lower["close"]: "close"})
        else:
            return pd.DataFrame(columns=["close", "volume"])

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    out = df[["close", "volume"]].dropna(subset=["close"])
    return out

def _download_close_window(
    yf_ticker: str,
    interval: str,
    *,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    lookback_minutes: Optional[int],
) -> pd.DataFrame:
    """
    Choose bounded start/end if provided; else derive a reasonable period from lookback+interval.
    Always trims to [start_ts, end_ts] if both are provided; otherwise trims to last lookback_minutes.
    """
    if start_ts is not None and end_ts is not None:
        df = _yf_download(yf_ticker, interval, start_ts=start_ts, end_ts=end_ts)
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    # Period mode
    if lookback_minutes is None:
        period_days = INTERVAL_CAP_DAYS.get(interval, 60)
    else:
        bars = int(np.ceil(lookback_minutes / _minutes_per_bar(interval)))
        period_days = int(np.ceil(bars * _minutes_per_bar(interval) / (24 * 60)))
        period_days = max(1, min(period_days, INTERVAL_CAP_DAYS.get(interval, 60)))

    df = _yf_download(yf_ticker, interval, period_days=period_days)
    if lookback_minutes is None:
        return df
    cutoff = pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(minutes=lookback_minutes)
    return df.loc[df.index >= cutoff]

# ---------------- CSV helpers (append with dedup & column preservation) ----------------

def _read_existing_csv(path: str, expected_close_col: Optional[str] = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # timestamp column
    ts_col = None
    for c in df.columns:
        if str(c).lower() == "timestamp":
            ts_col = c; break
    if ts_col is None:
        ts_col = df.columns[0]

    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(__ts=idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()

    cols_lower = {c.lower(): c for c in df.columns}

    # Decide which close column to keep/normalize
    if expected_close_col and expected_close_col in df.columns:
        out = df[[expected_close_col]].copy()
    elif expected_close_col and "close" in cols_lower:
        out = df[[cols_lower["close"]]].copy().rename(columns={cols_lower["close"]: expected_close_col})
    else:
        close_like = [c for c in df.columns if c.lower().endswith("_close") or c.lower() == "close"]
        out = df[close_like].copy() if close_like else pd.DataFrame(index=df.index)

    # keep volume if present
    if "volume" in df.columns:
        out["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # enforce numeric on close-like columns
    for c in out.columns:
        if c.lower().endswith("close") or c.lower() == "close":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.dropna(how="all")

def _last_ts_in_csv(path: str) -> Optional[pd.Timestamp]:
    if not path or not os.path.exists(path):
        return None
    try:
        df = _read_existing_csv(path)  # close col name not needed for max ts
        if df.empty: return None
        return pd.to_datetime(df.index.max()).tz_convert("UTC")
    except Exception:
        return None

def _merge_and_save(new_df: pd.DataFrame, path: str, append: bool,
                    expected_close_col: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Ensure new_df uses expected <KEY>_close name if provided
    if expected_close_col and "close" in new_df.columns and expected_close_col not in new_df.columns:
        new_df = new_df.rename(columns={"close": expected_close_col})

    if append and os.path.exists(path):
        old = _read_existing_csv(path, expected_close_col=expected_close_col)

        # Align columns (union)
        all_cols = list(dict.fromkeys(list(old.columns) + list(new_df.columns)))  # preserve order
        old = old.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)

        merged = pd.concat([old, new_df], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(path, index_label="timestamp")
        print(f"Appended {len(new_df)} rows → {path} (total {len(merged)})")
    else:
        new_df.to_csv(path, index_label="timestamp")
        print(f"Wrote {path} ({len(new_df)})")

# ---------------- Public API ----------------

def load_macro_window(
    end_ts,
    lookback_minutes: int,
    *,
    intervals: Optional[Dict[str, str]] = None,
    keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Pull a multi-column DataFrame over [end - lookback, end] using per-ticker intervals.
    Columns named <KEY>_close for each selected key.
    """
    intervals = intervals or DEFAULT_INTERVALS
    if keys is None:
        keys = list(TICKERS.keys())

    end_ts = pd.to_datetime(end_ts, utc=True).floor("min")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)

    pieces = []
    for key in keys:
        if key not in TICKERS:
            continue
        yf_symbol = TICKERS[key]
        interval = intervals.get(key, "60m")
        s = _download_close_window(
            yf_symbol, interval,
            start_ts=None, end_ts=None,
            lookback_minutes=lookback_minutes
        )
        if not s.empty:
            s = s.rename(columns={"close": f"{key}_close"})
            pieces.append(s[[f"{key}_close"]])

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).sort_index()
    return out.loc[start_ts:end_ts]

# ------------------------------ __main__ ------------------------------ #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Macro proxies via yfinance with per-ticker intervals (append + overlap).")
    ap.add_argument("--key", action="append", default=["DXY","ES","XAU","XAG"],
                    help="Logical keys to fetch (default: DXY, ES, XAU, XAG). Repeatable.")
    ap.add_argument("--end-ts", default="now", help="ISO or 'now' (UTC).")
    ap.add_argument("--lookback-min", type=int, default=24*60, help="Baseline lookback in minutes.")
    ap.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                    help="Directory to save per-key CSVs (<KEY>_<INTERVAL>.csv). If omitted, no save.")
    ap.add_argument("--append", action="store_true",
                    help="Append with dedup by timestamp; auto-shortens fetch using last ts + overlap.")
    ap.add_argument("--overlap-minutes", type=int, default=180,
                    help="Refetch tail overlap when appending (default: 180).")
    # Optional overrides: e.g., --interval DXY=30m --interval ES=15m
    ap.add_argument("--interval", action="append", default=[],
                    help="Override per-key interval, e.g. DXY=30m (repeatable).")

    args = ap.parse_args()

    # Build intervals map
    intervals = dict(DEFAULT_INTERVALS)
    for ov in args.interval:
        if "=" in ov:
            k, v = ov.split("=", 1)
            intervals[k.strip()] = v.strip()

    # Resolve end time
    end_ts = (pd.Timestamp.now(tz="UTC").floor("min")
              if args.end_ts == "now" else pd.to_datetime(args.end_ts, utc=True).floor("min"))

    # Prepare save dir
    save_dir = args.save_to_csv
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Process each key independently (different intervals + append)
    for key in args.key:
        if key not in TICKERS:
            print(f"[warn] Unknown key {key}; skipping.")
            continue

        yf_symbol = TICKERS[key]
        interval = intervals.get(key, "60m")

        # Determine start_ts (append-aware)
        start_ts = end_ts - pd.Timedelta(minutes=args.lookback_min)
        csv_path = os.path.join(save_dir, f"{key}_{interval}.csv") if save_dir else None
        if args.append and csv_path and os.path.exists(csv_path):
            last_ts = _last_ts_in_csv(csv_path)
            if last_ts is not None:
                start_ts = (last_ts - pd.Timedelta(minutes=max(0, args.overlap_minutes))).floor("min")
                print(f"[{key}] append: last={last_ts.isoformat()}, overlap={args.overlap_minutes}m → "
                      f"start={start_ts.isoformat()} interval={interval}")

        # Download (bounded by lookback and interval’s cap)
        minutes = max(1, int((end_ts - start_ts).total_seconds() // 60))
        s = _download_close_window(
            yf_symbol, interval,
            start_ts=None, end_ts=None,
            lookback_minutes=minutes
        )
        if s.empty:
            print(f"[{key}] no data.")
            continue

        # Hard trim to [start,end], then rename close -> <KEY>_close
        s = s.loc[(s.index >= start_ts) & (s.index <= end_ts)].copy()
        if s.empty:
            print(f"[{key}] no rows within trimmed window.")
            continue

        out = s.rename(columns={"close": f"{key}_close"})
        print(f"[{key}] {len(out)} rows {out.index[0].isoformat()} → {out.index[-1].isoformat()} (interval {interval})")

        # Save if requested (preserving <KEY>_close on append)
        if save_dir:
            _merge_and_save(out, csv_path, append=args.append, expected_close_col=f"{key}_close")

    print("OK ✓")
