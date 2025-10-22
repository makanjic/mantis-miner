# ohlcv_data.py
# Fetch 1-minute OHLCV for crypto via CCXT and for FX/metals via yfinance.
# Returns UTC-indexed DataFrames with columns: open, high, low, close, volume

from __future__ import annotations
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

# ---- Optional deps (install: pip install ccxt yfinance) ----
try:
    import ccxt
    _HAS_CCXT = True
except Exception:
    _HAS_CCXT = False

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False


# -------------------- Crypto via CCXT (Binance by default) -------------------- #

def fetch_crypto_1m_ccxt(
    symbol: str = "BTC/USDT",
    exchange: str = "binance",
    lookback_days: float = 5.0,
    max_chunk: int = 1000,
) -> pd.DataFrame:
    """
    Fetch 1m OHLCV for a crypto symbol via CCXT from (now - lookback_days) to now.
    Returns UTC-indexed DataFrame with columns open,high,low,close,volume
    """
    if not _HAS_CCXT:
        raise RuntimeError("ccxt not installed. pip install ccxt")

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    timeframe = "1m"
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=float(lookback_days))).timestamp() * 1000)

    all_rows = []
    fetch_since = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=max_chunk)
        if not batch:
            break
        all_rows.extend(batch)
        # advance since by last timestamp + 60_000ms (next minute)
        last_ts = batch[-1][0]
        fetch_since = last_ts + 60_000
        # stop if we've crossed "now"
        if fetch_since >= int(time.time() * 1000):
            break
        # be nice to the API
        time.sleep(ex.rateLimit / 1000.0 if hasattr(ex, "rateLimit") else 0.3)

    if not all_rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).set_index("timestamp").sort_index()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df[["open","high","low","close","volume"]]


# -------------------- FX / Metals via yfinance (1m ≤ 7d limit) -------------------- #
# Metals updated: GC=F (gold), SI=F (silver)
_YF_TICKER_MAP: Dict[str, str] = {
    # FX
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "CADUSD": "CADUSD=X",
    "NZDUSD": "NZDUSD=X",
    "CHFUSD": "CHFUSD=X",
    # Metals (futures)
    "XAUUSD": "GC=F",  # gold futures
    "XAGUSD": "SI=F",  # silver futures
}

def _yf_download_1m(
    yf_ticker: str,
    *,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    period_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Best-effort 1m download for intraday via yfinance.
    - If start/end are provided, try a bounded fetch (Yahoo may still return up to 7d).
    - Else use period_days (must be <= 7).
    Always returns UTC-indexed DataFrame with columns open,high,low,close,volume (volume may be NaN; we fill 0).
    """
    if not _HAS_YF:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    kwargs = dict(
        tickers=yf_ticker,
        interval="1m",
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=True,
    )

    df = None
    if start_ts is not None and end_ts is not None:
        # yfinance expects naive timestamps; supply UTC-naive
        kwargs["start"] = pd.Timestamp(start_ts).tz_convert("UTC").tz_localize(None)
        kwargs["end"]   = pd.Timestamp(end_ts).tz_convert("UTC").tz_localize(None)
        df = yf.download(**kwargs)

    if df is None or df.empty:
        # Fallback to period (cap to 7d)
        days = 7 if (period_days is None) else max(1, min(7, int(period_days)))
        kwargs.pop("start", None); kwargs.pop("end", None)
        kwargs["period"] = f"{days}d"
        df = yf.download(**kwargs)

    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    colmap = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        f"Open_{yf_ticker}": "open", f"High_{yf_ticker}": "high", f"Low_{yf_ticker}": "low",
        f"Close_{yf_ticker}": "close", f"Adj Close_{yf_ticker}": "adj_close", f"Volume_{yf_ticker}": "volume",
    }
    df = df.rename(columns=colmap)
    # ensure UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    else:
        df["volume"] = 0.0

    out = df[["open","high","low","close","volume"]].dropna()
    return out


def fetch_yfinance_1m(
    yf_ticker: str,
    *,
    lookback_days: Optional[float] = None,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Wrapper that chooses the best yfinance call for 1m intraday:
    - If start/end provided → try bounded download; trim to [start,end].
    - Else if lookback_days provided → period=ceil(lookback_days) days (≤ 7); trim to last 'lookback_days'.
    """
    end_ts = pd.Timestamp.now(tz=timezone.utc) if end_ts is None else pd.to_datetime(end_ts, utc=True)
    if start_ts is not None:
        start_ts = pd.to_datetime(start_ts, utc=True)

    if start_ts is not None:
        df = _yf_download_1m(yf_ticker, start_ts=start_ts, end_ts=end_ts)
        # trim hard to the requested window
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    # fallback to period-days mode
    days = 7 if lookback_days is None else int(np.ceil(lookback_days))
    days = max(1, min(7, days))
    df = _yf_download_1m(yf_ticker, period_days=days)
    if lookback_days is None:
        return df
    else:
        cutoff = end_ts - timedelta(days=float(lookback_days))
        return df.loc[df.index >= cutoff]


# -------------------- Unified helper for your 9 symbols -------------------- #

def get_ohlcv_for_symbol(
    symbol_key: str,
    *,
    lookback_days: Optional[float] = None,
    exchange: str = "binance",
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    ccxt_max_days: float = 30.0,
) -> pd.DataFrame:
    """
    Dispatch per symbol:
      - BTC/ETH via CCXT (precise since in minutes)
      - FX/metals via yfinance (1m ≤ 7d)
    Supports either 'lookback_days' or (start_ts,end_ts).
    """
    sym = symbol_key.upper()

    if sym in ("BTC", "ETH"):
        pair = "BTC/USDT" if sym == "BTC" else "ETH/USDT"
        if start_ts is not None and end_ts is not None:
            # compute float days precisely for CCXT
            days = max(0.01, (pd.to_datetime(end_ts, utc=True) - pd.to_datetime(start_ts, utc=True)).total_seconds() / 86400.0)
            return fetch_crypto_1m_ccxt(pair, exchange=exchange, lookback_days=days)
        # default to lookback_days or cap by ccxt_max_days if None
        days = ccxt_max_days if lookback_days is None else float(lookback_days)
        return fetch_crypto_1m_ccxt(pair, exchange=exchange, lookback_days=days)

    if sym in _YF_TICKER_MAP:
        yf_ticker = _YF_TICKER_MAP[sym]
        return fetch_yfinance_1m(yf_ticker, lookback_days=lookback_days, start_ts=start_ts, end_ts=end_ts)

    raise ValueError(f"Unknown symbol_key={symbol_key}. Expected BTC, ETH, or one of {list(_YF_TICKER_MAP.keys())}.")


# -------------------- CSV save/append utilities -------------------- #

def _read_existing_csv(path: str) -> pd.DataFrame:
    """Robustly read an existing OHLCV CSV with a timestamp column."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # find timestamp column
    ts_col = None
    for c in df.columns:
        if str(c).lower() == "timestamp":
            ts_col = c
            break
    if ts_col is None:
        # maybe index saved as first col
        ts_col = df.columns[0]

    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(__ts=idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()

    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    need = ["open","high","low","close","volume"]
    out = pd.DataFrame(index=df.index)
    for k in need:
        if k in cols:
            out[k] = pd.to_numeric(df[cols[k]], errors="coerce")
        else:
            out[k] = np.nan
    return out.dropna(how="all")

def _last_timestamp_in_csv(path: str) -> Optional[pd.Timestamp]:
    """Read only the last timestamp (UTC) from an existing CSV."""
    if not path or not os.path.exists(path):
        return None
    try:
        df = _read_existing_csv(path)
        if df.empty:
            return None
        return pd.to_datetime(df.index.max()).tz_convert("UTC")
    except Exception:
        return None

def _merge_and_save_csv(new_df: pd.DataFrame, path: str, append: bool) -> None:
    """
    If append=True and file exists: merge existing + new on index (timestamp), drop dups (keep last), sort, overwrite.
    Else: write new_df fresh.
    """
    if not append or not os.path.exists(path):
        new_df.to_csv(path, index_label="timestamp")
        print(f"Saved {path} ({len(new_df)} rows)")
        return

    old = _read_existing_csv(path)
    merged = pd.concat([old, new_df], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(path, index_label="timestamp")
    added = len(merged) - len(old)
    print(f"Appended {added} new rows → {path} (total {len(merged)})")


# ------------------------------ __main__ ------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch 1m OHLCV for crypto/FX/metals (append-smart).")
    parser.add_argument("--symbol", action="append",
                        default=["BTC","ETH","EURUSD","GBPUSD","CADUSD","NZDUSD","CHFUSD","XAUUSD","XAGUSD"],
                        help="Symbol keys to fetch (repeatable). Default: all.")
    parser.add_argument("--days", type=float, default=None,
                        help="Baseline lookback days. If None: auto (see --append rules).")
    parser.add_argument("--exchange", type=str, default="binance", help="CCXT exchange for crypto.")
    parser.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                        help="Directory to save CSVs. If provided with no path, uses current directory.")
    parser.add_argument("--append", action="store_true",
                        help="When saving, merge into existing CSV and de-duplicate by timestamp. Also auto-shorten lookback based on existing file when --days is None.")
    parser.add_argument("--overlap-minutes", type=int, default=120,
                        help="Safety overlap to refetch behind last saved minute when appending (default: 120).")
    parser.add_argument("--max-ccxt-days", type=float, default=30.0,
                        help="When --days is None and not appending, max days for CCXT crypto fetch.")
    args = parser.parse_args()

    # Prepare save dir only if requested
    save_dir = args.save_to_csv
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    now_utc = datetime.now(timezone.utc)

    for sym in args.symbol:
        try:
            csv_path = os.path.join(save_dir, f"{sym}_1m.csv") if save_dir else None
            last_ts = _last_timestamp_in_csv(csv_path) if (save_dir and args.append) else None

            # Decide fetching window for this symbol
            start_ts: Optional[pd.Timestamp] = None
            end_ts: Optional[pd.Timestamp] = None
            lookback_days: Optional[float] = None

            if args.days is not None:
                # explicit baseline lookback takes precedence
                lookback_days = float(args.days)
                print(f"{sym}: using explicit --days={lookback_days:.2f}")
            else:
                if args.append and last_ts is not None:
                    # minute-precise incremental window with overlap
                    start_ts = last_ts - timedelta(minutes=max(0, args.overlap_minutes))
                    end_ts = now_utc
                    print(f"{sym}: incremental from {start_ts.isoformat()} to {end_ts.isoformat()} (overlap {args.overlap_minutes}m)")
                else:
                    # largest safe window
                    if sym in ("BTC", "ETH"):
                        lookback_days = float(args.max_ccxt_days)
                        print(f"{sym}: auto lookback (CCXT) {lookback_days:.2f}d (max-ccxt-days)")
                    else:
                        lookback_days = 7.0   # yfinance intraday cap
                        print(f"{sym}: auto lookback (yfinance) {lookback_days:.0f}d cap")

            # Fetch
            df = get_ohlcv_for_symbol(
                sym,
                lookback_days=lookback_days,
                exchange=args.exchange,
                start_ts=start_ts,
                end_ts=end_ts,
                ccxt_max_days=args.max_ccxt_days,
            )

            if df.empty:
                print(f"{sym}: 0 rows (no data)")
                continue

            # If we used a start_ts/end_ts window, hard-trim (yfinance may return extras)
            if start_ts is not None and end_ts is not None:
                df = df.loc[(df.index >= pd.to_datetime(start_ts, utc=True)) & (df.index <= pd.to_datetime(end_ts, utc=True))]

            print(f"{sym}: {len(df):,} fetched rows {df.index[0].isoformat()} → {df.index[-1].isoformat()}")

            # Save/append if requested
            if save_dir:
                fname = os.path.join(save_dir, f"{sym}_1m.csv")
                _merge_and_save_csv(df, fname, append=args.append)

        except Exception as e:
            print(f"{sym}: ERROR -> {e}")
