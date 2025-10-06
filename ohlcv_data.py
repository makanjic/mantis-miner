# ohlcv_data.py
# Fetch 1-minute OHLCV for crypto via CCXT and for FX/metals via yfinance.
# Returns UTC-indexed DataFrames with columns: open, high, low, close, volume
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

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


# -------------------- helpers -------------------- #

def _utc_ts(x) -> pd.Timestamp:
    """Parse ISO/unix/ts → tz-aware UTC minute timestamp (floored to minute)."""
    ts = pd.to_datetime(x, utc=True)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.floor("T")

def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize("UTC").floor("T")

def _clip_lookback_for_yf(lookback_minutes: int) -> int:
    # yfinance 1m offers ~7 days
    max_min = 7 * 24 * 60
    return min(lookback_minutes, max_min)


# -------------------- Crypto via CCXT (Binance by default) -------------------- #

def fetch_crypto_1m_ccxt_window(
    symbol: str,
    end_ts,
    lookback_minutes: int,
    exchange: str = "binance",
    max_chunk: int = 1000,
) -> pd.DataFrame:
    """
    Fetch 1m OHLCV for a crypto symbol via CCXT in a window [end_ts - lookback, end_ts].
    symbol: "BTC/USDT", "ETH/USDT", ...
    Returns UTC-indexed DataFrame with columns open,high,low,close,volume
    """
    if not _HAS_CCXT:
        raise RuntimeError("ccxt not installed. pip install ccxt")

    end_ts = _utc_ts(end_ts)
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    timeframe = "1m"
    since_ms = int(start_ts.timestamp() * 1000)

    all_rows = []
    fetch_since = since_ms

    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=max_chunk)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        # advance by one minute to avoid duplicates
        fetch_since = last_ts + 60_000
        # stop if we've reached or passed the desired end time or "now" (in ms)
        if last_ts >= int(end_ts.timestamp() * 1000):
            break
        # rate limit friend
        time.sleep(ex.rateLimit / 1000.0 if hasattr(ex, "rateLimit") else 0.25)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).set_index("timestamp").sort_index()

    # Filter exact window and coerce numerics
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()[["open", "high", "low", "close", "volume"]]


def fetch_crypto_1m_ccxt(
    symbol: str = "BTC/USDT",
    exchange: str = "binance",
    lookback_days: int = 5,
    max_chunk: int = 1000,
) -> pd.DataFrame:
    """
    Back-compat convenience: fetch last N days ending 'now'.
    """
    end_ts = _now_utc()
    lookback_minutes = lookback_days * 24 * 60
    return fetch_crypto_1m_ccxt_window(symbol, end_ts, lookback_minutes, exchange=exchange, max_chunk=max_chunk)


# -------------------- FX / Metals via yfinance (7 days limit) -------------------- #

_YF_TICKER_MAP: Dict[str, str] = {
    # FX
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "CADUSD": "CADUSD=X",
    "NZDUSD": "NZDUSD=X",
    "CHFUSD": "CHFUSD=X",
    # Metals
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
}

def fetch_yfinance_1m_window(
    yf_ticker: str,
    end_ts,
    lookback_minutes: int,
) -> pd.DataFrame:
    """
    Fetch 1m OHLCV via yfinance for a window [end_ts - lookback, end_ts].
    yfinance only provides ~7d of 1m bars; lookback is clipped accordingly.
    """
    if not _HAS_YF:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    end_ts = _utc_ts(end_ts)
    lookback_minutes = _clip_lookback_for_yf(lookback_minutes)
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)

    # yfinance wants naive timestamps (no tz) for start/end
    start_naive = start_ts.tz_convert(None)
    end_naive = (end_ts + pd.Timedelta(minutes=1)).tz_convert(None)  # include end minute

    df = yf.download(
        tickers=yf_ticker,
        start=start_naive,
        end=end_naive,
        interval="1m",
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Normalize possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    colmap = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        f"Open_{yf_ticker}": "open", f"High_{yf_ticker}": "high", f"Low_{yf_ticker}": "low",
        f"Close_{yf_ticker}": "close", f"Adj Close_{yf_ticker}": "adj_close", f"Volume_{yf_ticker}": "volume",
    }
    df = df.rename(columns=colmap)

    # Ensure UTC index
    idx = df.index
    if idx.tz is None:
        df.index = idx.tz_localize("UTC")
    else:
        df.index = idx.tz_convert("UTC")

    # Coerce numerics and fill volume if missing
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    else:
        df["volume"] = 0.0

    out = df[["open", "high", "low", "close", "volume"]].dropna()
    # Exact window trim
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    return out


def fetch_yfinance_1m(
    yf_ticker: str,
    lookback_days: int = 5,
) -> pd.DataFrame:
    """
    Back-compat convenience: fetch last N days ending 'now'.
    """
    end_ts = _now_utc()
    lookback_minutes = lookback_days * 24 * 60
    return fetch_yfinance_1m_window(yf_ticker, end_ts, lookback_minutes)


# -------------------- Unified helpers -------------------- #

def get_ohlcv_window_for_symbol(
    symbol_key: str,
    end_ts,
    lookback_minutes: int,
    exchange: str = "binance",
) -> pd.DataFrame:
    """
    symbol_key:
      - "BTC" -> CCXT "BTC/USDT"
      - "ETH" -> CCXT "ETH/USDT"
      - "EURUSD","GBPUSD","CADUSD","NZDUSD","CHFUSD","XAUUSD","XAGUSD" -> yfinance tickers
    Returns a UTC-indexed DataFrame with columns open,high,low,close,volume (1-minute).
    """
    sym = symbol_key.upper()
    end_ts = _utc_ts(end_ts)

    if sym in ("BTC", "ETH"):
        pair = "BTC/USDT" if sym == "BTC" else "ETH/USDT"
        return fetch_crypto_1m_ccxt_window(pair, end_ts, lookback_minutes, exchange=exchange)

    if sym in _YF_TICKER_MAP:
        return fetch_yfinance_1m_window(_YF_TICKER_MAP[sym], end_ts, lookback_minutes)

    raise ValueError(f"Unknown symbol_key={symbol_key}. Expected BTC, ETH, or one of {list(_YF_TICKER_MAP.keys())}.")


def get_ohlcv_for_symbol(
    symbol_key: str,
    lookback_days: int = 5,
    exchange: str = "binance",
) -> pd.DataFrame:
    """
    Back-compat convenience: fetch last N days ending 'now' (same behavior as before).
    """
    end_ts = _now_utc()
    return get_ohlcv_window_for_symbol(symbol_key, end_ts, lookback_days * 24 * 60, exchange=exchange)


# ------------------------------ __main__ demo ------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch 1m OHLCV for crypto/FX/metals")
    parser.add_argument("--symbol", action="append",
                        default=["BTC","ETH","EURUSD","GBPUSD","CADUSD","NZDUSD","CHFUSD","XAUUSD","XAGUSD"],
                        help="Symbol keys to fetch (repeatable). Default: all.")
    # Two ways to specify window: --days (like before) OR --end-ts + --lookback-min
    parser.add_argument("--days", type=int, default=None, help="Lookback days (<=7 for yfinance).")
    parser.add_argument("--end-ts", default=None, help="ISO timestamp for window end (e.g., 2025-10-03T08:14:00Z)")
    parser.add_argument("--lookback-min", type=int, default=None, help="Lookback minutes for precise window.")
    parser.add_argument("--exchange", type=str, default="binance", help="CCXT exchange for crypto (e.g., binance, okx, coinbase).")
    parser.add_argument("--save-csv", action="store_true", help="Save each symbol to CSV in cwd.")
    args = parser.parse_args()

    # Resolve window
    use_precise = args.end_ts is not None or args.lookback_min is not None
    if use_precise:
        if args.end_ts is None or args.lookback_min is None:
            raise SystemExit("When using --end-ts/--lookback-min, provide BOTH.")
        end_ts = _utc_ts(args.end_ts)
        lookback_minutes = int(args.lookback_min)
    else:
        # Fallback to days mode (back-compat)
        days = args.days if args.days is not None else 5
        end_ts = _now_utc()
        lookback_minutes = days * 24 * 60

    for sym in args.symbol:
        try:
            df = get_ohlcv_window_for_symbol(sym, end_ts, lookback_minutes, exchange=args.exchange)
            if df.empty:
                print(f"{sym}: no data returned for window { (end_ts - pd.Timedelta(minutes=lookback_minutes)).isoformat() } → { end_ts.isoformat() }")
                continue
            print(f"{sym}: {len(df):,} rows from {df.index[0].isoformat()} → {df.index[-1].isoformat()}")
            print(df.tail(3))
            if args.save_csv:
                outname = f"{sym}_1m.csv"
                df.to_csv(outname, index_label="timestamp")
                print(f"Saved {outname}")
        except Exception as e:
            print(f"{sym}: ERROR -> {e}")
