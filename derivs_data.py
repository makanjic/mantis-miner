# derivs_data.py
# Fetch derivatives data up to a SPECIFIED timestamp (UTC).
from __future__ import annotations

import requests
import pandas as pd
from typing import Optional, Dict

BINANCE_FAPI = "https://fapi.binance.com"  # USDⓈ-M Futures
BINANCE_SPOT = "https://api.binance.com"   # Spot

def _to_ms(t) -> int:
    if isinstance(t, pd.Timestamp):
        return int(t.tz_convert("UTC").timestamp() * 1000)
    if isinstance(t, (int, float)):
        return int(t if t > 10_000_000_000 else t * 1000)
    raise TypeError("Unsupported time type for _to_ms")

def _get(url: str, params: dict, timeout: int = 20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_perp_klines_window(symbol: str, end_ts, lookback_minutes: int, interval: str = "1m") -> pd.DataFrame:
    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "interval": interval, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts), "limit": 1500}
    js = _get(f"{BINANCE_FAPI}/fapi/v1/klines", params)
    if not js:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(js, columns=["t","o","h","l","c","v","ct","qv","n","taker_bv","taker_qv","i"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    out = df.set_index("timestamp")[["o","h","l","c","v"]].astype(float).rename(
        columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}
    )
    return out.sort_index().loc[start_ts:end_ts]

def fetch_spot_klines_window(symbol: str, end_ts, lookback_minutes: int, interval: str = "1m") -> pd.DataFrame:
    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "interval": interval, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts), "limit": 1500}
    js = _get(f"{BINANCE_SPOT}/api/v3/klines", params)
    if not js:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(js, columns=["t","o","h","l","c","v","ct","qv","n","taker_bv","taker_qv","i","b","a"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    out = df.set_index("timestamp")[["o","h","l","c","v"]].astype(float).rename(
        columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}
    )
    return out.sort_index().loc[start_ts:end_ts]

def fetch_funding_window(symbol: str, end_ts, lookback_minutes: int, limit: int = 1000) -> pd.DataFrame:
    """Funding prints (8h cadence). Has startTime/endTime params. Docs confirm. """
    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts), "limit": limit}
    js = _get(f"{BINANCE_FAPI}/fapi/v1/fundingRate", params)
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame(columns=["fundingRate"])
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df.set_index("timestamp")[["fundingRate"]].astype(float).sort_index().loc[start_ts:end_ts]

def fetch_long_short_ratio_window(symbol: str, end_ts, lookback_minutes: int, period: str = "5m", limit: int = 500) -> pd.DataFrame:
    """Global Long/Short Account Ratio; supports startTime/endTime (30d history limit)."""
    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "period": period, "limit": limit, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts)}
    js = _get(f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio", params)
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame(columns=["longShortRatio","longAccount","shortAccount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    cols = ["longShortRatio","longAccount","shortAccount"]
    df[cols] = df[cols].astype(float)
    return df.set_index("timestamp")[cols].sort_index().loc[start_ts:end_ts]

def fetch_open_interest_stats_window(symbol: str, end_ts, lookback_minutes: int, period: str = "5m", limit: int = 500) -> pd.DataFrame:
    """Historical OI stats (not just point-in-time OI). 30d availability. """
    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "period": period, "limit": limit, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts)}
    js = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist", params)
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame(columns=["sumOpenInterest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")[["sumOpenInterest"]].astype(float).sort_index().loc[start_ts:end_ts]

def fetch_liquidations_window(symbol: str, end_ts, lookback_minutes: int, limit: int = 1000) -> pd.DataFrame:
    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "limit": limit, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts)}
    js = _get(f"{BINANCE_FAPI}/fapi/v1/allForceOrders", params)
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame(columns=["side","price","executedQty","liq_quote"])
    df["timestamp"] = pd.to_datetime(df["updateTime"], unit="ms", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["executedQty"] = pd.to_numeric(df["executedQty"], errors="coerce")
    df["liq_quote"] = df["price"] * df["executedQty"]
    return df.set_index("timestamp")[["side","price","executedQty","liq_quote"]].dropna().sort_index().loc[start_ts:end_ts]

# ------------------------------ __main__ ------------------------------ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Derivatives data window fetcher (Binance)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--end-ts", default="now", help="ISO8601 or 'now'")
    parser.add_argument("--lookback-min", type=int, default=360)
    args = parser.parse_args()

    end_ts = pd.Timestamp.utcnow().tz_convert("UTC") if args.end_ts == "now" else pd.to_datetime(args.end_ts, utc=True)

    print("PERP (futures) klines:")
    print(fetch_perp_klines_window(args.symbol, end_ts, args.lookback_min).tail(3), "\n")

    print("SPOT klines:")
    print(fetch_spot_klines_window(args.symbol, end_ts, args.lookback_min).tail(3), "\n")

    print("Funding prints:")
    print(fetch_funding_window(args.symbol, end_ts, args.lookback_min).tail(5), "\n")

    print("Global L/S ratio:")
    print(fetch_long_short_ratio_window(args.symbol, end_ts, args.lookback_min, period="5m").tail(5), "\n")

    print("Open interest stats:")
    print(fetch_open_interest_stats_window(args.symbol, end_ts, args.lookback_min, period="5m").tail(5), "\n")

    print("Liquidations:")
    print(fetch_liquidations_window(args.symbol, end_ts, args.lookback_min).tail(5), "\n")

    print("OK ✓")
