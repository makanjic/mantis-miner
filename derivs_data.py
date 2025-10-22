# derivs_data.py
# Fetch derivatives data up to a SPECIFIED timestamp (UTC) with CSV append/merge support.
from __future__ import annotations

import os
import requests
import pandas as pd
from datetime import timedelta
from typing import Optional

BINANCE_FAPI = "https://fapi.binance.com"  # USDⓈ-M Futures
BINANCE_SPOT = "https://api.binance.com"   # Spot

# ----------------------------- HTTP / time utils -----------------------------

def _to_ms(t) -> int:
    if isinstance(t, pd.Timestamp):
        return int(pd.Timestamp(t).tz_convert("UTC").timestamp() * 1000)
    if isinstance(t, (int, float)):
        return int(t if t > 10_000_000_000 else t * 1000)
    raise TypeError("Unsupported time type for _to_ms")

def _get(url: str, params: dict, timeout: int = 20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _floor_min(ts) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True).floor("min")

# ----------------------------- Fetch functions ------------------------------

def fetch_perp_klines_window(symbol: str, end_ts, lookback_minutes: int, interval: str = "1m") -> pd.DataFrame:
    end_ts = _floor_min(end_ts)
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
    end_ts = _floor_min(end_ts)
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "interval": interval, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts), "limit": 1500}
    js = _get(f"{BINANCE_SPOT}/api/v3/klines", params)
    if not js:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    columns=["t","o","h","l","c","v","ct","qv","n","taker_bv","taker_qv","i","b","a"]
    df = pd.DataFrame([row[:12] for row in js], columns=columns[:12])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    out = df.set_index("timestamp")[["o","h","l","c","v"]].astype(float).rename(
        columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}
    )
    return out.sort_index().loc[start_ts:end_ts]

def fetch_funding_window(symbol: str, end_ts, lookback_minutes: int, limit: int = 1000) -> pd.DataFrame:
    """Funding prints (8h cadence). Has startTime/endTime params."""
    end_ts = _floor_min(end_ts)
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts), "limit": limit}
    js = _get(f"{BINANCE_FAPI}/fapi/v1/fundingRate", params)
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame(columns=["fundingRate"])
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df.set_index("timestamp")[["fundingRate"]].astype(float).sort_index().loc[start_ts:end_ts]

def fetch_long_short_ratio_window(symbol: str, end_ts, lookback_minutes: int, period: str = "5m", limit: int = 500) -> pd.DataFrame:
    """Global Long/Short Account Ratio; supports startTime/endTime (≈30d history)."""
    end_ts = _floor_min(end_ts)
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
    """Historical OI stats (not just point-in-time OI). ≈30d availability."""
    end_ts = _floor_min(end_ts)
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    params = {"symbol": symbol, "period": period, "limit": limit, "startTime": _to_ms(start_ts), "endTime": _to_ms(end_ts)}
    js = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist", params)
    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame(columns=["sumOpenInterest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")[["sumOpenInterest"]].astype(float).sort_index().loc[start_ts:end_ts]

def fetch_liquidations_window(symbol: str, end_ts, lookback_minutes: int, limit: int = 1000) -> pd.DataFrame:
    """(Optional) Liquidations; can be noisy and rate-limited—left here for completeness."""
    end_ts = _floor_min(end_ts)
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

# ----------------------------- CSV helpers ----------------------------------

def _read_existing_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    ts_col = None
    for c in df.columns:
        if str(c).lower() == "timestamp":
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]
    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(__ts=idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()
    # keep all other columns as-is
    return df

def _last_ts_from_csv(path: str) -> Optional[pd.Timestamp]:
    if not path or not os.path.exists(path):
        return None
    try:
        df = _read_existing_csv(path)
        if df.empty:
            return None
        return pd.to_datetime(df.index.max()).tz_convert("UTC")
    except Exception:
        return None

def _merge_and_save(df_new: pd.DataFrame, path: str, append: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if append and os.path.exists(path):
        df_old = _read_existing_csv(path)
        merged = pd.concat([df_old, df_new], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(path, index_label="timestamp")
        print(f"Appended {len(df_new)} rows → {path} (total {len(merged)})")
    else:
        df_new.to_csv(path, index_label="timestamp")
        print(f"Wrote {path} ({len(df_new)} rows)")

# ----------------------------- __main__ -------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Derivatives data window fetcher (Binance) with CSV append.")
    parser.add_argument("--symbol", default="BTCUSDT", help="e.g., BTCUSDT, ETHUSDT")
    parser.add_argument("--end-ts", default="now", help="ISO8601 or 'now'")
    parser.add_argument("--lookback-min", type=int, default=360, help="Baseline lookback in minutes")
    parser.add_argument("--perp-interval", default="1m", help="Perp klines interval (e.g., 1m, 5m)")
    parser.add_argument("--spot-interval", default="1m", help="Spot klines interval (e.g., 1m, 5m)")
    parser.add_argument("--ls-period",    default="5m", help="Global L/S ratio period (e.g., 5m, 15m, 1h, 4h)")
    parser.add_argument("--oi-period",    default="5m", help="Open interest stats period (e.g., 5m, 15m, 1h, 4h)")

    # Save like ohlcv_data.py: optional value, defaults to current dir if flag present.
    parser.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                        help="If set, save each dataset to CSV in the given directory (default: current dir).")

    # Incremental append options
    parser.add_argument("--append", action="store_true",
                        help="Incremental: derive start from last timestamp in existing CSVs, refetch with overlap, and merge+dedup.")
    parser.add_argument("--overlap-minutes", type=int, default=120,
                        help="Safety overlap minutes when appending (re-fetch tail to fix boundary/late prints).")

    args = parser.parse_args()

    # Resolve end_ts
    end_ts = (pd.Timestamp.now(tz="UTC") if args.end_ts == "now"
              else pd.to_datetime(args.end_ts, utc=True)).floor("min")

    # Determine per-dataset output paths (if saving)
    out_dir = args.save_to_csv
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        perp_path = os.path.join(out_dir, f"{args.symbol}_perp_{args.perp_interval}.csv")
        spot_path = os.path.join(out_dir, f"{args.symbol}_spot_{args.spot_interval}.csv")
        fund_path = os.path.join(out_dir, f"{args.symbol}_funding.csv")
        lsr_path  = os.path.join(out_dir, f"{args.symbol}_lsratio_{args.ls_period}.csv")
        oi_path   = os.path.join(out_dir, f"{args.symbol}_oi_{args.oi_period}.csv")
    else:
        perp_path = spot_path = fund_path = lsr_path = oi_path = None

    # Helper: compute effective lookback for a dataset based on its CSV (if appending)
    def effective_lookback_minutes(path_for_dataset: Optional[str]) -> int:
        if not args.append:
            return int(args.lookback_min)
        last_ts = _last_ts_from_csv(path_for_dataset) if path_for_dataset else None
        if last_ts is None:
            return int(args.lookback_min)
        start_needed = (last_ts - pd.Timedelta(minutes=max(0, args.overlap_minutes))).floor("min")
        minutes = max(1, int((end_ts - start_needed).total_seconds() // 60))
        print(f"[append] {os.path.basename(path_for_dataset) if path_for_dataset else 'N/A'} "
              f"last={last_ts.isoformat()}, overlap={args.overlap_minutes}m → lookback={minutes}m")
        return minutes

    # Caps for endpoints with history limits (Binance docs ≈ 30 days typical)
    MAX_MINUTES_30D = 30 * 24 * 60

    # ---------------- Fetch each dataset with its own effective window ----------------

    print("PERP (futures) klines:")
    lb_perp = effective_lookback_minutes(perp_path)
    perp = fetch_perp_klines_window(args.symbol, end_ts, lb_perp, interval=args.perp_interval)
    print(perp.tail(3), "\n")

    print("SPOT klines:")
    lb_spot = effective_lookback_minutes(spot_path)
    spot = fetch_spot_klines_window(args.symbol, end_ts, lb_spot, interval=args.spot_interval)
    print(spot.tail(3), "\n")

    print("Funding prints:")
    lb_fund = effective_lookback_minutes(fund_path)
    funding = fetch_funding_window(args.symbol, end_ts, lb_fund)
    print(funding.tail(5), "\n")

    print("Global L/S ratio:")
    lb_lsr = min(effective_lookback_minutes(lsr_path), MAX_MINUTES_30D)
    ls = fetch_long_short_ratio_window(args.symbol, end_ts, lb_lsr, period=args.ls_period)
    print(ls.tail(5), "\n")

    print("Open interest stats:")
    lb_oi = min(effective_lookback_minutes(oi_path), MAX_MINUTES_30D)
    oi = fetch_open_interest_stats_window(args.symbol, end_ts, lb_oi, period=args.oi_period)
    print(oi.tail(5), "\n")

    # ---------------- Save (optional) ----------------
    if out_dir is not None:
        def _save(df: pd.DataFrame, path: Optional[str]):
            if path and df is not None and not df.empty:
                _merge_and_save(df, path, append=args.append)

        _save(perp,   perp_path)
        _save(spot,   spot_path)
        _save(funding,fund_path)
        _save(ls,     lsr_path)
        _save(oi,     oi_path)

    print("OK ✓")
