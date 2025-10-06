# macro_data.py
# Macro proxies window ending at SPECIFIED timestamp using yfinance start/end.
from __future__ import annotations

import pandas as pd
import yfinance as yf

# Choose your proxies here:
TICKERS = {
    "DXY": "^DXY",   # Dollar Index (alt: DX=F)
    "ES": "ES=F",    # S&P 500 E-mini
    "XAU": "GC=F",   # Gold
    "UST10Y": "^TNX",# 10Y yield *10 (45.3 => 4.53%)
    "UST2Y": "^FVX", # 5Y yield *10 (proxy if 2Y missing)
}

def _yf_range(ticker: str, start_ts, end_ts) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_ts, utc=True)
    end_ts   = pd.to_datetime(end_ts,   utc=True) + pd.Timedelta(minutes=1)  # include end minute
    df = yf.download(tickers=ticker, start=start_ts.tz_convert(None), end=end_ts.tz_convert(None),
                     interval="1m", progress=False, auto_adjust=False, prepost=False, threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.rename(columns=str.lower).set_index(idx)
    return df[["close"]].rename(columns={"close": f"{ticker}_close"})

def load_macro_window(end_ts, lookback_minutes: int = 24*60) -> pd.DataFrame:
    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)
    pieces = []
    for key, tick in TICKERS.items():
        d = _yf_range(tick, start_ts, end_ts)
        if not d.empty:
            d.columns = [f"{key}_close"]
            pieces.append(d)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).sort_index()
    return out.loc[start_ts:end_ts]

# ------------------------------ __main__ ------------------------------ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Macro window via yfinance (1m)")
    parser.add_argument("--end-ts", default="now", help="ISO or 'now'")
    parser.add_argument("--lookback-min", type=int, default=1440)
    args = parser.parse_args()

    end_ts = pd.Timestamp.utcnow().tz_localize("UTC") if args.end_ts == "now" else pd.to_datetime(args.end_ts, utc=True)
    df = load_macro_window(end_ts, args.lookback_min)
    if df.empty:
        print("No macro data (period may be outside 7d 1m window or markets closed).")
    else:
        print(df.tail(5))
        print("Columns:", list(df.columns))
    print("OK ✓")
