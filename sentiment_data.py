# sentiment_data.py
# CryptoPanic windowed counts at a SPECIFIED timestamp using paging.
from __future__ import annotations

import os
import requests
import pandas as pd
from typing import Literal, Optional, Dict

CP_API = "https://cryptopanic.com/api/v1/posts/"

def _fetch_page(currency: str, kind: str, token: str, page: int = 1, signal: str | None = None) -> dict:
    params = {"auth_token": token, "currencies": currency, "kind": kind, "public": "true", "regions": "en", "page": page}
    if signal:
        params["filter"] = signal
    r = requests.get(CP_API, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def cryptopanic_counts_window(
    end_ts,
    lookback_minutes: int = 60,
    currency: str = "BTC",
    include_filters: tuple[str, ...] = ("bullish", "bearish", "important"),
    max_pages: int = 10,
) -> pd.DataFrame:
    """
    Count posts within (end_ts - lookback_minutes, end_ts] for both 'news' and 'media',
    optionally split by CryptoPanic 'filter' tags (bullish/bearish/important).
    """
    token = os.environ.get("CRYPTOPANIC_TOKEN")
    if not token:
        raise RuntimeError("Set CRYPTOPANIC_TOKEN environment variable.")

    end_ts = pd.to_datetime(end_ts, utc=True).floor("T")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)

    def _count_for(kind: str, signal: str | None) -> int:
        total = 0
        page = 1
        while page <= max_pages:
            js = _fetch_page(currency, kind, token, page=page, signal=signal)
            results = js.get("results", [])
            if not results:
                break
            older = False
            for item in results:
                t = pd.to_datetime(item.get("published_at"), utc=True)
                if t <= end_ts and t > start_ts:
                    total += 1
                if t < start_ts:
                    older = True
            if older:
                break
            page += 1
        return total

    out = {"ts": [end_ts]}
    # totals
    out["news_total"]  = [_count_for("news", None)]
    out["media_total"] = [_count_for("media", None)]
    # filtered
    for f in include_filters:
        out[f"news_{f}"]  = [_count_for("news", f)]
        out[f"media_{f}"] = [_count_for("media", f)]
    df = pd.DataFrame(out).set_index("ts")
    return df

# ------------------------------ __main__ ------------------------------ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CryptoPanic windowed counts")
    parser.add_argument("--end-ts", default="now", help="ISO or 'now'")
    parser.add_argument("--lookback-min", type=int, default=60)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--filters", default="bullish,bearish,important")
    args = parser.parse_args()

    end_ts = pd.Timestamp.utcnow().tz_localize("UTC") if args.end_ts == "now" else pd.to_datetime(args.end_ts, utc=True)
    filters = tuple(x for x in args.filters.split(",") if x)

    df = cryptopanic_counts_window(end_ts, args.lookback_min, currency=args.currency, include_filters=filters)
    print(df)
    print("OK ✓")
