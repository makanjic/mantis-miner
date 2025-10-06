# sentiment_features.py
# Sentiment (CryptoPanic/news/social) feature extraction + live per-minute collector → CSV.
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from features_helpers import (
    utc_floor_minute,
    _prep_series, _leq, _resample_1m_ffill,
    _z_last, _clip01,
)
from cache_utils import MinuteCache


# ---------------- existing feature extractor ---------------- #

@dataclass(frozen=True)
class SentimentConfig:
    z_fast: int = 6 * 60  # ~6h
    freq:   str = "1min"

def extract_sentiment_features_at(
    caches: Mapping[str, pd.Series | pd.DataFrame],
    ts,
    cfg: SentimentConfig = SentimentConfig(),
) -> Tuple[List[float], List[str]]:
    """
    Expected optional keys:
      - "news_bull", "news_bear", "news_important"
      - "social_volume", "social_sentiment"
    """
    ts = utc_floor_minute(ts)

    def S(key: str) -> pd.Series:
        s = _prep_series(caches.get(key, pd.Series(dtype=float)))
        s = _leq(s, ts)
        return _resample_1m_ffill(s, ts, cfg.freq)

    news_bull = S("news_bull")
    news_bear = S("news_bear")
    news_imp  = S("news_important")
    soc_vol   = S("social_volume")
    soc_sent  = S("social_sentiment")

    names = [
        "news_bull_z_fast", "news_bear_z_fast", "news_important_z_fast",
        "social_volume_z_fast", "social_sentiment_z_fast",
    ]
    vals = [
        _z_last(news_bull, cfg.z_fast),
        _z_last(news_bear, cfg.z_fast),
        _z_last(news_imp,  cfg.z_fast),
        _z_last(soc_vol,   cfg.z_fast),
        _z_last(soc_sent,  cfg.z_fast),
    ]
    vals = [_clip01(float(v)) for v in vals]
    return vals, names


# ---------------- live collector ---------------- #

def collect_sentiment_features_once(ts: pd.Timestamp, cache: MinuteCache) -> Tuple[List[float], List[str]]:
    """
    Count CryptoPanic posts in the last minute and compute features.
    Requires sentiment_data.cryptopanic_counts_window().
    """
    from sentiment_data import cryptopanic_counts_window

    try:
        df = cryptopanic_counts_window(
            end_ts=ts, lookback_minutes=1, currency="BTC",
            include_filters=("bullish", "bearish", "important")
        )
        if not df.empty:
            r = df.iloc[-1]
            cache.upsert("news_bull",      ts, float(r.get("news_bull", 0.0)))
            cache.upsert("news_bear",      ts, float(r.get("news_bear", 0.0)))
            cache.upsert("news_important", ts, float(r.get("news_important", 0.0)))
            cache.upsert("social_volume",  ts, float(r.get("media_total", 0.0)))
            # If you have polarity, replace 0.0 below.
            cache.upsert("social_sentiment", ts, 0.0)
    except Exception:
        pass

    vals, names = extract_sentiment_features_at(cache.as_mapping(), ts, SentimentConfig())
    return vals, names

def _append_csv(path: str, row: Dict[str, Any]):
    import os
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)

if __name__ == "__main__":
    import argparse, time, os
    parser = argparse.ArgumentParser(description="Collect sentiment features each minute → CSV.")
    parser.add_argument("--csv", default="data/sentiment_features_1m.csv")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    cache = MinuteCache(max_minutes=7*24*60)
    print(f"[sentiment] writing features to {args.csv}")
    last = None
    while True:
        ts = utc_floor_minute(pd.Timestamp.utcnow())
        if last is None or ts > last:
            vals, names = collect_sentiment_features_once(ts, cache)
            row = {"timestamp": ts.isoformat()}
            row.update({n: v for n, v in zip(names, vals)})
            _append_csv(args.csv, row)
            print(f"[sentiment] {ts.isoformat()} → wrote {len(vals)} feats")
            last = ts
            if args.once: break
            now = pd.Timestamp.utcnow()
            time.sleep(max(0.0, 60 - (now.second + now.microsecond/1e6)) + 0.25)
        else:
            time.sleep(1.0)
