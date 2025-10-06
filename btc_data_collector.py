# btc_data_collector.py
# Collect ALL inputs required to build the BTC embedding vector at a caller-supplied ts.
# No block logic here. The caller aligns to Bittensor blocks and passes ts.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

# ----- Your data source modules (already implemented earlier) -----
from features_helpers import utc_floor_minute
from cache_utils import MinuteCache

from ohlcv_data import get_ohlcv_window_for_symbol
from derivs_data import (
    fetch_perp_klines_window,
    fetch_spot_klines_window,
    fetch_funding_window,
    fetch_long_short_ratio_window,
    fetch_open_interest_stats_window,
    fetch_liquidations_window,
)
from options_data import get_dvol_now
from mempool_data import mempool_snapshot_now
from sentiment_data import cryptopanic_counts_window
from macro_data import load_macro_window


@dataclass
class CollectorConfig:
    # Lookbacks for windowed sources
    ohlcv_lookback_min: int = 5 * 24 * 60    # BTC OHLCV window
    derivs_lookback_min: int = 24 * 60       # funding/lsr/oi/liqs windows
    macro_lookback_min: int  = 24 * 60       # yfinance macro window
    # Providers
    exchange: str = "binance"
    # Sentiment filters to count (CryptoPanic); features will z-score series independently
    cryptopanic_filters: tuple[str, ...] = ("bullish", "bearish", "important")


@dataclass
class BTCInputs:
    """
    Unified container of all inputs needed by your feature builders at timestamp ts.
    """
    ts: pd.Timestamp
    ohlcv_btc_1m: pd.DataFrame
    derivs_caches: Dict[str, pd.Series]
    options_caches: Dict[str, pd.Series]
    mempool_caches: Dict[str, pd.Series]
    sentiment_caches: Dict[str, pd.Series]
    macro_caches: Dict[str, pd.Series]


@dataclass
class FeatureCaches:
    """Rolling caches for minute-indexed inputs (especially now-only feeds)."""
    derivs: MinuteCache = MinuteCache(max_minutes=7 * 24 * 60)
    options: MinuteCache = MinuteCache(max_minutes=7 * 24 * 60)
    mempool: MinuteCache = MinuteCache(max_minutes=24 * 60)
    sentiment: MinuteCache = MinuteCache(max_minutes=7 * 24 * 60)
    macro: MinuteCache = MinuteCache(max_minutes=7 * 24 * 60)


# ============================ Refreshers per source ============================

def refresh_derivs_windows(c: FeatureCaches, ts: pd.Timestamp, cfg: CollectorConfig):
    """Fetch derivatives series up to ts and store into caches."""
    # Perp / spot close (for basis)
    perp = fetch_perp_klines_window("BTCUSDT", ts, cfg.derivs_lookback_min)
    spot = fetch_spot_klines_window("BTCUSDT", ts, cfg.derivs_lookback_min)
    if not perp.empty:
        c.derivs.set_series("perp_close", perp["close"])
    if not spot.empty:
        c.derivs.set_series("spot_close", spot["close"])

    # Funding (8h prints)
    fund = fetch_funding_window("BTCUSDT", ts, cfg.derivs_lookback_min)
    if not fund.empty:
        c.derivs.set_series("funding", fund["fundingRate"])

    # Long/Short ratio (5m)
    lsr = fetch_long_short_ratio_window("BTCUSDT", ts, cfg.derivs_lookback_min, period="5m")
    if not lsr.empty:
        c.derivs.set_series("long_short", lsr["longShortRatio"])

    # Open interest history (5m)
    oih = fetch_open_interest_stats_window("BTCUSDT", ts, cfg.derivs_lookback_min, period="5m")
    if not oih.empty:
        c.derivs.set_series("open_interest", oih["sumOpenInterest"])

    # Liquidations (aggregate to per-minute net by side)
    liq = fetch_liquidations_window("BTCUSDT", ts, cfg.derivs_lookback_min)
    if not liq.empty:
        liq = liq.copy()
        liq["minute"] = liq.index.floor("T")
        grp = liq.groupby(["minute", "side"])["liq_quote"].sum().unstack(fill_value=0.0)
        c.derivs.set_series("liq_buy", grp.get("BUY", pd.Series(dtype=float)))
        c.derivs.set_series("liq_sell", grp.get("SELL", pd.Series(dtype=float)))


def refresh_options_now(c: FeatureCaches, ts: pd.Timestamp):
    """Append DVOL snapshot at ts (maintain rolling series)."""
    try:
        dvol_df = get_dvol_now("BTC")  # 1-row at 'now'
        if not dvol_df.empty:
            c.options.upsert("dvol_30d", ts, float(dvol_df["dvol_30d"].iloc[-1]))
    except Exception:
        pass
    # If you archive skew/term (iv_7d, iv_30d, skew_25d), upsert them here too.


def refresh_mempool_now(c: FeatureCaches, ts: pd.Timestamp):
    """Append mempool snapshot at ts; map to expected keys (vsize, fee proxy)."""
    try:
        snap = mempool_snapshot_now()
        if not snap.empty:
            row = snap.iloc[-1]
            c.mempool.upsert("mempool_vsize", ts, float(row.get("vsize", 0.0)))
            c.mempool.upsert("fee_median",   ts, float(row.get("hourFee", row.get("economyFee", 0.0))))
            # If you compute tx/s elsewhere, upsert("txps", ts, value)
    except Exception:
        pass


def refresh_sentiment_now(c: FeatureCaches, ts: pd.Timestamp, cfg: CollectorConfig):
    """Count CryptoPanic posts in the last minute ending at ts and append to series."""
    try:
        df = cryptopanic_counts_window(
            end_ts=ts, lookback_minutes=1, currency="BTC", include_filters=cfg.cryptopanic_filters
        )
        if not df.empty:
            row = df.iloc[-1]
            # Base buckets
            c.sentiment.upsert("news_bull",      ts, float(row.get("news_bull", 0.0)))
            c.sentiment.upsert("news_bear",      ts, float(row.get("news_bear", 0.0)))
            c.sentiment.upsert("news_important", ts, float(row.get("news_important", 0.0)))
            # Social volume proxy (media_total). If you have a richer social feed, replace this.
            c.sentiment.upsert("social_volume",  ts, float(row.get("media_total", 0.0)))
            # Social sentiment (polarity): keep 0 unless you compute a score elsewhere.
            c.sentiment.upsert("social_sentiment", ts, 0.0)
    except Exception:
        pass


def refresh_macro_windows(c: FeatureCaches, ts: pd.Timestamp, cfg: CollectorConfig):
    """Fetch macro window up to ts and set into cache as level series."""
    try:
        df = load_macro_window(ts, cfg.macro_lookback_min)
        if not df.empty:
            for col in df.columns:
                key = col.replace("_close", "")
                c.macro.set_series(key, df[col].rename(key))
    except Exception:
        pass


# ============================ Public one-shot API ============================

def collect_btc_inputs_at(
    ts,
    *,
    caches: Optional[FeatureCaches] = None,
    cfg: CollectorConfig = CollectorConfig(),
    prefetch_ohlcv: bool = True,
) -> BTCInputs:
    """
    Collect all inputs needed to build BTC features at timestamp `ts`.
    Returns a BTCInputs object with:
      - ohlcv_btc_1m  (DataFrame)
      - {derivs|options|mempool|sentiment|macro}_caches  (dict[str, Series])
    """
    ts = utc_floor_minute(ts)
    caches = caches or FeatureCaches()

    # Refresh windowed + snapshot sources
    refresh_derivs_windows(caches, ts, cfg)
    refresh_options_now(caches, ts)
    refresh_mempool_now(caches, ts)
    refresh_sentiment_now(caches, ts, cfg)
    refresh_macro_windows(caches, ts, cfg)

    # OHLCV (BTC) window (direct DataFrame for common/ohlcv features)
    ohlcv_df = pd.DataFrame()
    if prefetch_ohlcv:
        ohlcv_df = get_ohlcv_window_for_symbol("BTC", ts, cfg.ohlcv_lookback_min, exchange=cfg.exchange)

    return BTCInputs(
        ts=ts,
        ohlcv_btc_1m=ohlcv_df,
        derivs_caches=caches.derivs.as_mapping(),
        options_caches=caches.options.as_mapping(),
        mempool_caches=caches.mempool.as_mapping(),
        sentiment_caches=caches.sentiment.as_mapping(),
        macro_caches=caches.macro.as_mapping(),
    )


# ============================ __main__ smoke test ============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect all BTC inputs at a timestamp (no block logic).")
    parser.add_argument("--ts", default="now", help="UTC ISO8601 or 'now'")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--ohlcv-lookback-min", type=int, default=5*24*60)
    parser.add_argument("--derivs-lookback-min", type=int, default=24*60)
    parser.add_argument("--macro-lookback-min", type=int, default=24*60)
    args = parser.parse_args()

    ts = pd.Timestamp.utcnow().tz_localize("UTC").floor("T") if args.ts == "now" else utc_floor_minute(args.ts)

    cfg = CollectorConfig(
        ohlcv_lookback_min=args.ohlcv_lookback_min,
        derivs_lookback_min=args.derivs_lookback_min,
        macro_lookback_min=args.macro_lookback_min,
        exchange=args.exchange,
    )
    caches = FeatureCaches()

    inputs = collect_btc_inputs_at(ts, caches=caches, cfg=cfg, prefetch_ohlcv=True)

    print(f"\nCollected BTC inputs at {inputs.ts.isoformat()}:")
    print(f"- OHLCV rows: {len(inputs.ohlcv_btc_1m)}")
    for name, m in [
        ("derivs", inputs.derivs_caches),
        ("options", inputs.options_caches),
        ("mempool", inputs.mempool_caches),
        ("sentiment", inputs.sentiment_caches),
        ("macro", inputs.macro_caches),
    ]:
        keys = list(m.keys())
        print(f"- {name:<9}: {len(keys)} keys -> {keys[:6]}{' ...' if len(keys) > 6 else ''}")

    print("\nOK ✓")
