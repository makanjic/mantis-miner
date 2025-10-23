# btc_features.py
# Build the full BTC feature vector at a specified timestamp by composing:
#   - Base OHLCV+time features (from common_features.py)
#   - BTC add-ons: derivatives, options, mempool, sentiment, macro
#
# Output: (values, names) with all values ∈ [-1, 1], in a fixed, documented order.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, List, Optional, Dict
import pandas as pd


# ---- Imports from your existing modules ----
from ohlcv_data import get_ohlcv_window_for_symbol
from common_features import extract_ohlcv_features_at, FeatureConfig

from derivs_features import extract_derivs_features_at, DerivsConfig
from options_features import extract_options_features_at, OptionsConfig
from mempool_features import extract_mempool_features_at, MempoolConfig
from sentiment_features import extract_sentiment_features_at, SentimentConfig
from macro_features import extract_macro_features_at, MacroConfig


@dataclass(frozen=True)
class BTCFeaturesConfig:
    """
    Configure window sizes and sub-configs for each feature block.
    - lookback_minutes: how much OHLCV history to fetch for base features (and to align caches by ts).
    - *Config: pass through to each sub-feature extractor.
    """
    lookback_minutes: int = 5 * 24 * 60  # 5 days of 1m OHLCV
    ohlcv: FeatureConfig = FeatureConfig()
    derivs: DerivsConfig = DerivsConfig()
    options: OptionsConfig = OptionsConfig()
    mempool: MempoolConfig = MempoolConfig()
    sentiment: SentimentConfig = SentimentConfig()
    macro: MacroConfig = MacroConfig()


def _clip_unit(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def build_btc_feature_vector_at(
    ts,
    *,
    cfg: BTCFeaturesConfig = BTCFeaturesConfig(),
    exchange: str = "binance",
    ohlcv_df: Optional[pd.DataFrame] = None,
    derivs_caches: Optional[Mapping[str, pd.Series]] = None,
    options_caches: Optional[Mapping[str, pd.Series]] = None,
    mempool_caches: Optional[Mapping[str, pd.Series]] = None,
    sentiment_caches: Optional[Mapping[str, pd.Series]] = None,
    macro_caches: Optional[Mapping[str, pd.Series]] = None,
) -> Tuple[List[float], List[str], Dict[str, Tuple[int, int]]]:
    """
    Build the BTC feature vector at timestamp `ts` (UTC).
    Parameters:
      ts                : pd.Timestamp | str | unix seconds
      cfg               : BTCFeaturesConfig with window sizes & sub-configs
      exchange          : CCXT exchange for BTC OHLCV (default: 'binance')
      ohlcv_df          : optional pre-fetched BTC OHLCV 1m DataFrame; if None we fetch
      *_caches          : dicts of minute-indexed pd.Series for each add-on block.
                          Missing or empty caches are handled (features → 0).
    Returns:
      (values, names, spans)
        values : List[float]   all in [-1, 1]
        names  : List[str]     aligned names
        spans  : dict mapping block name -> (start_idx, end_idx) slice in the final vector
                 helpful for debugging/ablation
    """
    ts = pd.to_datetime(ts, utc=True).floor("T")

    # --- 1) Base OHLCV+time features ---
    if ohlcv_df is None:
        ohlcv_df = get_ohlcv_window_for_symbol(
            "BTC", end_ts=ts, lookback_minutes=cfg.lookback_minutes, exchange=exchange
        )

    base_vals, base_names = extract_ohlcv_features_at(ohlcv_df, ts, cfg.ohlcv)
    vals: List[float] = [float(v) for v in base_vals]
    names: List[str] = list(base_names)
    spans: Dict[str, Tuple[int, int]] = {"ohlcv": (0, len(vals))}

    # --- 2) Derivatives ---
    if derivs_caches is not None:
        d_vals, d_names = extract_derivs_features_at(derivs_caches or {}, ts, cfg.derivs)
        names.extend(d_names); vals.extend(d_vals)
        spans["derivs"] = (spans["ohlcv"][1], spans["ohlcv"][1] + len(d_vals))

    # --- 3) Options ---
    if options_caches is not None:
        o_vals, o_names = extract_options_features_at(options_caches or {}, ts, cfg.options)
        names.extend(o_names); vals.extend(o_vals)
        spans["options"] = (spans["derivs"][1], spans["derivs"][1] + len(o_vals))

    # --- 4) Mempool ---
    if mempool_caches is not None:
        m_vals, m_names = extract_mempool_features_at(mempool_caches or {}, ts, cfg.mempool)
        names.extend(m_names); vals.extend(m_vals)
        spans["mempool"] = (spans["options"][1], spans["options"][1] + len(m_vals))

    # --- 5) Sentiment ---
    if sentiment_caches is not None:
        s_vals, s_names = extract_sentiment_features_at(sentiment_caches or {}, ts, cfg.sentiment)
        names.extend(s_names); vals.extend(s_vals)
        spans["sentiment"] = (spans["mempool"][1], spans["mempool"][1] + len(s_vals))

    # --- 6) Macro ---
    if macro_caches is not None:
        x_vals, x_names = extract_macro_features_at(macro_caches or {}, ts, cfg.macro)
        names.extend(x_names); vals.extend(x_vals)
        spans["macro"] = (spans["sentiment"][1], spans["sentiment"][1] + len(x_vals))

    # Safety: hard-clip to [-1, 1]
    vals = [_clip_unit(float(v)) for v in vals]

    return vals, names, spans


# ------------------------------ __main__ demo ------------------------------ #
if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Assemble BTC features at a timestamp")
    parser.add_argument("--ts", default="2025-10-03T08:14:00Z", help="UTC timestamp (ISO8601)")
    parser.add_argument("--exchange", default="binance", help="CCXT exchange for BTC OHLCV")
    parser.add_argument("--lookback-min", type=int, default=5*24*60, help="OHLCV lookback minutes")
    parser.add_argument("--live-ohlcv", action="store_true", help="Fetch real BTC OHLCV (requires ccxt/yfinance)")
    args = parser.parse_args()

    ts = pd.to_datetime(args.ts, utc=True)

    # --------- Prepare OHLCV (live or synthetic) ---------
    if args.live_ohlcv:
        df_btc = get_ohlcv_window_for_symbol("BTC", ts, args.lookback_min, exchange=args.exchange)
        if df_btc.empty:
            print("Live OHLCV fetch returned empty; falling back to synthetic.")
            args.live_ohlcv = False
    if not args.live_ohlcv:
        # Synthetic OHLCV (random walk with volatility)
        rng = np.random.default_rng(0)
        idx = pd.date_range(ts - pd.Timedelta(minutes=args.lookback_min-1), ts, freq="1min", tz="UTC")
        logp = np.cumsum(rng.normal(0, 0.002, size=len(idx)))
        close = 60000 * np.exp(logp)
        high = close * (1 + np.abs(rng.normal(0, 0.0015, size=len(idx))))
        low  = close * (1 - np.abs(rng.normal(0, 0.0015, size=len(idx))))
        open_ = np.r_[close[0], close[:-1]]
        vol = np.abs(rng.normal(50, 10, size=len(idx)))
        df_btc = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
        )

    # --------- Synthetic caches for add-on blocks (so __main__ works offline) ---------
    rng = np.random.default_rng(123)

    def synth_series(n=2000, start=None, vol=1.0, base=0.0, abspos=False):
        if start is None:
            start = (ts - pd.Timedelta(minutes=n-1)).isoformat()
        idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
        x = base + np.cumsum(rng.normal(0.0, vol, size=n))
        if abspos:
            x = np.abs(x)
        return pd.Series(x, index=idx)

    # Derivatives caches (keys expected by derivs_features.py)
    derivs_caches = {
        "funding": 0.0001 + 0.0002 * synth_series(2000, vol=0.01),
        "perp_close": 60000 + synth_series(2000, vol=30).cumsum().abs(),
        "spot_close": 59950 + synth_series(2000, vol=30).cumsum().abs(),
        "open_interest": 1.0e9 + 1.0e7 * synth_series(2000, vol=0.1),
        "long_short": 1.0 + 0.05 * synth_series(2000, vol=0.02),
        #"liq_buy": np.abs(synth_series(2000, vol=1.0)),
        #"liq_sell": np.abs(synth_series(2000, vol=1.1)),
    }

    # Options caches (keys expected by options_features.py)
    options_caches = {
        "dvol_30d": 0.55 + 0.02 * synth_series(2000, vol=0.02),
        "skew_25d": 0.00 + 0.01 * synth_series(2000, vol=0.02),
        "iv_7d":    0.58 + 0.02 * synth_series(2000, vol=0.02),
        "iv_30d":   0.52 + 0.02 * synth_series(2000, vol=0.02),
    }

    # Mempool caches (keys expected by mempool_features.py)
    mempool_caches = {
        "txps": synth_series(2000, vol=0.2, abspos=True),
        "mempool_vsize": 100 + synth_series(2000, vol=0.5, abspos=True),
        "fee_median": 25 + synth_series(2000, vol=0.05, abspos=True),
    }

    # Sentiment caches (keys expected by sentiment_features.py)
    # Use Poisson-like counts + a bounded sentiment index
    idx_sent = pd.date_range(ts - pd.Timedelta(minutes=1999), ts, freq="1min", tz="UTC")
    news_bull = pd.Series(rng.poisson(lam=1.0, size=len(idx_sent)).astype(float), index=idx_sent)
    news_bear = pd.Series(rng.poisson(lam=0.7, size=len(idx_sent)).astype(float), index=idx_sent)
    news_imp  = pd.Series(rng.poisson(lam=0.3, size=len(idx_sent)).astype(float), index=idx_sent)
    social_vol = pd.Series(rng.poisson(lam=10.0, size=len(idx_sent)).astype(float), index=idx_sent)
    social_sent = pd.Series(pd.Series(rng.normal(0, 0.02, size=len(idx_sent)), index=idx_sent).cumsum()).apply(lambda v: float(max(-1, min(1, v))))
    sentiment_caches = {
        "news_bull": news_bull, "news_bear": news_bear, "news_important": news_imp,
        "social_volume": social_vol, "social_sentiment": social_sent
    }

    # Macro caches (keys expected by macro_features.py)
    macro_caches = {
        "DXY": 105 + synth_series(2000, vol=0.01).cumsum().abs(),
        "ES":  5300 + synth_series(2000, vol=0.3).cumsum().abs(),
        "XAU": 2350 + synth_series(2000, vol=0.1).cumsum().abs(),
        "UST2Y": 4.90 + 0.001 * synth_series(2000, vol=0.2),
        "UST10Y":4.25 + 0.001 * synth_series(2000, vol=0.15),
    }

    # --------- Build the vector ---------
    btc_cfg = BTCFeaturesConfig(lookback_minutes=args.lookback_min)
    vals, names, spans = build_btc_feature_vector_at(
        ts,
        cfg=btc_cfg,
        exchange=args.exchange,
        ohlcv_df=df_btc,
        derivs_caches=derivs_caches,
        options_caches=options_caches,
        mempool_caches=mempool_caches,
        sentiment_caches=sentiment_caches,
        macro_caches=macro_caches,
    )

    print(f"BTC feature vector at {ts.isoformat()}")
    print(f"Total dims: {len(vals)}")
    for block, (a, b) in spans.items():
        print(f"  - {block:<10}: [{a:>3}, {b:>3})  -> {b-a} dims")

    # Quick peek at the first few & last few features
    print("\nFirst 8 features:")
    for n, v in list(zip(names, vals))[:8]:
        print(f"{n:>24}: {v:+.6f}")

    print("\nLast 8 features:")
    for n, v in list(zip(names, vals))[-8:]:
        print(f"{n:>24}: {v:+.6f}")

    # Range check
    mn, mx = min(vals), max(vals)
    print(f"\nRange check: min={mn:+.6f}, max={mx:+.6f} (should be within [-1, 1])")
    print("OK ✓")
