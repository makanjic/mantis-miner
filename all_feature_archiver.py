# all_feature_archiver.py
# Unified live archiver that collects RAW snapshots for:
#   - mempool (mempool.space-like metrics)
#   - options (DVOL + surface → skew_25d, iv_7d, iv_30d)
#   - sentiment (CryptoPanic minute counts)  [disabled by default]
#
# Writes per-source CSVs *and* an optional combined wide CSV (one row/minute).
# These CSVs match what train_btc_encoder.py expects via --mempool-csv/--options-csv/--sentiment-csv.

from __future__ import annotations

import os
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Your shared helper
from features_helpers import utc_floor_minute


# --------------------------- small IO helpers --------------------------- #

def utc_now_floor_min() -> pd.Timestamp:
    return utc_floor_minute(pd.Timestamp.utcnow())

def ensure_parent(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)

def append_row_csv(path: str, row: Dict[str, Any]):
    if not path:
        return
    ensure_parent(path)
    write_header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)

def sleep_until_next_minute():
    now = pd.Timestamp.utcnow()
    secs = 60 - (now.second + now.microsecond / 1e6)
    time.sleep(max(0.0, secs) + 0.25)


# --------------------------- collectors: RAW --------------------------- #

def collect_mempool(ts: pd.Timestamp) -> Dict[str, Any]:
    """
    Snapshot mempool raw metrics at ts.
    Expected fields from mempool_data.mempool_snapshot_now():
      tx_count, vsize, total_fee, fastestFee, halfHourFee, hourFee, economyFee, minimumFee
    """
    out = {
        "timestamp": ts.isoformat(),
        "tx_count": np.nan,
        "vsize": np.nan,
        "total_fee": np.nan,
        "fastestFee": np.nan,
        "halfHourFee": np.nan,
        "hourFee": np.nan,
        "economyFee": np.nan,
        "minimumFee": np.nan,
    }
    try:
        from mempool_data import mempool_snapshot_now
        snap = mempool_snapshot_now()
        if not snap.empty:
            r = snap.iloc[-1]
            out.update({
                "tx_count":   float(r.get("tx_count", np.nan)),
                "vsize":      float(r.get("vsize", np.nan)),
                "total_fee":  float(r.get("total_fee", np.nan)),
                "fastestFee": float(r.get("fastestFee", np.nan)),
                "halfHourFee":float(r.get("halfHourFee", np.nan)),
                "hourFee":    float(r.get("hourFee", np.nan)),
                "economyFee": float(r.get("economyFee", np.nan)),
                "minimumFee": float(r.get("minimumFee", np.nan)),
            })
    except Exception as e:
        print(f"[mempool] {ts.isoformat()} error: {e}")
    return out


def _compute_option_terms(surface_df: pd.DataFrame) -> dict:
    """
    From Deribit surface snapshot → approx:
      - iv_7d, iv_30d (ATM IV on nearest ~7d and ~30d tenors)
      - skew_25d     (30d 25Δ RR: IV_call_Δ≈+0.25 - IV_put_Δ≈-0.25)
    Returns {} if data insufficient.
    """
    out: dict = {}
    if surface_df is None or surface_df.empty:
        return out

    s = surface_df.copy()
    s["expiry"] = pd.to_datetime(s["expiry"], utc=True)
    now = s["snapshot_ts"].iloc[0]
    s["ttm_days"] = (s["expiry"] - now).dt.total_seconds() / 86400.0

    # ATM IV helper
    def atm_iv(target_days: float) -> Optional[float]:
        pool = s.loc[s["ttm_days"] > 0]
        if pool.empty:
            return None
        exp = pool.iloc[(pool["ttm_days"] - target_days).abs().argsort()].head(1)["expiry"].iloc[0]
        slc = s.loc[s["expiry"] == exp]
        if slc.empty:
            return None
        und = slc["underlying"].median()

        def nearest(opt_type: str) -> Optional[float]:
            sub = slc.loc[slc["type"] == opt_type]
            if sub.empty:
                return None
            sub = sub.assign(dist=(sub["strike"] - und).abs()).sort_values("dist")
            iv = sub["mark_iv"].iloc[0]
            return float(iv) if pd.notna(iv) else None

        c_iv = nearest("C")
        p_iv = nearest("P")
        if c_iv is None and p_iv is None:
            return None
        if c_iv is None:
            return p_iv
        if p_iv is None:
            return c_iv
        return 0.5 * (c_iv + p_iv)

    iv7  = atm_iv(7.0)
    iv30 = atm_iv(30.0)
    if iv7  is not None: out["iv_7d"]  = float(iv7)
    if iv30 is not None: out["iv_30d"] = float(iv30)

    # 25Δ risk reversal near 30d
    pool = s.loc[s["ttm_days"] > 0]
    if not pool.empty:
        exp = pool.iloc[(pool["ttm_days"] - 30.0).abs().argsort()].head(1)["expiry"].iloc[0]
        slc = s.loc[s["expiry"] == exp]
        if not slc.empty and "delta" in slc.columns:
            calls = slc.loc[slc["type"] == "C"].dropna(subset=["delta", "mark_iv"])
            puts  = slc.loc[slc["type"] == "P"].dropna(subset=["delta", "mark_iv"])
            if not calls.empty and not puts.empty:
                c = calls.iloc[(calls["delta"] - 0.25).abs().argsort()].head(1)
                p = puts.iloc[(puts["delta"] + 0.25).abs().argsort()].head(1)
                if not c.empty and not p.empty:
                    out["skew_25d"] = float(c["mark_iv"].iloc[0] - p["mark_iv"].iloc[0])

    return out


def collect_options(ts: pd.Timestamp) -> Dict[str, Any]:
    """
    Snapshot options raw metrics at ts:
      dvol_30d, skew_25d, iv_7d, iv_30d
    """
    out = {
        "timestamp": ts.isoformat(),
        "dvol_30d": np.nan,
        "skew_25d": np.nan,
        "iv_7d": np.nan,
        "iv_30d": np.nan,
    }
    try:
        from options_data import get_dvol_now
        dvol = get_dvol_now("BTC")
        if not dvol.empty:
            out["dvol_30d"] = float(dvol["dvol_30d"].iloc[-1])
    except Exception as e:
        print(f"[options:dvol] {ts.isoformat()} error: {e}")

    try:
        from options_data import surface_snapshot_now
        surf = surface_snapshot_now("BTC", max_instruments=600)
        feats = _compute_option_terms(surf)
        for k in ("skew_25d", "iv_7d", "iv_30d"):
            if k in feats:
                out[k] = feats[k]
    except Exception as e:
        # Surface might not be available; keep NaNs
        print(f"[options:surface] {ts.isoformat()} warn: {e}")

    return out


def collect_sentiment(ts: pd.Timestamp) -> Dict[str, Any]:
    """
    Snapshot sentiment counts for the last minute ending at ts.
    Requires sentiment_data.cryptopanic_counts_window().
    """
    out = {
        "timestamp": ts.isoformat(),
        "news_total": 0.0,
        "news_bull": 0.0,
        "news_bear": 0.0,
        "news_important": 0.0,
        "media_total": 0.0,
        "social_sentiment": 0.0,  # placeholder if you add polarity later
    }
    try:
        from sentiment_data import cryptopanic_counts_window
        df = cryptopanic_counts_window(
            end_ts=ts, lookback_minutes=1, currency="BTC",
            include_filters=("bullish", "bearish", "important")
        )
        if not df.empty:
            r = df.iloc[-1]
            out.update({
                "news_total":     float(r.get("news_total", 0.0)),
                "news_bull":      float(r.get("news_bull", 0.0)),
                "news_bear":      float(r.get("news_bear", 0.0)),
                "news_important": float(r.get("news_important", 0.0)),
                "media_total":    float(r.get("media_total", 0.0)),
            })
    except Exception as e:
        print(f"[sentiment] {ts.isoformat()} error: {e}")

    return out


# --------------------------- one-shot tick --------------------------- #

def collect_all_once(
    *,
    mempool_csv: Optional[str],
    options_csv: Optional[str],
    sentiment_csv: Optional[str],
    combined_csv: Optional[str],
    enable_mempool: bool,
    enable_options: bool,
    enable_sentiment: bool,
) -> None:
    ts = utc_now_floor_min()

    mem = collect_mempool(ts) if enable_mempool else None
    opt = collect_options(ts)  if enable_options  else None
    sen = collect_sentiment(ts) if enable_sentiment else None

    # write per-source
    if mem and mempool_csv:
        append_row_csv(mempool_csv, mem)
    if opt and options_csv:
        append_row_csv(options_csv, opt)
    if sen and sentiment_csv:
        append_row_csv(sentiment_csv, sen)

    # combined (wide) row
    if combined_csv:
        row = {"timestamp": ts.isoformat()}
        if mem:
            row.update({f"mempool_{k}": v for k, v in mem.items() if k != "timestamp"})
        if opt:
            row.update({f"options_{k}": v for k, v in opt.items() if k != "timestamp"})
        if sen:
            row.update({f"sent_{k}": v for k, v in sen.items() if k != "timestamp"})
        append_row_csv(combined_csv, row)

    print(f"[all] {ts.isoformat()} appended →"
          f"{' mempool' if mem else ''}{' options' if opt else ''}{' sentiment' if sen else ''}")


# ------------------------------- CLI -------------------------------- #

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Unified per-minute RAW archiver for mempool/options/sentiment.")
    ap.add_argument("--mempool-csv",   default="data/mempool_archive_1m.csv", help="'' to disable")
    ap.add_argument("--options-csv",   default="data/options_archive_1m.csv", help="'' to disable")
    ap.add_argument("--sentiment-csv", default="", help="'' to disable (default off)")
    ap.add_argument("--combined-csv",  default="data/features_1m.csv", help="'' to disable combined wide file")
    ap.add_argument("--no-mempool",    action="store_true", help="Disable mempool collection")
    ap.add_argument("--no-options",    action="store_true", help="Disable options collection")
    ap.add_argument("--enable-sentiment", action="store_true", help="Enable sentiment collection (off by default)")
    ap.add_argument("--once",          action="store_true", help="Run a single tick and exit")
    ap.add_argument("--sleep-sec",     type=float, default=1.0, help="Idle sleep while waiting for next minute")
    args = ap.parse_args()

    enable_mempool   = not args.no_mempool and bool(args.mempool_csv.strip())
    enable_options   = not args.no_options and bool(args.options_csv.strip())
    enable_sentiment = args.enable_sentiment and bool(args.sentiment_csv.strip())  # default OFF

    print("[archiver] live per-minute collection")
    print(f"  mempool  : {'ON' if enable_mempool else 'OFF'} → {args.mempool_csv or '(disabled)'}")
    print(f"  options  : {'ON' if enable_options else 'OFF'} → {args.options_csv or '(disabled)'}")
    print(f"  sentiment: {'ON' if enable_sentiment else 'OFF'} → {args.sentiment_csv or '(disabled)'}")
    print(f"  combined : {args.combined_csv or '(disabled)'}")

    last_written = None
    while True:
        now_min = utc_now_floor_min()
        if last_written is None or now_min > last_written:
            collect_all_once(
                mempool_csv=args.mempool_csv if enable_mempool else None,
                options_csv=args.options_csv if enable_options else None,
                sentiment_csv=args.sentiment_csv if enable_sentiment else None,
                combined_csv=(args.combined_csv if args.combined_csv.strip() else None),
                enable_mempool=enable_mempool,
                enable_options=enable_options,
                enable_sentiment=enable_sentiment,
            )
            last_written = now_min
            if args.once:
                break
            sleep_until_next_minute()
        else:
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
