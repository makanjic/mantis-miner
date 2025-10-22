# options_data.py
# Deribit public API utilities:
#   - DVOL: windowed fetch (requires start_timestamp, end_timestamp, resolution)
#   - Options surface (fast): single-request bulk snapshot with optional moneyness/expiry filters
from __future__ import annotations

import os
import math
from typing import Optional, List, Dict, Any

import requests
import pandas as pd

DERIBIT = "https://www.deribit.com/api/v2"

# ------------------------------- HTTP helpers -------------------------------

def _to_ms(ts) -> int:
    ts = pd.to_datetime(ts, utc=True)
    return int(ts.timestamp() * 1000)

# ------------------------------- DVOL (window) -------------------------------

def fetch_dvol_window(end_ts, lookback_minutes: int, currency: str = "BTC", resolution: str = "60") -> pd.DataFrame:
    """
    Fetch Deribit DVOL candles for [end - lookback, end] at given resolution.
    resolution ∈ {"1","60","3600","43200","1D"} (per Deribit).
    Returns a DataFrame indexed by UTC with one column 'dvol' (close of each candle).
    """
    end_ts = pd.to_datetime(end_ts, utc=True).floor("min")
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)

    params = {
        "currency": currency.upper(),
        "start_timestamp": _to_ms(start_ts),
        "end_timestamp": _to_ms(end_ts),
        "resolution": resolution,
    }
    with requests.Session() as sess:
        r = sess.get(f"{DERIBIT}/public/get_volatility_index_data", params=params, timeout=20)
        r.raise_for_status()
        js = r.json()

    res = js.get("result") or {}
    data = res.get("data") or []

    if not data:
        return pd.DataFrame(columns=["dvol"])

    # data rows: [timestamp_ms, open, high, low, close]
    cols = ["timestamp", "open", "high", "low", "close"]
    df = pd.DataFrame(data, columns=cols[:len(data[0])])
    if "timestamp" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(columns=["dvol"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    out = df[["close"]].rename(columns={"close": "dvol"}).dropna()
    # hard trim to window for safety
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    return out

# ------------------------ Option surface snapshot (fast) ----------------------

def surface_snapshot_now_minimal(
    currency: str = "BTC",
    *,
    moneyness_pct: float = 0.25,  # set 0 to disable moneyness filter
    max_rows: int = 1000,         # cap after filtering (0 = disable cap)
    max_expiries: int = 0         # 0 = keep all expiries; else keep nearest N
) -> pd.DataFrame:
    """
    Fast bulk snapshot using get_book_summary_by_currency.
    - Single HTTP call (no per-instrument ticker).
    - Parses strike from instrument_name for optional moneyness filtering.
    - Optional nearest-expiries filter and row cap.

    Returns columns: instrument, expiry (if available), strike, mark_iv (if available),
                     underlying, snapshot_ts
    """
    def _parse_strike_from_name(name: str):
        # e.g. "BTC-25SEP26-180000-C" -> 180000
        try:
            parts = str(name).split("-")
            return float(parts[-2])
        except Exception:
            return float("nan")

    with requests.Session() as sess:
        r = sess.get(
            f"{DERIBIT}/public/get_book_summary_by_currency",
            params={"currency": currency.upper(), "kind": "option"},
            timeout=20,
        )
        r.raise_for_status()
        js = r.json()
        rows = js.get("result", [])
        if not rows:
            return pd.DataFrame(columns=["instrument","expiry","strike","mark_iv","underlying","snapshot_ts"])

        df = pd.DataFrame(rows)

        # Core fields
        df["instrument"] = df.get("instrument_name", df.get("instrument", ""))

        # expiry if available
        if "expiration_timestamp" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiration_timestamp"], unit="ms", utc=True, errors="coerce")
        else:
            df["expiry"] = pd.NaT

        # underlying & mark_iv if present
        df["underlying"] = pd.to_numeric(df.get("underlying_price"), errors="coerce")
        df["mark_iv"] = pd.to_numeric(df.get("mark_iv"), errors="coerce") if "mark_iv" in df.columns else pd.NA

        # strike (summary sometimes lacks explicit strike field; parse from name if needed)
        if "strike" in df.columns:
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        else:
            df["strike"] = df["instrument"].apply(_parse_strike_from_name)

        # keep essential rows
        df = df.dropna(subset=["instrument", "strike"])
        if df.empty:
            return pd.DataFrame(columns=["instrument","expiry","strike","mark_iv","underlying","snapshot_ts"])

        # moneyness filter (only if we have a usable spot and pct > 0)
        spot = float(df["underlying"].median()) if "underlying" in df.columns else float("nan")
        if moneyness_pct and math.isfinite(spot) and spot > 0:
            lo, hi = spot * (1.0 - moneyness_pct), spot * (1.0 + moneyness_pct)
            filtered = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
            # if filter removes everything (e.g., spot missing), fall back to unfiltered
            if not filtered.empty:
                df = filtered

        # optional: keep nearest N expiries if we have expiry info
        if max_expiries and "expiry" in df.columns and df["expiry"].notna().any():
            exps = sorted(pd.to_datetime(df["expiry"]).dropna().unique())[:max_expiries]
            if len(exps) > 0:
                df = df[df["expiry"].isin(exps)]

        # final ordering and cap
        df = df.sort_values(["expiry", "strike"], na_position="last")
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)

        out = df[["instrument","expiry","strike","mark_iv","underlying"]].copy()
        out["snapshot_ts"] = pd.Timestamp.now(tz="UTC").floor("min")
        return out.reset_index(drop=True)

# ------------------------------ CSV utilities -------------------------------

def _append_or_overwrite(path: str, df: pd.DataFrame, append: bool, index_label: Optional[str] = None) -> None:
    """
    Append with de-dup by timestamp index (if index_label is provided and matches df.index),
    else fall back to row-wise de-dup for the surface snapshot (instrument + snapshot_ts).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not append or not os.path.exists(path):
        if index_label:
            df.to_csv(path, index_label=index_label)
        else:
            df.to_csv(path, index=False)
        print(f"Wrote {path} ({len(df)} rows)")
        return

    # append mode
    try:
        old = pd.read_csv(path)
    except Exception:
        old = pd.DataFrame()

    if index_label and df.index.name == index_label:
        # index-based time series (DVOL)
        old_ts_col = None
        for c in old.columns:
            if str(c).lower() == index_label.lower():
                old_ts_col = c
                break
        if old_ts_col:
            old_idx = pd.to_datetime(old[old_ts_col], utc=True, errors="coerce")
            old = old.assign(__ts=old_idx).dropna(subset=["__ts"]).set_index("__ts").sort_index()
            merged = pd.concat([old, df], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_csv(path, index_label=index_label)
            print(f"Appended {len(df)} rows → {path} (total {len(merged)})")
            return

    # fallback: surface table de-dup by instrument + snapshot_ts
    merged = pd.concat([old, df], ignore_index=True)
    if "snapshot_ts" in merged.columns:
        merged["snapshot_ts"] = pd.to_datetime(merged["snapshot_ts"], utc=True, errors="coerce")
        merged = merged.dropna(subset=["snapshot_ts"])
        # include instrument + snapshot_ts; if duplicates, keep last
        merged = merged.drop_duplicates(subset=["instrument", "snapshot_ts"], keep="last")
        merged = merged.sort_values(["snapshot_ts", "expiry", "strike"], na_position="last")
    merged.to_csv(path, index=False)
    print(f"Appended {len(df)} rows → {path} (total {len(merged)})")

# --------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Deribit options data (DVOL window + fast surface snapshot)")
    # DVOL
    ap.add_argument("--currency", default="BTC", help="BTC or ETH")
    ap.add_argument("--end-ts", default="now", help="ISO8601 or 'now' (UTC) for DVOL window end")
    ap.add_argument("--lookback-min", type=int, default=1440, help="DVOL window size in minutes")
    ap.add_argument("--resolution", default="60",
                    help="DVOL resolution: one of {'1','60','3600','43200','1D'} (default: '60' = 1-minute)")
    # Surface (fast, minimal)
    ap.add_argument("--moneyness-pct", type=float, default=0.25,
                    help="Keep strikes within ±pct of underlying (0.25=±25%; 0 disables filter)")
    ap.add_argument("--max-rows", type=int, default=1000, help="Cap rows after filtering (0 disables cap)")
    ap.add_argument("--max-expiries", type=int, default=0, help="Keep nearest N expiries (0 keeps all)")
    # IO
    ap.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                    help="Directory to save CSVs (default '.' if flag present without a path)")
    ap.add_argument("--append", action="store_true", help="Append with de-dup (else overwrite)")
    ap.add_argument("--print", action="store_true", help="Print fetched data (tails)")
    args = ap.parse_args()

    # Resolve times
    end_ts = (pd.Timestamp.now(tz="UTC").floor("min")
              if args.end_ts == "now" else pd.to_datetime(args.end_ts, utc=True).floor("min"))
    cur = args.currency.upper()

    # --- DVOL window ---
    print("DVOL (window):")
    try:
        dvol = fetch_dvol_window(end_ts, args.lookback_min, currency=cur, resolution=args.resolution)
        if dvol.empty:
            print("No DVOL data in window.")
        else:
            if args.print or True:
                print(dvol.tail(5))
    except requests.HTTPError as e:
        print("DVOL error:", e)
        dvol = pd.DataFrame(columns=["dvol"])

    # --- Surface snapshot (fast, minimal) ---
    print("\nSurface snapshot (now, head):")
    try:
        surf = surface_snapshot_now_minimal(
            cur,
            moneyness_pct=args.moneyness_pct,
            max_rows=args.max_rows,
            max_expiries=args.max_expiries,
        )
        if args.print or True:
            print(surf.head(10))
    except Exception as e:
        print("Surface error:", e)
        surf = pd.DataFrame(columns=["instrument","expiry","strike","mark_iv","underlying","snapshot_ts"])

    # --- Save if requested ---
    if args.save_to_csv is not None:
        out_dir = args.save_to_csv or "."
        os.makedirs(out_dir, exist_ok=True)

        # DVOL → <CUR>_dvol.csv (timestamp index)
        if not dvol.empty:
            dvol_path = os.path.join(out_dir, f"{cur}_dvol.csv")
            _append_or_overwrite(dvol_path, dvol, append=args.append, index_label="timestamp")

        # Surface → <CUR>_surface.csv (wide table)
        if not surf.empty:
            surf_path = os.path.join(out_dir, f"{cur}_surface.csv")
            _append_or_overwrite(surf_path, surf, append=args.append, index_label=None)

    print("\nOK ✓")
