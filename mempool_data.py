# mempool_data.py
# mempool.space public API: "now"-only snapshot with CSV save/append + dedup support.
from __future__ import annotations

import os
import requests
import pandas as pd
from typing import Optional

MEMPOOL = "https://mempool.space/api"
MEMPOOL_V1 = "https://mempool.space/api/v1"


# ----------------------------- Fetchers -----------------------------

def _now_utc_min() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("min")

def fee_recommendations_now() -> pd.DataFrame:
    """
    Recommended fees (sat/vB) at 'now'.
    Returns 1-row DataFrame indexed by current UTC minute with:
      ['fastestFee','halfHourFee','hourFee','economyFee','minimumFee']
    """
    r = requests.get(f"{MEMPOOL_V1}/fees/recommended", timeout=10)
    r.raise_for_status()
    js = r.json()
    now = _now_utc_min()
    df = pd.DataFrame({k: [float(v)] for k, v in js.items()}, index=[now])
    return df

def mempool_summary_now() -> pd.DataFrame:
    """
    Mempool-level stats at 'now':
      ['tx_count','vsize','total_fee']  (vsize: vMB, total_fee: sat)
    """
    r = requests.get(f"{MEMPOOL}/mempool", timeout=10)
    r.raise_for_status()
    js = r.json()
    now = _now_utc_min()
    return pd.DataFrame({
        "tx_count":  [float(js.get("count", 0.0))],
        "vsize":     [float(js.get("vsize", 0.0))],
        "total_fee": [float(js.get("total_fee", 0.0))],
    }, index=[now])

def mempool_snapshot_now() -> pd.DataFrame:
    """
    Convenience: merge fees + summary into one 1-row frame (UTC, minute index).
    """
    fees = fee_recommendations_now()
    summ = mempool_summary_now()
    out = fees.join(summ, how="outer")
    return out


# ----------------------------- CSV helpers -----------------------------

def _read_existing_csv(path: str) -> pd.DataFrame:
    """
    Read an existing snapshot CSV. Robust to column order.
    Returns UTC-indexed DataFrame (timestamp index).
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # find timestamp column or default to first column
    ts_col = None
    for c in df.columns:
        if str(c).lower() == "timestamp":
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]

    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.assign(timestamp=idx).dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # coerce numeric where possible (ignore non-numeric columns)
    for c in df.columns:
        if c not in ("timestamp",):
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception as e:
                df[c] = 0.0
                print(f"Warning: non-numeric column {c!r} in {path}: {e}")
    return df

def _merge_and_save(new_df: pd.DataFrame, path: str, append: bool) -> None:
    """
    Append/overwrite with dedup by timestamp. Keeps all columns found.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Ensure index is UTC minute
    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")
    else:
        new_df.index = new_df.index.tz_convert("UTC")
    new_df.index = new_df.index.floor("min")

    if append and os.path.exists(path):
        old = _read_existing_csv(path)
        # Union columns (preserve order: old first, then any new columns)
        all_cols = list(dict.fromkeys(list(old.columns) + list(new_df.columns)))
        old = old.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)

        merged = pd.concat([old, new_df], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(path, index_label="timestamp")
        print(f"Appended 1 row → {path} (total {len(merged)})")
    else:
        new_df.to_csv(path, index_label="timestamp")
        print(f"Wrote {path} (1 row)")


# ----------------------------- __main__ -----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="mempool.space snapshot (now) with CSV save/append.")
    ap.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                    help="Directory to save snapshot CSV (default '.' if flag present without a path).")
    ap.add_argument("--append", action="store_true",
                    help="Append a single now-snapshot to CSV with dedup by timestamp (else overwrite).")
    ap.add_argument("--file-name", default="mempool_snapshot.csv",
                    help="Output file name (default: mempool_snapshot.csv).")
    ap.add_argument("--print", action="store_true", help="Print the snapshot to stdout.")
    args = ap.parse_args()

    # Fetch snapshot
    try:
        snap = mempool_snapshot_now()
    except Exception as e:
        print("snapshot error:", e)
        raise SystemExit(1)

    if args.print or args.save_to_csv is None:
        # Show most recent row
        print("Snapshot (now):")
        print(snap.tail(1))

    # Save if requested
    if args.save_to_csv is not None:
        out_dir = args.save_to_csv
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.file_name)
        _merge_and_save(snap, out_path, append=args.append)

    print("OK ✓")
