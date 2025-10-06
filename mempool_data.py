# mempool_data.py
# mempool.space public API: "now"-only snapshot (no timestamp args)
from __future__ import annotations

import requests
import pandas as pd

MEMPOOL = "https://mempool.space/api"
MEMPOOL_V1 = "https://mempool.space/api/v1"

def fee_recommendations_now() -> pd.DataFrame:
    """
    Recommended fees (sat/vB) at "now".
    Returns 1-row DataFrame indexed by current UTC minute:
      ['fastestFee','halfHourFee','hourFee','economyFee','minimumFee']
    """
    r = requests.get(f"{MEMPOOL_V1}/fees/recommended", timeout=10)
    r.raise_for_status()
    js = r.json()
    now = pd.Timestamp.utcnow().tz_convert("UTC").floor("T")
    df = pd.DataFrame({k: [float(v)] for k, v in js.items()}, index=[now])
    return df

def mempool_summary_now() -> pd.DataFrame:
    """
    Mempool-level stats at "now":
      ['tx_count','vsize','total_fee']  (vsize: vMB, total_fee: sat)
    """
    r = requests.get(f"{MEMPOOL}/mempool", timeout=10)
    r.raise_for_status()
    js = r.json()
    now = pd.Timestamp.utcnow().tz_convert("UTC").floor("T")
    return pd.DataFrame({
        "tx_count":  [float(js.get("count", 0.0))],
        "vsize":     [float(js.get("vsize", 0.0))],
        "total_fee": [float(js.get("total_fee", 0.0))],
    }, index=[now])

def mempool_snapshot_now() -> pd.DataFrame:
    """
    Convenience: merge fees + summary into one 1-row frame.
    """
    fees = fee_recommendations_now()
    summ = mempool_summary_now()
    out = fees.join(summ, how="outer")
    return out

# ------------------------------ __main__ ------------------------------ #
if __name__ == "__main__":
    print("Fee recommendations (now):")
    try:
        fees = fee_recommendations_now()
        print(fees.tail(1))
    except Exception as e:
        print("fees error:", e)

    print("\nMempool summary (now):")
    try:
        summ = mempool_summary_now()
        print(summ.tail(1))
    except Exception as e:
        print("summary error:", e)

    print("\nMerged snapshot (now):")
    try:
        snap = mempool_snapshot_now()
        print(snap.tail(1))
    except Exception as e:
        print("snapshot error:", e)

    print("\nOK ✓")
