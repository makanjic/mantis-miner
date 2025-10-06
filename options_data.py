# options_data.py
# Deribit public API: "now"-only DVOL and option surface snapshot (no timestamp args)
from __future__ import annotations

import requests
import pandas as pd

DERIBIT = "https://www.deribit.com/api/v2"

def _get(path: str, params: dict) -> dict:
    r = requests.get(f"{DERIBIT}{path}", params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    return js.get("result", js)

def get_dvol_now(currency: str = "BTC") -> pd.DataFrame:
    """
    Fetch current DVOL (volatility index) for the currency.
    Returns a 1-row DataFrame indexed by the current UTC minute:
      columns: ['dvol_30d']
    """
    res = _get("/public/get_volatility_index_data", {"currency": currency})
    now = pd.Timestamp.utcnow().tz_localize("UTC").floor("T")
    # Typical structure: {"current": {"dvol": ...}, ...}
    if isinstance(res, dict) and "current" in res and "dvol" in res["current"]:
        val = float(res["current"]["dvol"])
    else:
        val = float(res.get("dvol", float("nan")))
    return pd.DataFrame({"dvol_30d": [val]}, index=[now])

def surface_snapshot_now(currency: str = "BTC", max_instruments: int = 2000) -> pd.DataFrame:
    """
    Fetch a "now" snapshot of the listed options:
      columns: ['instrument','expiry','type','strike','mark_iv','delta','underlying','snapshot_ts']
    Note: this is a current snapshot only (no historical backfill).
    """
    insts = _get("/public/get_instruments", {"currency": currency, "kind": "option", "expired": "false"})
    rows = []
    for inst in insts[:max_instruments]:
        name = inst["instrument_name"]
        typ = "C" if name.endswith("-C") else "P"
        strike = float(inst.get("strike", 0.0))
        expiry = pd.to_datetime(inst["expiration_timestamp"], unit="ms", utc=True)
        try:
            tk = _get("/public/ticker", {"instrument_name": name})
            mark_iv = float(tk.get("mark_iv")) if tk.get("mark_iv") is not None else float("nan")
            greeks = tk.get("greeks") or {}
            delta = float(greeks.get("delta")) if "delta" in greeks else float("nan")
            und = float(tk.get("underlying_price")) if tk.get("underlying_price") is not None else float("nan")
        except Exception:
            mark_iv, delta, und = float("nan"), float("nan"), float("nan")
        rows.append({
            "instrument": name, "expiry": expiry, "type": typ, "strike": strike,
            "mark_iv": mark_iv, "delta": delta, "underlying": und
        })
    df = pd.DataFrame(rows)
    df["snapshot_ts"] = pd.Timestamp.utcnow().tz_localize("UTC").floor("T")
    return df.sort_values(["expiry", "strike"]).reset_index(drop=True)

# ------------------------------ __main__ ------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deribit options (now-only) fetcher")
    parser.add_argument("--currency", default="BTC", help="BTC or ETH")
    parser.add_argument("--max", type=int, default=400, help="Max instruments to pull for the surface snapshot")
    args = parser.parse_args()

    print("DVOL (now):")
    try:
        dvol = get_dvol_now(args.currency)
        print(dvol.tail(1))
    except Exception as e:
        print("DVOL error:", e)

    print("\nSurface snapshot (now, head):")
    try:
        surf = surface_snapshot_now(args.currency, max_instruments=args.max)
        print(surf.head(10))
    except Exception as e:
        print("Surface error:", e)

    print("\nOK ✓")
