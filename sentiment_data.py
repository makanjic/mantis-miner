# sentiment_data.py
# CryptoPanic windowed fetcher with CSV append/de-dup.
from __future__ import annotations

import os
import time
import urllib.parse
from typing import Optional, Dict, Any, List

import pandas as pd
import requests


CRYPTOPANIC = "https://cryptopanic.com/api/v1/posts/"

# ------------------------------ helpers ------------------------------ #

def _utc_floor_min(ts) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True).floor("min")

def _trim_window(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    t = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.assign(__ts=t).dropna(subset=["__ts"])
    df = df.loc[(df["__ts"] >= start_ts) & (df["__ts"] <= end_ts)].drop(columns="__ts")
    return df

def _append_or_overwrite(path: str, df: pd.DataFrame, append: bool, key_cols: Optional[List[str]] = None) -> None:
    """
    Append with de-dup by key_cols (default: ['id']), else overwrite.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not append or not os.path.exists(path):
        df.to_csv(path, index=False)
        print(f"Wrote {path} ({len(df)} rows)")
        return

    try:
        old = pd.read_csv(path)
    except Exception:
        old = pd.DataFrame()

    merged = pd.concat([old, df], ignore_index=True)
    if key_cols and all(k in merged.columns for k in key_cols):
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
    merged = merged.sort_values(by=["published_at", "id"], na_position="last")
    merged.to_csv(path, index=False)
    print(f"Appended {len(df)} rows → {path} (total {len(merged)})")


# ------------------------------ fetcher ------------------------------ #

def fetch_cryptopanic_window(
    *,
    end_ts,
    lookback_minutes: int = 24 * 60,
    currency: str = "BTC",
    kind: str = "news",        # 'news' | 'media' | 'all'
    public: bool = True,
    auth_token: Optional[str] = None,
    max_pages: int = 5,
    pause_sec: float = 0.6,    # be gentle with rate limit
) -> pd.DataFrame:
    """
    Fetch CryptoPanic posts for [end - lookback, end].
    Returns a DataFrame with columns:
      id, published_at (UTC ISO), title, slug, source, domain, url, currencies,
      kind, created_at, votes_* (if present).
    """
    end_ts = _utc_floor_min(end_ts)
    start_ts = end_ts - pd.Timedelta(minutes=lookback_minutes)

    token = auth_token or os.getenv("CRYPTOPANIC_TOKEN", "").strip()
    if not token:
        raise RuntimeError("CryptoPanic token missing. Pass --auth-token or set CRYPTOPANIC_TOKEN env var.")

    params = {
        "auth_token": token,
        "currencies": currency.upper(),   # e.g., BTC
        "public": "true" if public else "false",
        # 'kind' is optional; supply only when not 'all'
    }
    if kind and kind.lower() in ("news", "media"):
        params["kind"] = kind.lower()

    # CryptoPanic does not support arbitrary time range in query reliably,
    # so we paginate backward and locally trim to the window.
    rows: List[Dict[str, Any]] = []

    url = CRYPTOPANIC + "?" + urllib.parse.urlencode(params)
    page = 0
    with requests.Session() as sess:
        while url and page < max_pages:
            r = sess.get(url, timeout=15)
            r.raise_for_status()
            js = r.json()
            print(f"res {page}: {js}")

            # results is usually a list of post dicts
            items = js.get("results") or js.get("data") or []
            for it in items:
                # Robust field extraction
                pid = it.get("id")
                pub = it.get("published_at") or it.get("date") or it.get("created_at")
                pub_iso = pd.to_datetime(pub, utc=True, errors="coerce")
                # quickly skip if way outside window (optional optimization)
                if pd.isna(pub_iso):
                    continue
                if pub_iso < (start_ts - pd.Timedelta(days=7)):  # hard stop if we're far behind
                    continue

                src = (it.get("source") or {}).get("title") or (it.get("source") or {}).get("name")
                dom = (it.get("source") or {}).get("domain")
                votes = it.get("votes") or {}
                row = {
                    "id": pid,
                    "published_at": pub_iso.isoformat(),
                    "title": it.get("title"),
                    "slug": it.get("slug"),
                    "url": it.get("url"),
                    "source": src,
                    "domain": dom,
                    "kind": it.get("kind"),
                    "created_at": it.get("created_at"),
                    "currencies": ",".join([c.get("code") for c in (it.get("currencies") or []) if c.get("code")]),
                    # votes breakdown (not always present)
                    "votes_negative": votes.get("negative"),
                    "votes_positive": votes.get("positive"),
                    "votes_important": votes.get("important"),
                    "votes_liked": votes.get("liked"),
                    "votes_disliked": votes.get("disliked"),
                    "votes_lol": votes.get("lol"),
                    "votes_toxic": votes.get("toxic"),
                }
                rows.append(row)

            # pagination: CryptoPanic replies with 'next' URL (absolute)
            nxt = js.get("next")
            url = nxt if nxt else None
            page += 1
            if url:
                time.sleep(pause_sec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Final clean & window trim
    df = df.dropna(subset=["id", "published_at"])
    df = _trim_window(df, start_ts, end_ts)
    # De-dup on id (keep last)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    df = df.sort_values(by=["published_at", "id"]).reset_index(drop=True)
    return df


# ------------------------------ __main__ ------------------------------ #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CryptoPanic sentiment/news window fetcher")
    ap.add_argument("--end-ts", default="now", help="ISO8601 or 'now' (UTC)")
    ap.add_argument("--lookback-min", type=int, default=6*60, help="Window size in minutes (default 6h)")
    ap.add_argument("--currency", default="BTC", help="Currency code for CryptoPanic filter (e.g., BTC)")
    ap.add_argument("--kind", default="news", help="'news' | 'media' | 'all' (default: news)")
    ap.add_argument("--public", action="store_true", help="Use public=True (default False if flag not set)")
    ap.add_argument("--auth-token", default=None, help="CryptoPanic API token (or set CRYPTOPANIC_TOKEN env var)")
    ap.add_argument("--max-pages", type=int, default=5, help="Max pagination pages to follow")
    ap.add_argument("--pause-sec", type=float, default=0.6, help="Sleep between pages to respect rate limits")
    # IO
    ap.add_argument("--save-to-csv", nargs="?", const=".", default=None,
                    help="Directory to save CSV (default '.' if flag present without a path)")
    ap.add_argument("--append", action="store_true", help="Append with de-dup by id (else overwrite)")
    ap.add_argument("--print", action="store_true", help="Print head/tail")
    args = ap.parse_args()

    end_ts = (_utc_floor_min(pd.Timestamp.now(tz="UTC"))
              if args.end_ts == "now" else _utc_floor_min(pd.to_datetime(args.end_ts, utc=True)))

    try:
        df = fetch_cryptopanic_window(
            end_ts=end_ts,
            lookback_minutes=args.lookback_min,
            currency=args.currency,
            kind=args.kind,
            public=args.public,
            auth_token=args.auth_token,
            max_pages=args.max_pages,
            pause_sec=args.pause_sec,
        )
    except Exception as e:
        print("Error:", e)
        df = pd.DataFrame()

    if args.print or True:
        if df.empty:
            print("No posts in window.")
        else:
            print(df.head(5))
            print("...")
            print(df.tail(5))
            print(f"Rows: {len(df)}")

    if args.save_to_csv is not None and not df.empty:
        out_dir = args.save_to_csv or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"sentiment_cryptopanic_{args.currency.upper()}.csv")
        _append_or_overwrite(path, df, append=args.append, key_cols=["id"])
