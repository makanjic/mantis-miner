# train_btc_encoder.py
# Train a 100-dim BTC encoder optimized for next-1h direction (logistic head).
# Outputs are fixed:
#   - encoder weights: btc_encoder.pth
#   - feature names:   feature_names.json  (can be overridden via --feature-names-out)
#
# Usage:
#   python train_btc_encoder.py \
#     --start 2025-09-20T00:00:00Z \
#     --end   2025-10-03T00:00:00Z \
#     --exchange binance \
#     --epochs 6 --batch-size 2048 \
#     --feature-names-out feature_names.json \
#     [--mempool-csv mempool_archive_1m.csv] \
#     [--options-csv options_archive_1m.csv] \
#     [--sentiment-csv sentiment_archive_1m.csv]

from __future__ import annotations

import os, math, json, time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---- your pipeline pieces ----
from btc_data_collector import (
    CollectorConfig, FeatureCaches, collect_btc_inputs_at,
    refresh_derivs_windows, refresh_macro_windows
)
from btc_features import build_btc_features_from_inputs, BTCBlocksConfig

opt_paths = {
    "mempool": args.mempool_csv if args.mempool_csv and os.path.exists(args.mempool_csv) else None,
    "options": args.options_csv if args.options_csv and os.path.exists(args.options_csv) else None,
    "sentiment": args.sentiment_csv if args.sentiment_csv and os.path.exists(args.sentiment_csv) else None,
}

# ------------------------------ utils ------------------------------ #

def utc_floor_min(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.floor("T")

def minute_range(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="1min", tz="UTC"))

def log(msg: str):
    print(msg, flush=True)

def try_load_archives(
    caches: FeatureCaches,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mempool_csv: Optional[str],
    options_csv: Optional[str],
    sentiment_csv: Optional[str],
):
    """Optionally load archived series into caches; silently skip if not provided."""
    if mempool_csv and os.path.exists(mempool_csv):
        try:
            df = pd.read_csv(mempool_csv, parse_dates=[0])
            if "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index().loc[start:end]
            if "vsize" in df.columns:
                caches.mempool.set_series("mempool_vsize", df["vsize"])
            if "hourFee" in df.columns:
                caches.mempool.set_series("fee_median", df["hourFee"])
            elif "economyFee" in df.columns:
                caches.mempool.set_series("fee_median", df["economyFee"])
            if "txps" in df.columns:
                caches.mempool.set_series("txps", df["txps"])
            log(f"[archive] mempool: {len(df)} rows")
        except Exception as e:
            log(f"[archive] mempool error: {e}")
    else:
        log("[archive] no mempool archive provided")
        caches.mempool = None  # disable mempool features

    if options_csv and os.path.exists(options_csv):
        try:
            df = pd.read_csv(options_csv, parse_dates=[0])
            if "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index().loc[start:end]
            for k in ["dvol_30d", "skew_25d", "iv_7d", "iv_30d"]:
                if k in df.columns:
                    caches.options.set_series(k, df[k])
            log(f"[archive] options: {len(df)} rows")
        except Exception as e:
            log(f"[archive] options error: {e}")
    else:
        log("[archive] no options archive provided")
        caches.options = None  # disable options features

    if sentiment_csv and os.path.exists(sentiment_csv):
        try:
            df = pd.read_csv(sentiment_csv, parse_dates=[0])
            if "timestamp" not in df.columns:
                df = df.rename(columns={df.columns[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index().loc[start:end]
            mapping = {
                "news_bull": "news_bull",
                "news_bear": "news_bear",
                "news_important": "news_important",
                "media_total": "social_volume",
                "social_sentiment": "social_sentiment",
            }
            for src, dst in mapping.items():
                if src in df.columns:
                    caches.sentiment.set_series(dst, df[src])
            log(f"[archive] sentiment: {len(df)} rows")
        except Exception as e:
            log(f"[archive] sentiment error: {e}")
    else:
        log("[archive] no sentiment archive provided")
        caches.sentiment = None  # disable sentiment features

# ------------------------------ dataset builder ------------------------------ #

def make_dataset(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    collector_cfg: CollectorConfig,
    block_cfg: BTCBlocksConfig,
    lookahead_min: int = 60,
    label_deadzone_bps: float = 0.0,
    mempool_csv: Optional[str] = None,
    options_csv: Optional[str] = None,
    sentiment_csv: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix X and labels y for start..end-60min, minute cadence.
    Label: y=1 if log close(t+60) - log close(t) > +deadzone; y=0 if < -deadzone; else drop.
    """
    caches = FeatureCaches()
    pipe_cfg = collector_cfg

    # Preload windowed derivatives & macro covering the full span
    total_minutes = int((end - start).total_seconds() // 60)
    end_ts = end
    tmp_collector = CollectorConfig(
        ohlcv_lookback_min=pipe_cfg.ohlcv_lookback_min,
        derivs_lookback_min=max(pipe_cfg.derivs_lookback_min, total_minutes),
        macro_lookback_min=max(pipe_cfg.macro_lookback_min, total_minutes),
        exchange=pipe_cfg.exchange,
        cryptopanic_filters=pipe_cfg.cryptopanic_filters,
    )
    log("[preload] fetching derivatives & macro windows ...")
    refresh_derivs_windows(caches, end_ts, tmp_collector)
    refresh_macro_windows(caches, end_ts, tmp_collector)

    # Optionally load now-only archives
    try_load_archives(caches, start, end, mempool_csv, options_csv, sentiment_csv)

    # OHLCV window to cover both features and labels
    need_back = tmp_collector.ohlcv_lookback_min
    total_need = total_minutes + lookahead_min
    ohlcv_need = max(need_back, total_need)
    from ohlcv_data import get_ohlcv_window_for_symbol
    ohlcv_df = get_ohlcv_window_for_symbol("BTC", end_ts=end_ts, lookback_minutes=ohlcv_need, exchange=pipe_cfg.exchange)
    if ohlcv_df.empty:
        raise RuntimeError("Failed to fetch BTC OHLCV.")
    close = ohlcv_df["close"].copy().sort_index()

    # Build samples
    times = minute_range(start, end - pd.Timedelta(minutes=lookahead_min))
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    names_ref: Optional[List[str]] = None

    from btc_features import build_btc_features_at

    for i, ts in enumerate(times):
        feat_vals, feat_names, _ = build_btc_features_at(
            ts,
            ohlcv_df=ohlcv_df,
            derivs_caches=caches.derivs.as_mapping(),
            options_caches=caches.options.as_mapping(),
            mempool_caches=caches.mempool.as_mapping(),
            sentiment_caches=caches.sentiment.as_mapping(),
            macro_caches=caches.macro.as_mapping(),
        )
        if names_ref is None:
            names_ref = feat_names

        t2 = ts + pd.Timedelta(minutes=lookahead_min)
        try:
            p0 = float(close.loc[:ts].iloc[-1])
            p1 = float(close.loc[:t2].iloc[-1])
        except Exception:
            continue
        ret = math.log(p1 / p0)
        dz = label_deadzone_bps / 10000.0  # bps → approx log threshold
        if ret > +dz:
            y = 1
        elif ret < -dz:
            y = 0
        else:
            continue

        X_list.append(np.asarray(feat_vals, dtype=np.float32))
        y_list.append(y)

        if (i + 1) % 1000 == 0:
            log(f"[build] {i+1}/{len(times)} examples")

    if not X_list:
        raise RuntimeError("No training samples produced. Check data availability / period / archives.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    return X, y, names_ref or []

# ------------------------------ model ------------------------------ #

class Encoder(nn.Module):
    """MLP encoder -> tanh(100)."""
    def __init__(self, in_dim: int, out_dim: int = 100, dropout: float = 0.05):
        super().__init__()
        hid1 = max(128, in_dim // 2)
        hid2 = 128
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid1, hid2),
            nn.GELU(),
            nn.LayerNorm(hid2),
            nn.Linear(hid2, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderWithLR(nn.Module):
    """Encoder + linear logistic head."""
    def __init__(self, in_dim: int, emb_dim: int = 100, dropout: float = 0.05):
        super().__init__()
        self.encoder = Encoder(in_dim, emb_dim, dropout)
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        logit = self.head(z).squeeze(-1)
        return z, logit

class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ------------------------------ training loop ------------------------------ #

@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 2048
    lr: float = 3e-4
    weight_decay: float = 1e-4
    head_l2: float = 1e-3
    val_split: float = 0.2
    shuffle_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight: Optional[float] = None
    early_stop_patience: int = 3

def train_encoder(X: np.ndarray, y: np.ndarray, cfg: TrainConfig, emb_dim: int = 100) -> Tuple[Encoder, Dict[str, float]]:
    ds = ArrayDataset(X, y)
    n_val = int(len(ds) * cfg.val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.shuffle_seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    model = EncoderWithLR(in_dim=X.shape[1], emb_dim=emb_dim).to(cfg.device)

    p = float(y.mean())
    pos_weight = cfg.pos_weight if cfg.pos_weight is not None else ( (1-p)/max(p,1e-6) if 0<p<1 else None )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(cfg.device) if pos_weight else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state, patience = float("inf"), None, cfg.early_stop_patience
    for epoch in range(1, cfg.epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            _, logit = model(xb)
            bce = criterion(logit, yb)
            l2_head = sum((p_.pow(2).sum() for _, p_ in model.head.named_parameters()))
            loss = bce + cfg.head_l2 * l2_head
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                _, logit = model(xb)
                bce = criterion(logit, yb)
                l2_head = sum((p_.pow(2).sum() for _, p_ in model.head.named_parameters()))
                va_loss += (bce + cfg.head_l2 * l2_head).item() * xb.size(0)
        va_loss /= len(val_ds)
        log(f"[epoch {epoch}] train_loss={tr_loss:.5f}  val_loss={va_loss:.5f}")

        if va_loss < best_val - 1e-5:
            best_val, best_state, patience = va_loss, {k:v.cpu().clone() for k,v in model.state_dict().items()}, cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                log("[early-stop] no improvement"); break

    if best_state is not None:
        model.load_state_dict(best_state)

    enc = model.encoder.to("cpu").eval()
    metrics = {"val_loss": best_val, "train_size": len(train_ds), "val_size": len(val_ds), "pos_frac": p}
    return enc, metrics

# ------------------------------ main ------------------------------ #

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train a 100-dim BTC encoder for 1h direction.")
    ap.add_argument("--start", required=True, help="Start UTC (ISO8601)")
    ap.add_argument("--end", required=True, help="End UTC (ISO8601)")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-l2", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-deadzone-bps", type=float, default=0.0)
    # fixed filename for encoder weights (no CLI override)
    ap.add_argument("--feature-names-out", default="feature_names.json", help="Path to save feature names")
    ap.add_argument("--mempool-csv", default=None)
    ap.add_argument("--options-csv", default=None)
    ap.add_argument("--sentiment-csv", default=None)
    args = ap.parse_args()

    start = utc_floor_min(args.start)
    end   = utc_floor_min(args.end)
    if end <= start + pd.Timedelta(hours=2):
        raise SystemExit("End must be at least 2 hours after start to generate labels.")

    collector_cfg = CollectorConfig(
        ohlcv_lookback_min=5*24*60,
        derivs_lookback_min=24*60,
        macro_lookback_min=24*60,
        exchange=args.exchange,
    )
    block_cfg = BTCBlocksConfig()

    X, y, names = make_dataset(
        start, end,
        collector_cfg=collector_cfg,
        block_cfg=block_cfg,
        lookahead_min=60,
        label_deadzone_bps=args.label_deadzone_bps,
        mempool_csv=args.mempool_csv,
        options_csv=args.options_csv,
        sentiment_csv=args.sentiment_csv,
    )
    log(f"[data] X={X.shape}, pos_frac={y.mean():.3f}")

    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        head_l2=args.head_l2, weight_decay=args.weight_decay
    )
    encoder, metrics = train_encoder(X, y, tcfg)

    # ---- Fixed output filename ----
    weights_path = "btc_encoder.pth"
    torch.save(encoder.state_dict(), weights_path)
    with open(args.feature_names_out, "w") as f:
        json.dump(names, f, indent=2)

    log(f"[done] saved encoder -> {weights_path}")
    log(f"[done] saved feature names -> {args.feature_names_out}")
    log(f"[metrics] {metrics}")

if __name__ == "__main__":
    main()
