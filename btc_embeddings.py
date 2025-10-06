# btc_embedding.py
# Build a 100-dim BTC embedding vector (values ∈ [-1, 1]) at a specified UTC timestamp.
# - Uses your existing feature pipeline (btc_data_collector + btc_features)
# - If a PyTorch weights file is provided, uses that encoder; otherwise falls back to a
#   deterministic orthonormal projection (NumPy-only, no training needed).

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import numpy as np
import pandas as pd

# ---- your pipeline pieces ----
from btc_data_collector import collect_btc_inputs_at, FeatureCaches, CollectorConfig
from btc_features import build_btc_features_from_inputs, BTCBlocksConfig

# ---- optional: torch encoder (if available and weights provided) ----
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@dataclass(frozen=True)
class EncoderConfig:
    out_dim: int = 100
    # If you trained a torch encoder, pass path here; otherwise we use NumPy projection.
    weights_path: Optional[str] = None
    # Projection seed ensures deterministic embeddings without weights.
    proj_seed: int = 2025
    # Optional dropout used by torch model only (ignored for NumPy projection).
    dropout: float = 0.05


class _TorchBTCEncoder(nn.Module):
    """Small MLP encoder -> tanh(100) for [-1,1]."""
    def __init__(self, in_dim: int, out_dim: int = 100, dropout: float = 0.05):
        super().__init__()
        hid1, hid2 = max(128, in_dim // 2), 128
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid1, hid2),
            nn.GELU(),
            nn.LayerNorm(hid2),
            nn.Linear(hid2, out_dim),
            nn.Tanh(),  # ensure [-1,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _orthonormal_projection(in_dim: int, out_dim: int, seed: int = 2025) -> np.ndarray:
    """
    Deterministic orthonormal projection W ∈ R^{out_dim x in_dim}.
    We create a random Gaussian matrix with a fixed seed, QR-decompose, and take rows.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(in_dim, in_dim))
    # QR gives A = Q R, with Q orthonormal (in_dim x in_dim)
    Q, _ = np.linalg.qr(A)
    # Take first out_dim rows of Q^T  → shape (out_dim, in_dim)
    W = Q.T[:out_dim, :]
    return W.astype(np.float32)


def _embed_numpy(features: np.ndarray, cfg: EncoderConfig, proj_matrix_cache: dict) -> np.ndarray:
    """
    NumPy fallback: deterministic orthonormal projection + tanh.
    """
    in_dim = features.shape[-1]
    key = (in_dim, cfg.out_dim, cfg.proj_seed)
    if key not in proj_matrix_cache:
        proj_matrix_cache[key] = _orthonormal_projection(in_dim, cfg.out_dim, cfg.proj_seed)
    W = proj_matrix_cache[key]  # (out_dim, in_dim)
    z = W @ features.astype(np.float32)  # (out_dim,)
    # Optional small bias from a fixed seed to de-correlate dead rows
    rng = np.random.default_rng(cfg.proj_seed + 7)
    b = rng.normal(scale=0.01, size=(cfg.out_dim,)).astype(np.float32)
    z = z + b
    return np.tanh(z)  # clip to [-1,1]


def _embed_torch(features: np.ndarray, cfg: EncoderConfig) -> np.ndarray:
    """
    Torch path: load weights if provided; otherwise initialize randomly (deterministic) and use that.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not installed; cannot use torch encoder.")

    x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)  # [1, D]
    model = _TorchBTCEncoder(in_dim=features.shape[-1], out_dim=cfg.out_dim, dropout=cfg.dropout)

    if cfg.weights_path and os.path.exists(cfg.weights_path):
        state = torch.load(cfg.weights_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        # deterministic init for reproducibility when no weights file
        torch.manual_seed(cfg.proj_seed)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, +bound)

    model.eval()
    with torch.no_grad():
        z = model(x).squeeze(0).cpu().numpy()
    # Already tanh-bounded in the model, but clip again defensively:
    return np.clip(z, -1.0, 1.0)


def build_btc_embedding_at(
    ts,
    *,
    caches: Optional[FeatureCaches] = None,
    collector_cfg: CollectorConfig = CollectorConfig(),
    block_cfg: BTCBlocksConfig = BTCBlocksConfig(),
    encoder_cfg: EncoderConfig = EncoderConfig(),
    use_torch_if_available: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    One-shot: collect inputs → build feature vector → map to 100-dim embedding in [-1,1].
    Returns:
      embedding: np.ndarray shape (100,)
      feat_names: list[str]  (names of the concatenated features used as encoder input)
    """
    # 1) collect inputs up to ts (no block logic here)
    caches = caches or FeatureCaches()
    inputs = collect_btc_inputs_at(ts, caches=caches, cfg=collector_cfg, prefetch_ohlcv=True)

    # 2) build feature vector at ts
    feat_vals, feat_names, _ = build_btc_features_from_inputs(inputs)
    x = np.asarray(feat_vals, dtype=np.float32)  # already bounded [-1,1] by your feature modules

    # 3) encode → 100 dims in [-1,1]
    proj_cache: dict = {}
    if use_torch_if_available and _HAS_TORCH:
        z = _embed_torch(x, encoder_cfg)
    else:
        z = _embed_numpy(x, encoder_cfg, proj_cache)

    return z, feat_names


# -------------------------------- __main__ demo -------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build 100-dim BTC embedding at a timestamp.")
    parser.add_argument("--ts", default="now", help="UTC ISO time or 'now'")
    parser.add_argument("--weights", default=None, help="Path to torch .pth weights (optional)")
    parser.add_argument("--no-torch", action="store_true", help="Force NumPy projection even if torch is installed")
    parser.add_argument("--print", action="store_true", help="Print embedding values")
    args = parser.parse_args()

    ts = pd.Timestamp.utcnow().tz_localize("UTC").floor("T") if args.ts == "now" else pd.to_datetime(args.ts, utc=True).floor("T")

    enc_cfg = EncoderConfig(weights_path=args.weights)
    emb, names = build_btc_embedding_at(
        ts,
        encoder_cfg=enc_cfg,
        use_torch_if_available=not args.no_torch,
    )

    print(f"BTC embedding @ {ts.isoformat()} — shape: {emb.shape}, range: [{emb.min():.3f}, {emb.max():.3f}]")
    if args.print:
        for i, v in enumerate(emb):
            print(f"z[{i:03d}] = {v:+.6f}")
    print("OK ✓")
