# time_features.py
# Utilities for turning UTC timestamps into stable, bounded time features in [-1, 1].
# Works for single timestamps or arrays of timestamps.
from __future__ import annotations

import math, calendar
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable, List, Tuple, Sequence, Optional

try:
    # stdlib from Python 3.9+: robust timezone math (optional for session features)
    from zoneinfo import ZoneInfo
    _HAS_ZONEINFO = True
except Exception:
    _HAS_ZONEINFO = False

# ------------------------- Core scalar primitive ------------------------- #

def time_scalar(
    t_unix: float,
    period_s: float,
    *,
    kind: str = "cos",      # "cos", "sin", "saw", "tri", "vm" (von Mises bump)
    phase_s: float = 0.0,   # shift in seconds (align cycle peak)
    k: int = 1,             # harmonic multiplier (1=base, 2=second harmonic, ...)
    kappa: float = 4.0      # sharpness for "vm" (higher = narrower)
) -> float:
    """
    Generic, bounded time feature: timestamp -> scalar in [-1, 1].

    - period_s: cycle length in seconds (e.g., 86400 day, 604800 week, 28800 8h)
    - kind:
        * "cos"/"sin": smooth Fourier terms
        * "saw"      : sawtooth wave (bounded)
        * "tri"      : triangle wave (bounded)
        * "vm"       : von Mises bump (single peak per cycle), normalized to [-1,1]
    - phase_s: shifts the cycle (e.g., align to session opens)
    - k     : harmonic multiplier
    """
    if period_s <= 0:
        raise ValueError("period_s must be > 0")

    # Reduce to [0,1) fractional phase, then to angle
    frac = ((t_unix + phase_s) % period_s) / period_s
    angle = 2 * math.pi * (k * frac)

    if kind == "cos":
        return math.cos(angle)
    if kind == "sin":
        return math.sin(angle)
    if kind == "saw":
        f = ((k * frac) % 1.0)        # [0,1)
        return 2.0 * f - 1.0          # [-1,1)
    if kind == "tri":
        f = ((k * frac) % 1.0)
        return 1.0 - 4.0 * abs(f - 0.5)  # [-1,1], peaks at mid-cycle
    if kind == "vm":
        # von Mises: exp(kappa * cos(angle)) ∈ [e^-kappa, e^kappa] → normalize to [-1,1]
        ex = math.exp(kappa * math.cos(angle))
        ex_min = math.exp(-kappa)
        ex_max = math.exp(kappa)
        u01 = (ex - ex_min) / (ex_max - ex_min + 1e-12)
        return 2.0 * u01 - 1.0
    raise ValueError(f"Unknown kind={kind}")

# ------------------------- Spec-driven builder -------------------------- #

@dataclass(frozen=True)
class TimeSpec:
    name: str
    period_s: float
    kind: str = "cos"
    phase_s: float = 0.0
    k: int = 1
    kappa: float = 4.0

def time_vector(t_unix: float, specs: Sequence[TimeSpec]) -> Tuple[List[float], List[str]]:
    vals: List[float] = []
    names: List[str] = []
    for s in specs:
        v = time_scalar(
            t_unix, s.period_s,
            kind=s.kind, phase_s=s.phase_s, k=s.k, kappa=s.kappa
        )
        # Safety clamp to [-1,1]
        v = max(-1.0, min(1.0, v))
        vals.append(v)
        names.append(s.name)
    return vals, names

def time_matrix(t_unix_list: Sequence[float], specs: Sequence[TimeSpec]) -> Tuple[List[List[float]], List[str]]:
    """
    Vectorized: many timestamps -> matrix [N, len(specs)].
    Returns (matrix, names) where each row corresponds to t_unix_list[i].
    """
    mat: List[List[float]] = []
    names: List[str] = [s.name for s in specs]
    for t in t_unix_list:
        row, _ = time_vector(t, specs)
        mat.append(row)
    return mat, names

# ---------------------- Calendar/seasonality block ---------------------- #

def _frac_day(dt: datetime) -> float:
    # minute-of-day fraction in [0,1)
    return (dt.hour * 60 + dt.minute) / 1440.0

def weekday_sin_cos(t_unix: float) -> Tuple[float, float]:
    """
    Weekday with smooth intraday progression (Mon=0..Sun=6), each in [-1,1].
    """
    dt = datetime.fromtimestamp(t_unix, tz=timezone.utc)
    phase = dt.weekday() + _frac_day(dt)   # 0..~7
    ang = 2 * math.pi * phase / 7.0
    return math.sin(ang), math.cos(ang)

def dom_sin_cos(t_unix: float) -> Tuple[float, float]:
    """
    Day-of-month as a smooth cycle over the actual month length (28..31), in [-1,1].
    """
    dt = datetime.fromtimestamp(t_unix, tz=timezone.utc)
    dim = calendar.monthrange(dt.year, dt.month)[1]
    frac = ((dt.day - 1) + _frac_day(dt)) / float(dim)  # [0,1)
    ang = 2 * math.pi * frac
    return math.sin(ang), math.cos(ang)

def eom_proximity(t_unix: float, window_hours: int = 36) -> float:
    """
    Cosine "proximity" to End-Of-Month (EOM): +1 near EOM, ~-1 far away.
    window_hours controls bump width (try 24–48). Output ∈ [-1, 1].
    """
    dt = datetime.fromtimestamp(t_unix, tz=timezone.utc)
    dim = calendar.monthrange(dt.year, dt.month)[1]
    eom = datetime(dt.year, dt.month, dim, 23, 59, 59, tzinfo=timezone.utc)

    if dt > eom:
        # Past EOM: use next month's EOM
        y2, m2 = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
        dim2 = calendar.monthrange(y2, m2)[1]
        eom = datetime(y2, m2, dim2, 23, 59, 59, tzinfo=timezone.utc)

    hours_to_eom = (eom - dt).total_seconds() / 3600.0
    # Map remaining hours → [0,1], clamp beyond window
    x = max(0.0, min(1.0, 1.0 - hours_to_eom / float(window_hours)))
    # Cosine window: -1 far, +1 near EOM
    return max(-1.0, min(1.0, math.cos(math.pi * (1.0 - x))))

def moy_sin_cos(t_unix: float) -> Tuple[float, float]:
    """
    Month-of-year cyclical encoding (smooth within month), in [-1,1].
    """
    dt = datetime.fromtimestamp(t_unix, tz=timezone.utc)
    dim = calendar.monthrange(dt.year, dt.month)[1]
    frac_year = (dt.month - 1 + (dt.day - 1 + _frac_day(dt)) / dim) / 12.0
    ang = 2 * math.pi * frac_year
    return math.sin(ang), math.cos(ang)

def calendar_time_vector(t_unix: float, window_hours: int = 36) -> Tuple[List[float], List[str]]:
    """
    Fixed-order calendar vector (values ∈ [-1,1]) + names:
      [dow_sin, dow_cos, dom_sin, dom_cos, eom_prox, moy_sin, moy_cos]
    """
    dow_s, dow_c = weekday_sin_cos(t_unix)
    dom_s, dom_c = dom_sin_cos(t_unix)
    eom = eom_proximity(t_unix, window_hours=window_hours)
    moy_s, moy_c = moy_sin_cos(t_unix)
    vals = [dow_s, dow_c, dom_s, dom_c, eom, moy_s, moy_c]
    names = ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "eom_prox", "moy_sin", "moy_cos"]
    return vals, names

# ------------------- Optional: sessions & funding cycles ----------------- #

def session_open_cycle(
    t_unix: float,
    *,
    tz_name: str,
    open_hour: int,
    open_minute: int = 0
) -> float:
    """
    Cosine proximity (∈ [-1,1]) to the next daily *local* session open in tz_name.
    Handles DST properly if zoneinfo is available. If not, falls back to a UTC cosine.
    """
    if _HAS_ZONEINFO:
        tz = ZoneInfo(tz_name)
        now_local = datetime.fromtimestamp(t_unix, tz=timezone.utc).astimezone(tz)
        event = now_local.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
        if event < now_local:
            event += timedelta(days=1)
        minutes_to_open = (event - now_local).total_seconds() / 60.0
        # Map minutes to a daily cosine cycle (peak at open)
        # Convert to seconds offset from "now" to next open
        # Fraction of day until open:
        frac = (minutes_to_open / (24*60.0)) % 1.0
        ang = 2 * math.pi * (1.0 - frac)
        return math.cos(ang)
    else:
        # Approximate with fixed UTC phase: you can set phase_s manually via TimeSpec
        return time_scalar(t_unix, period_s=24*60*60, kind="cos")

def funding_cycle_8h(t_unix: float, kappa: float = 6.0) -> float:
    """Convenience bump for typical 8h perp funding rhythm ([-1,1])."""
    return time_scalar(t_unix, period_s=8*60*60, kind="vm", kappa=kappa)

# ------------------------ Defaults & convenience ------------------------ #

DAY = 24 * 60 * 60
WEEK = 7 * DAY
H8  = 8 * 60 * 60
H6  = 6 * 60 * 60
H4  = 4 * 60 * 60

DEFAULT_SPECS: List[TimeSpec] = [
    # Daily harmonics
    TimeSpec("tod_cos_1", DAY, "cos", k=1),
    TimeSpec("tod_sin_1", DAY, "sin", k=1),
    TimeSpec("tod_cos_2", DAY, "cos", k=2),
    TimeSpec("tod_sin_2", DAY, "sin", k=2),
    TimeSpec("tod_cos_3", DAY, "cos", k=3),
    TimeSpec("tod_sin_3", DAY, "sin", k=3),

    # Weekly cycle
    # TimeSpec("dow_cos", WEEK, "cos"),
    # TimeSpec("dow_sin", WEEK, "sin"),

    # Funding rhythm (8h) + shorter cycles if desired
    TimeSpec("fund8h_cos", H8, "cos"),
    TimeSpec("fund8h_sin", H8, "sin"),
    TimeSpec("fund8h_bump", H8, "vm", kappa=6.0),
    # You can add H6/H4 if you like:
    # TimeSpec("cyc_6h_cos", H6, "cos"), TimeSpec("cyc_6h_sin", H6, "sin"),
    # TimeSpec("cyc_4h_cos", H4, "cos"), TimeSpec("cyc_4h_sin", H4, "sin"),
]

def make_default_time_features(
    t_unix: float,
    *,
    include_calendar: bool = True,
    include_defaults: bool = True,
    extra_specs: Optional[Sequence[TimeSpec]] = None,
    eom_window_hours: int = 36
) -> Tuple[List[float], List[str]]:
    """
    Build a full time feature vector (values in [-1,1]) + names.
    - include_calendar: add weekday, DOM, EOM proximity, MOY (your request)
    - include_defaults: add DEFAULT_SPECS (daily/weekly/funding cycles)
    - extra_specs: optional additional TimeSpec entries
    """
    vals: List[float] = []
    names: List[str] = []

    if include_defaults:
        v, n = time_vector(t_unix, DEFAULT_SPECS)
        vals.extend(v); names.extend(n)

    if include_calendar:
        v, n = calendar_time_vector(t_unix, window_hours=eom_window_hours)
        vals.extend(v); names.extend(n)

    if extra_specs:
        v, n = time_vector(t_unix, list(extra_specs))
        vals.extend(v); names.extend(n)

    return vals, names

__all__ = [
    "time_scalar", "TimeSpec", "time_vector", "time_matrix",
    "weekday_sin_cos", "dom_sin_cos", "eom_proximity", "moy_sin_cos",
    "calendar_time_vector", "session_open_cycle", "funding_cycle_8h",
    "DEFAULT_SPECS", "make_default_time_features"
]

if __name__ == "__main__":
    import time
    import argparse
    from pprint import pprint

    # CLI
    parser = argparse.ArgumentParser(
        description="Quick test runner for time_features.py"
    )
    parser.add_argument(
        "--ts",
        type=float,
        default=None,
        help="UTC Unix timestamp (seconds). Default: now()",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=5,
        help="Number of rows for matrix demo (default: 5).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=60,
        help="Seconds between rows for matrix demo (default: 60s).",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Exclude DEFAULT_SPECS (daily/weekly/funding cycles).",
    )
    parser.add_argument(
        "--no-calendar",
        action="store_true",
        help="Exclude calendar features (weekday, DOM, EOM, MOY).",
    )
    parser.add_argument(
        "--eom-window-hours",
        type=int,
        default=36,
        help="Window (hours) for EOM proximity bump (default: 36).",
    )
    args = parser.parse_args()

    # Choose timestamp
    t0 = args.ts or time.time()
    dt0 = datetime.fromtimestamp(t0, tz=timezone.utc)

    print("\n=== Single-timestamp features ===")
    print(f"UTC time: {dt0.isoformat()}  (unix={t0:.3f})")

    vals, names = make_default_time_features(
        t0,
        include_calendar=not args.no_calendar,
        include_defaults=not args.no_defaults,
        eom_window_hours=args.eom_window_hours,
    )
    # Pretty print name -> value
    for n, v in zip(names, vals):
        print(f"{n:>16}: {v:+.6f}")

    # Range check
    vmin, vmax = min(vals), max(vals)
    print(f"\nRange check: min={vmin:+.6f}, max={vmax:+.6f}  (should be within [-1, 1])")

    # Custom session bumps (example specs)
    print("\n=== Custom session bumps (London 08:00 UTC, US 14:30 UTC) ===")
    ldn_08utc = TimeSpec("ldn_open_bump", DAY, kind="vm", phase_s=-(8 * 3600), kappa=8.0)
    us_1430utc = TimeSpec("us_open_bump", DAY, kind="vm", phase_s=-(14 * 3600 + 30 * 60), kappa=8.0)
    extra_vals, extra_names = time_vector(t0, [ldn_08utc, us_1430utc])
    for n, v in zip(extra_names, extra_vals):
        print(f"{n:>16}: {v:+.6f}")

    # Matrix demo over a small timeline
    print(f"\n=== Matrix demo (batch={args.batch}, step={args.step}s) ===")
    ts_list = [t0 - i * args.step for i in range(args.batch)]
    ts_list.reverse()  # ascending time
    mat, mnames = time_matrix(ts_list, [ldn_08utc, us_1430utc])
    print("names:", mnames)
    for i, (t, row) in enumerate(zip(ts_list, mat)):
        dti = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S")
        row_str = " ".join(f"{x:+.4f}" for x in row)
        print(f"{i:02d} {dti}  {row_str}")

    print("\nOK ✓  (All values are bounded; tweak specs or flags to explore)")
