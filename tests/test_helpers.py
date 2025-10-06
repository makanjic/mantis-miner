import pandas as pd
import numpy as np
from features_helpers import (
    utc_floor_minute, _prep_series, _resample_1m_ffill,
    _z_last, _delta_last, _ret_last, _bp_change_last, _clip01
)

def test_utc_floor_minute():
    t = utc_floor_minute("2025-10-06T12:34:56Z")
    assert str(t) == "2025-10-06 12:34:00+00:00"

def test_series_hygiene_and_resample():
    idx = pd.to_datetime(["2025-10-06T12:34:00Z","2025-10-06T12:36:00Z"])
    s = pd.Series([1.0, 3.0], index=idx)
    s2 = _resample_1m_ffill(_prep_series(s), "2025-10-06T12:36:00Z")
    assert s2.index.freq is None  # range index, but minute spaced
    assert s2.iloc[-1] == 3.0 and s2.iloc[-3] == 1.0  # ffilled

def test_small_transforms():
    idx = pd.date_range("2025-10-06T12:00:00Z", periods=61, freq="1min")
    s = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    assert -1 <= _z_last(s, 30) <= 1
    assert -1 <= _delta_last(s, 10) <= 1
    assert -1 <= _ret_last(s, 10) <= 1
    assert -1 <= _bp_change_last(s, 10) <= 1
    assert _clip01(5) == 1 and _clip01(-5) == -1
