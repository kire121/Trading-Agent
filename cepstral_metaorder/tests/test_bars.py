import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cepstral_metaorder import bars


def _utc(y, m, d, h, mi):
    return pd.Timestamp(dt.datetime(y, m, d, h, mi), tz="UTC")


def test_session_minute_grid_is_360_minutes_0945_to_1544():
    grid = bars.session_minute_grid()
    assert len(grid) == 360
    assert grid["minute_of_day"].iloc[0] == 9 * 60 + 45
    assert grid["minute_of_day"].iloc[-1] == 15 * 60 + 44


def test_to_rth_grid_trims_extended_hours_and_buckets_by_session_date():
    # 2024-01-03 is a normal winter trading day, EST = UTC-5, so local ET
    # clock time = UTC - 5h. RTH 09:30-16:00 ET == 14:30-21:00 UTC. Trimmed
    # window 09:45-15:45 ET == 14:45-20:45 UTC. dense['minute_of_day'] is in
    # LOCAL (ET) minutes-since-midnight, e.g. 14:50 UTC -> 09:50 ET -> 590.
    rows = [
        (_utc(2024, 1, 3, 12, 0), 100.0, 999),   # 07:00 ET premarket -> must be dropped
        (_utc(2024, 1, 3, 14, 50), 101.0, 50),   # 09:50 ET -> inside trimmed window
        (_utc(2024, 1, 3, 15, 0), 102.0, 60),    # 10:00 ET -> inside trimmed window
        (_utc(2024, 1, 3, 20, 40), 103.0, 70),   # 15:40 ET -> inside trimmed window (near close)
        (_utc(2024, 1, 3, 22, 0), 104.0, 999),   # 17:00 ET postmarket -> must be dropped
    ]
    raw = pd.DataFrame(rows, columns=["datetime", "close", "volume"])

    dense = bars.to_rth_grid(raw)

    assert set(dense["session_date"].unique()) == {dt.date(2024, 1, 3)}
    assert len(dense) == 360  # full dense grid for the one session

    # minutes with no trade get volume 0, not NaN/dropped
    assert (dense["volume"] == 0).sum() == 360 - 3
    assert dense.loc[dense["minute_of_day"] == 9 * 60 + 50, "close"].iloc[0] == 101.0

    # forward-filled close on a no-trade minute right after a trade
    row_951 = dense.loc[dense["minute_of_day"] == 9 * 60 + 51]
    assert row_951["volume"].iloc[0] == 0
    assert row_951["close"].iloc[0] == 101.0


def test_sessions_with_full_coverage_drops_thin_names():
    grid = bars.session_minute_grid()
    n = len(grid)
    thin = pd.DataFrame({
        "session_date": [dt.date(2024, 1, 3)] * n,
        "minute_of_day": grid["minute_of_day"],
        "volume": [1.0] + [0.0] * (n - 1),  # only 1 trade all day
    })
    liquid = pd.DataFrame({
        "session_date": [dt.date(2024, 1, 4)] * n,
        "minute_of_day": grid["minute_of_day"],
        "volume": np.full(n, 10.0),
    })
    dense = pd.concat([thin, liquid], ignore_index=True)
    kept = bars.sessions_with_full_coverage(dense, min_nonzero_frac=0.5)
    assert kept == [dt.date(2024, 1, 4)]
