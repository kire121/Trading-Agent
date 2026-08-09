import numpy as np
import pandas as pd

from oglegrinden.signal import (
    weekly_fridays,
    smooth_median4,
    expanding_percentile,
    hysteresis_gate,
    compute_raw_signal_series,
)
from oglegrinden.tests.test_backtest import _make_synthetic_panel


def test_weekly_fridays_picks_last_trading_day_per_week():
    # A week with a Friday holiday: Mon-Thu trade, Friday closed.
    dates = pd.to_datetime(
        [
            "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08",  # full week
            "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14",  # Friday 1/15 holiday, missing
        ]
    )
    fridays = weekly_fridays(dates)
    assert pd.Timestamp("2021-01-08") in fridays
    assert pd.Timestamp("2021-01-14") in fridays  # last trading day standing in for the holiday Friday
    assert len(fridays) == 2


def test_smooth_median4_is_trailing_and_causal():
    s = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0, 6.0])
    smoothed = smooth_median4(s)
    # 4th obs onward uses a trailing 4-window median
    assert smoothed.iloc[3] == np.median([1.0, 2.0, 3.0, 100.0])
    assert smoothed.iloc[4] == np.median([2.0, 3.0, 100.0, 5.0])
    # causal: later values must not affect earlier smoothed output
    s2 = s.copy()
    s2.iloc[5] = -999.0
    assert smooth_median4(s2).iloc[:5].equals(smoothed.iloc[:5])


def test_expanding_percentile_no_lookahead_and_warmup_nan():
    idx = pd.date_range("2000-01-01", periods=200, freq="W-FRI")
    values = np.arange(200, dtype=float)
    s = pd.Series(values, index=idx)
    # min_history_years=1, obs_per_year defaults to 52 -> warmup of 52 obs
    pct = expanding_percentile(s, min_history_years=1.0, obs_per_year=52)
    assert pct.iloc[:51].isna().all()
    assert not np.isnan(pct.iloc[51])
    # monotonically increasing series -> percentile of the newest (= max
    # so far) observation should always be 100
    assert (pct.dropna() == 100.0).all()

    # no-lookahead: truncating the series after some date must not change
    # percentiles computed strictly before that date
    cut = 150
    pct_full = expanding_percentile(s, min_history_years=1.0, obs_per_year=52)
    pct_truncated = expanding_percentile(s.iloc[:cut], min_history_years=1.0, obs_per_year=52)
    assert pct_full.iloc[:cut].equals(pct_truncated)


def test_hysteresis_gate_primary_holds_in_band_and_starts_off():
    # percentiles: starts below lower (off), rises through the band
    # (holds off), crosses upper (on), dips into band (holds on),
    # falls below lower (off again).
    pct = pd.Series([10, 35, 45, 55, 65, 80, 55, 45, 35, 20])
    gate = hysteresis_gate(pct, upper=60, lower=40, direction="primary")
    expected = [False, False, False, False, True, True, True, True, False, False]
    assert list(gate) == expected


def test_hysteresis_gate_mirror_is_flipped_trigger_not_flipped_state_trace():
    pct = pd.Series([80, 55, 45, 35, 20, 45, 55, 65, 80])
    gate = hysteresis_gate(pct, upper=60, lower=40, direction="mirror")
    # mirror: ON when pct<lower, OFF when pct>upper, hold in band
    expected = [False, False, False, True, True, True, True, False, False]
    assert list(gate) == expected


def test_hysteresis_gate_nan_holds_previous_state():
    pct = pd.Series([70.0, np.nan, np.nan, 20.0])
    gate = hysteresis_gate(pct, upper=60, lower=40, direction="primary")
    assert list(gate) == [True, True, True, False]


def test_compute_raw_signal_series_drops_frozen_ticker_instead_of_crashing():
    """A ticker whose price is frozen for a stretch >= corr_window has zero
    variance over that window, which would otherwise make
    correlation_distance raise. compute_raw_signal_series must exclude
    that ticker from the week's cloud rather than let one stale feed kill
    the whole signal computation (regression test for a real, if never
    fired on live data, bug found in adversarial review)."""
    panel = _make_synthetic_panel(n_tickers=20, n_days=300, seed=9)
    frozen_ticker = "T05"
    panel.close.loc[:, frozen_ticker] = panel.close[frozen_ticker].iloc[0]
    panel.adjclose.loc[:, frozen_ticker] = panel.adjclose[frozen_ticker].iloc[0]
    panel.log_returns[frozen_ticker] = 0.0

    raw = compute_raw_signal_series(panel, corr_window=60, min_universe_size=15)
    assert len(raw) > 0
    assert raw["L"].notna().all()
    assert raw["n_assets"].max() <= 19  # frozen ticker never counted
