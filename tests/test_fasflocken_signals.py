import numpy as np
import pandas as pd
import pytest

from fasflocken.signals import (
    causal_bandpass,
    rolling_analytic_phase,
    kuramoto_order_parameter,
    resample_weekly_last,
    rolling_zscore,
)


def test_causal_bandpass_is_future_blind():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y_short = causal_bandpass(x, 5, 20)
    x_extended = np.concatenate([x, rng.normal(size=50)])
    y_long = causal_bandpass(x_extended, 5, 20)
    assert np.allclose(y_short, y_long[: len(x)], equal_nan=True)


def test_causal_bandpass_rejects_bad_band():
    x = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError):
        causal_bandpass(x, 20, 5)  # high must exceed low


def test_causal_bandpass_edges_stay_nan_short_gaps_bridged():
    x = np.full(200, np.nan)
    x[50:150] = np.random.default_rng(1).normal(size=100)
    x[100] = np.nan  # a single-day gap inside the valid range
    y = causal_bandpass(x, 5, 20)
    assert np.isnan(y[:50]).all()
    assert np.isnan(y[150:]).all()
    assert not np.isnan(y[51:150]).any()  # bridged, not poisoned


def test_causal_bandpass_long_gap_poisons_rest_of_segment():
    x = np.full(400, np.nan)
    x[50:150] = np.random.default_rng(2).normal(size=100)
    x[190:230] = np.nan  # 40-day gap, well beyond default max_gap=5
    x[230:350] = np.random.default_rng(3).normal(size=120)
    y = causal_bandpass(x, 5, 20, max_gap=5)

    # bridged for max_gap samples past the last valid point (150..154), then
    # the unbridged center of the gap poisons sosfilt's IIR state for good --
    # including the later, otherwise-valid 230:350 stretch of the same segment.
    assert not np.isnan(y[50:155]).any()
    assert np.isnan(y[160:350]).all()


def test_rolling_analytic_phase_matches_direct_hilbert_on_window():
    from scipy.signal import hilbert

    rng = np.random.default_rng(3)
    x = rng.normal(size=150)
    window = 90
    phase = rolling_analytic_phase(x, window=window)
    assert np.isnan(phase[: window - 1]).all()

    t = window + 10
    expected = np.angle(hilbert(x[t - window + 1 : t + 1]))[-1]
    assert np.isclose(phase[t], expected)


def test_kuramoto_amplitude_invariance():
    t = np.arange(500)
    base = np.sin(2 * np.pi * t / 10)
    amps = np.array([0.3, 1.0, 2.5, 5.0, 0.05])
    mat = np.column_stack([a * base for a in amps])
    bp = causal_bandpass(mat, 5, 20)
    phase = rolling_analytic_phase(bp, window=90)
    R = kuramoto_order_parameter(phase, min_constituents=3)
    assert np.all(R[-10:] > 0.999)


def test_kuramoto_random_phases_near_zero():
    rng = np.random.default_rng(1)
    N, T = 200, 50
    phases = rng.uniform(-np.pi, np.pi, size=(T, N))
    R = kuramoto_order_parameter(phases, min_constituents=5)
    assert np.nanmean(R) < 3 / np.sqrt(N)  # generous bound around the ~1/sqrt(N) expectation


def test_kuramoto_destructive_interference():
    phi = np.zeros((5, 100))
    phi[:, 50:] = np.pi
    R = kuramoto_order_parameter(phi, min_constituents=3)
    assert np.allclose(R, 0.0, atol=1e-9)


def test_kuramoto_respects_min_constituents():
    phi = np.zeros((3, 10))
    phi[:, 4:] = np.nan  # only 4 valid columns
    R = kuramoto_order_parameter(phi, min_constituents=5)
    assert np.isnan(R).all()


def test_kuramoto_bounded_0_1():
    rng = np.random.default_rng(5)
    phi = rng.uniform(-np.pi, np.pi, size=(50, 30))
    R = kuramoto_order_parameter(phi, min_constituents=5)
    assert np.all((R >= 0) & (R <= 1 + 1e-9))


def test_resample_weekly_last_labels_on_friday():
    idx = pd.bdate_range("2020-01-01", periods=40)
    s = pd.Series(np.arange(40), index=idx)
    w = resample_weekly_last(s)
    assert (w.index.weekday == 4).all()
    # the value on each Friday should be that week's last business-day value
    assert w.iloc[0] == s.loc[s.index <= w.index[0]].iloc[-1]


def test_resample_weekly_last_holiday_robust():
    idx = pd.bdate_range("2020-01-01", periods=10)  # Wed 1/1 .. Tue 1/14
    first_friday = pd.Timestamp("2020-01-03")
    assert first_friday in idx and first_friday.weekday() == 4
    s = pd.Series(np.arange(10), index=idx)
    s_no_friday = s.drop(first_friday)
    w = resample_weekly_last(s_no_friday)
    # first week's bin now ends on Thursday 1/2 (the last remaining day <= that Friday)
    expected = s_no_friday.loc[:first_friday].iloc[-1]
    assert w.iloc[0] == expected


def test_rolling_zscore_burn_in_and_correctness():
    idx = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    vals = np.arange(30, dtype=float)
    s = pd.Series(vals, index=idx)
    z = rolling_zscore(s, lookback_weeks=10, min_periods=10)
    assert z.iloc[:9].isna().all()
    window = vals[0:10]
    expected = (vals[9] - window.mean()) / window.std(ddof=1)
    assert np.isclose(z.iloc[9], expected)


def test_rolling_zscore_zero_std_is_nan():
    idx = pd.date_range("2020-01-03", periods=15, freq="W-FRI")
    s = pd.Series(np.full(15, 5.0), index=idx)
    z = rolling_zscore(s, lookback_weeks=5, min_periods=5)
    assert z.iloc[4:].isna().all()
