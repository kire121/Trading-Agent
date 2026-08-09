import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fasflocken.universe import SyntheticUniverseProvider
from fasflocken.config import SignalParams, GICS_SECTORS
from fasflocken.backtest import (
    run_backtest,
    _week_windows,
    annualized_sharpe,
    annualized_return,
    annualized_vol,
    max_drawdown,
)

SECTORS = tuple(list(GICS_SECTORS)[:9])  # the 9 always-on sectors, fast test universe
FAST_PARAMS = SignalParams(
    band_low_days=5, band_high_days=20, analytic_window=45, z_lookback_weeks=26, n_legs=3, hysteresis_band=4,
    min_constituents=4,
)


@pytest.fixture(scope="module")
def small_provider():
    start, end = dt.date(2016, 1, 1), dt.date(2018, 12, 31)
    return SyntheticUniverseProvider(start=start, end=end, n_per_sector=8, seed=11, sectors=SECTORS), start, end


def test_week_windows_cover_calendar_without_overlap():
    daily = pd.bdate_range("2020-01-01", periods=60)
    fridays = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
    windows = _week_windows(fridays, daily)
    assert len(windows) == len(fridays)
    all_days = pd.DatetimeIndex(np.concatenate([w.values for w in windows if len(w)]))
    assert all_days.is_unique

    for i, w in enumerate(windows):
        lower = fridays[i]
        upper = fridays[i + 1] if i + 1 < len(fridays) else daily.max()
        assert (w > lower).all() and (w <= upper).all()


def test_run_backtest_shapes_and_bounds(small_provider):
    provider, start, end = small_provider
    result = run_backtest(provider, start, end, FAST_PARAMS, sectors=SECTORS)

    assert len(result.weekly_returns) == len(result.weights) == len(result.turnover) == len(result.k_scale)
    assert (result.turnover >= -1e-9).all()
    assert (result.k_scale >= -1e-9).all()
    assert (result.weights.abs().sum(axis=1) <= 2.0 + 1e-6).all()
    # net return should never exceed gross (costs are non-negative)
    assert (result.weekly_returns <= result.weekly_gross_returns + 1e-12).all()


def test_run_backtest_gates_unlisted_etfs(small_provider):
    provider, _, _ = small_provider
    start, end = dt.date(2010, 1, 1), dt.date(2014, 12, 31)  # entirely before XLRE/XLC inception
    provider_early = SyntheticUniverseProvider(start=start, end=end, n_per_sector=6, seed=3, sectors=GICS_SECTORS)
    result = run_backtest(provider_early, start, end, FAST_PARAMS, sectors=GICS_SECTORS)
    assert "XLRE" not in result.weights.columns or (result.weights["XLRE"] == 0).all()
    assert "XLC" not in result.weights.columns or (result.weights["XLC"] == 0).all()


def test_run_backtest_no_trades_before_burnin():
    start, end = dt.date(2016, 1, 1), dt.date(2016, 12, 31)  # far short of a 26-week burn-in twice over? use long lookback
    provider = SyntheticUniverseProvider(start=start, end=end, n_per_sector=6, seed=5, sectors=SECTORS)
    params = SignalParams(z_lookback_weeks=104, analytic_window=45, n_legs=3, hysteresis_band=4)  # needs 2yr, sample is 1yr
    result = run_backtest(provider, start, end, params, sectors=SECTORS)
    assert (result.turnover == 0).all()
    assert (result.weekly_returns == 0).all()


def test_annualized_stats_basic_properties():
    idx = pd.date_range("2020-01-03", periods=52, freq="W-FRI")
    r = pd.Series(0.001, index=idx)  # constant positive return, zero vol
    assert annualized_return(r) == pytest.approx(0.001 * 52)
    assert np.isnan(annualized_sharpe(r))  # zero std -> Sharpe undefined
    assert annualized_vol(r) == 0.0
    assert max_drawdown(r) == 0.0

    r2 = pd.Series([0.01, -0.02, 0.01, -0.02] * 13, index=idx)
    assert max_drawdown(r2) < 0


def test_z_override_skips_signal_recompute(small_provider):
    provider, start, end = small_provider
    dates = pd.date_range(start, end, freq="W-FRI")
    fake_z = pd.DataFrame(0.0, index=dates, columns=list(SECTORS))
    result = run_backtest(provider, start, end, FAST_PARAMS, sectors=SECTORS, z_override=fake_z)
    assert result.sector_signals == {}  # confirms the expensive path was skipped
    # all-zero Z -> ranking is a tie; select_legs must still return something usable
    assert len(result.weekly_returns) == len(dates)
