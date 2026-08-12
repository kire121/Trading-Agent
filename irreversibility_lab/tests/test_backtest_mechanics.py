"""Sanity checks for causality and cost mechanics of the backtest engine --
independent of whether the irreversibility signal itself has any edge."""

import numpy as np
import pandas as pd
import pytest

from irreversibility_lab import backtest, signal


def _toy_calendar(n=40):
    return pd.bdate_range("2021-01-04", periods=n)  # starts on a Monday


def test_daily_weights_start_after_effective_date_not_before():
    idx = _toy_calendar()
    anchor = idx[4]  # a Friday
    weekly_w = pd.DataFrame({"X": [1.0]}, index=[anchor])
    daily_w = backtest.daily_weights_from_weekly(weekly_w, idx)
    eff = signal.next_trading_day(idx, anchor)
    assert (daily_w.loc[:anchor, "X"] == 0.0).all(), "weight must not be active on/before the decision date"
    assert daily_w.loc[eff, "X"] == 1.0
    assert (daily_w.loc[eff:, "X"] == 1.0).all(), "weight should hold constant until the next effective date"


def test_no_lookahead_in_backtest_pnl():
    """The return realized on the decision (Friday) date itself must be
    unaffected by the weight decided that same day."""
    idx = _toy_calendar()
    ret = pd.DataFrame({"X": np.zeros(len(idx))}, index=idx)
    ret.loc[idx[4], "X"] = 0.05  # a large return exactly on the Friday decision date
    weekly_w = pd.DataFrame({"X": [1.0]}, index=[idx[4]])
    px = pd.DataFrame({"X": 100 * (1 + ret["X"]).cumprod()}, index=idx)
    bt = backtest.run_backtest(px, ret, weekly_w)
    assert bt["portfolio_return"].loc[idx[4]] == pytest.approx(0.0), (
        "weight decided using Friday's own close must not earn Friday's own return (lookahead)"
    )


def test_turnover_and_cost_only_on_effective_dates():
    idx = _toy_calendar()
    ret = pd.DataFrame({"X": np.zeros(len(idx))}, index=idx)
    weekly_w = pd.DataFrame({"X": [1.0, -1.0]}, index=[idx[4], idx[11]])
    px = pd.DataFrame({"X": np.full(len(idx), 100.0)}, index=idx)
    bt = backtest.run_backtest(px, ret, weekly_w)
    nonzero_turnover_dates = bt["turnover"][bt["turnover"] > 1e-12].index
    eff1 = signal.next_trading_day(idx, idx[4])
    eff2 = signal.next_trading_day(idx, idx[11])
    assert set(nonzero_turnover_dates) == {eff1, eff2}
    assert bt["cost"].loc[eff1] > 0
    assert bt["cost"].loc[eff2] > 0


def test_gross_cap_enforced():
    from irreversibility_lab import strategy
    direction = pd.DataFrame({f"A{i}": [1.0] for i in range(12)})
    vol_like_returns = pd.DataFrame(
        {f"A{i}": np.random.default_rng(i).normal(0, 0.001, 70) for i in range(12)},
        index=pd.bdate_range("2020-01-01", periods=70),
    )
    anchors = vol_like_returns.index[-1:]
    direction.index = anchors
    w = strategy.build_weekly_weights(direction, vol_like_returns, anchors,
                                       per_inst_cap=0.5, gross_cap=1.5)
    assert w.abs().sum(axis=1).iloc[0] <= 1.5 + 1e-9


def test_hysteresis_regime_holds_between_thresholds():
    from irreversibility_lab import signal as sig
    z = pd.Series([0.6, 0.3, -0.1, -0.6, 0.1, 0.9], index=pd.RangeIndex(6))
    regime = sig.classify_regime_series(z, upper=0.5, lower=-0.5)
    assert list(regime) == ["TREND", "TREND", "TREND", "MEANREV", "MEANREV", "TREND"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
