from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vridmomentet.backtest import (
    _last_valid_price_in_range,
    benchmark_weekly_returns,
    next_trading_day,
    run_backtest,
    weekly_decision_dates,
)
from vridmomentet.config import CostModel, PortfolioParams
from vridmomentet.data import Panel
from vridmomentet.signal import SignalResult
from vridmomentet.universe import MembershipInterval, PointInTimeMembership


def _make_panel(dates: pd.DatetimeIndex, prices: dict[str, np.ndarray], volumes: dict[str, np.ndarray] | None = None) -> Panel:
    close = pd.DataFrame(prices, index=dates)
    adj_close = close.copy()
    adj_open = close.copy()
    if volumes is None:
        volumes = {k: np.full(len(dates), 1_000_000.0) for k in prices}
    volume = pd.DataFrame(volumes, index=dates)
    # Every synthetic name is "in the universe" for the whole test window --
    # eligible_on() would otherwise filter everyone out against an empty
    # membership, silently turning every downstream test vacuous.
    start, end = dates.min().date(), dates.max().date() + dt.timedelta(days=1)  # end is exclusive
    membership = PointInTimeMembership(
        [MembershipInterval(ticker=t, source="sp500", start=start, end=end, point_in_time=True) for t in prices]
    )
    return Panel(close=close, adj_close=adj_close, adj_open=adj_open, volume=volume, membership=membership)


class TestWeeklyDecisionDates:
    def test_picks_last_trading_day_per_iso_week(self):
        # Mon 2024-01-01 .. Fri 2024-01-19, 3 full weeks, business days only.
        dates = pd.bdate_range("2024-01-01", "2024-01-19")
        decisions = weekly_decision_dates(dates)
        assert list(decisions.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-12", "2024-01-19"]

    def test_holiday_shifted_friday_uses_thursday(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-05").drop(pd.Timestamp("2024-01-05"))  # Friday missing
        decisions = weekly_decision_dates(dates)
        assert decisions[-1] == pd.Timestamp("2024-01-04")


class TestNextTradingDay:
    def test_returns_first_day_strictly_after(self):
        cal = pd.bdate_range("2024-01-01", "2024-01-19")
        nxt = next_trading_day(cal, pd.Timestamp("2024-01-05"))
        assert nxt == pd.Timestamp("2024-01-08")

    def test_returns_none_past_end_of_calendar(self):
        cal = pd.bdate_range("2024-01-01", "2024-01-19")
        assert next_trading_day(cal, pd.Timestamp("2024-01-19")) is None


class TestLastValidPriceInRange:
    def test_uses_last_trade_before_delisting(self):
        dates = pd.bdate_range("2024-01-08", "2024-01-12")  # Mon..Fri
        prices = pd.DataFrame({"A": [100.0, 102.0, np.nan, np.nan, np.nan]}, index=dates)
        out = _last_valid_price_in_range(prices, ["A"], dates[0], dates[-1])
        assert out["A"] == pytest.approx(102.0)

    def test_uses_exit_date_price_when_still_trading(self):
        dates = pd.bdate_range("2024-01-08", "2024-01-12")
        prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)
        out = _last_valid_price_in_range(prices, ["A"], dates[0], dates[-1])
        assert out["A"] == pytest.approx(104.0)


class TestRunBacktestMechanics:
    def _two_week_setup(self, execution="monday_close"):
        # ~85 trading days of burn-in before the decision Fridays we
        # actually test (2024-01-05, -12, -19), so ADV20 (min_periods=15)
        # and vol60 (min_periods=45) are already warmed up by then --
        # eligible_on() correctly returns nobody eligible without this. A
        # few extra trading days after 01-19 give that 3rd decision an
        # entry_date to enter on.
        dates = pd.bdate_range("2023-09-01", "2024-01-24")
        names = [f"T{i}" for i in range(10)]
        rng = np.random.default_rng(0)
        base = 100.0
        prices = {n: base + np.cumsum(rng.normal(scale=0.5, size=len(dates))) for n in names}
        panel = _make_panel(dates, prices)

        # Hand-crafted signal: T0..T4 always highest (long bucket), T5..T9
        # always lowest (short bucket), stable across all three decision
        # dates (T0=10 highest ... T9=1 lowest).
        decision_index = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-19")]
        s_vals = pd.DataFrame({n: [10 - i] * len(decision_index) for i, n in enumerate(names)}, index=decision_index)
        signal = SignalResult(u=None, q=None, r_n=None, s=s_vals)
        return panel, signal, dates

    def test_no_lookahead_weight_uses_only_data_through_friday(self):
        """Injecting a huge return strictly *after* the decision Friday must
        not change the weights decided for that week (portfolio depends
        only on `signal.s` and `panel.vol60`, both of which are already
        proven causal in test_signal.py; this test checks backtest.py
        itself doesn't peek at post-decision prices when building weights).
        """
        panel, signal, dates = self._two_week_setup()
        decisions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])
        params = PortfolioParams(n_buckets=2, gross_per_leg=0.5, per_name_cap=0.5, min_names_per_leg=2)

        bt1 = run_backtest(panel, signal, decisions, params, CostModel(0, 0), price_min=0, adv_min=0)

        panel2, _, _ = self._two_week_setup()
        panel2.adj_close.loc[pd.Timestamp("2024-01-16"):, :] *= 100.0  # blow up prices well after both entries
        bt2 = run_backtest(panel2, signal, decisions, params, CostModel(0, 0), price_min=0, adv_min=0)

        pd.testing.assert_series_equal(bt1.weights_by_date[decisions[0]], bt2.weights_by_date[decisions[0]])

    def test_full_replacement_turnover_between_two_disjoint_weeks(self):
        panel, signal, dates = self._two_week_setup()
        # A 3rd decision date is required for week 2 (2024-01-12) to have a
        # known exit point (exit_i = entry_{i+1}); with only 2 decisions,
        # week 2 has nowhere to exit and is correctly dropped.
        decisions = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-19")]
        )
        params = PortfolioParams(n_buckets=2, gross_per_leg=0.5, per_name_cap=0.5, min_names_per_leg=2)
        bt = run_backtest(panel, signal, decisions, params, CostModel(0, 0), price_min=0, adv_min=0)
        assert len(bt.weekly_turnover) == 2
        # Week 1 starts from flat -> turnover equals the full entry gross.
        assert bt.weekly_turnover.iloc[0] == pytest.approx(1.0, abs=1e-6)  # 0.5 long + 0.5 short from flat
        # Week 2 selects the *same 10 names* (signal is stable) -> turnover
        # should be small, but not exactly zero: inverse-vol weights within
        # each leg are recomputed from a 60d window that has rolled forward
        # by a week, so relative weights shift slightly even with an
        # unchanged membership.
        assert 0.0 < bt.weekly_turnover.iloc[1] < 0.1
        assert set(bt.longs_by_date[decisions[0]]) == set(bt.longs_by_date[decisions[1]])
        assert set(bt.shorts_by_date[decisions[0]]) == set(bt.shorts_by_date[decisions[1]])

    def test_costs_reduce_gross_return_by_turnover_times_one_way_bps(self):
        panel, signal, dates = self._two_week_setup()
        decisions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])
        params = PortfolioParams(n_buckets=2, gross_per_leg=0.5, per_name_cap=0.5, min_names_per_leg=2)
        costs = CostModel(commission_bps_per_side=2.0, half_spread_bps=5.0)
        bt = run_backtest(panel, signal, decisions, params, costs, price_min=0, adv_min=0)
        expected_cost = bt.weekly_turnover * (costs.one_way_bps() / 10_000.0)
        pd.testing.assert_series_equal(bt.weekly_costs, expected_cost, check_names=False)
        pd.testing.assert_series_equal(bt.weekly_returns, bt.weekly_gross_returns - expected_cost, check_names=False)

    def test_costs_hand_computed_example_not_double_counted(self):
        """Concrete, hand-computed check independent of the production cost
        formula (guards against a formula and its test drifting together):
        week 1 starts from flat, so turnover == the full entry gross
        (1.0 = 0.5 long + 0.5 short here). Trading $1.00 total notional at
        7bps one-way (2bp commission + 5bp half-spread) costs exactly
        $0.0007 -- not $0.0014, which is what charging the round-trip rate
        (14bps) against this same $1.00 would give.
        """
        panel, signal, dates = self._two_week_setup()
        decisions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])
        params = PortfolioParams(n_buckets=2, gross_per_leg=0.5, per_name_cap=0.5, min_names_per_leg=2)
        costs = CostModel(commission_bps_per_side=2.0, half_spread_bps=5.0)
        bt = run_backtest(panel, signal, decisions, params, costs, price_min=0, adv_min=0)
        assert bt.weekly_turnover.iloc[0] == pytest.approx(1.0, abs=1e-6)
        assert bt.weekly_costs.iloc[0] == pytest.approx(0.0007, abs=1e-9)

    def test_delisted_name_mid_week_uses_last_trade_not_nan(self):
        panel, signal, dates = self._two_week_setup()
        # T0 (top of the long leg) stops trading on 01-10, mid-week-1.
        panel.adj_close.loc[pd.Timestamp("2024-01-10"):, "T0"] = np.nan
        panel.close.loc[pd.Timestamp("2024-01-10"):, "T0"] = np.nan
        decisions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])
        params = PortfolioParams(n_buckets=2, gross_per_leg=0.5, per_name_cap=0.5, min_names_per_leg=2)
        bt = run_backtest(panel, signal, decisions, params, CostModel(0, 0), price_min=0, adv_min=0)
        # Must not raise, and week 1 gross return must be finite (not NaN).
        assert np.isfinite(bt.weekly_gross_returns.iloc[0])

    def test_execution_variant_monday_open_uses_open_prices(self):
        panel, signal, dates = self._two_week_setup()
        # A *uniform* scale factor would cancel out of an entry-to-exit
        # ratio; use per-day noise so open- and close-based returns
        # genuinely diverge.
        rng = np.random.default_rng(42)
        noise = 1.0 + rng.normal(scale=0.05, size=panel.adj_close.shape)
        panel.adj_open = panel.adj_close * noise
        decisions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])
        params = PortfolioParams(n_buckets=2, gross_per_leg=0.5, per_name_cap=0.5, min_names_per_leg=2)
        bt_close = run_backtest(panel, signal, decisions, params, CostModel(0, 0), execution="monday_close", price_min=0, adv_min=0)
        bt_open = run_backtest(panel, signal, decisions, params, CostModel(0, 0), execution="monday_open", price_min=0, adv_min=0)
        assert not np.isclose(bt_close.weekly_gross_returns.iloc[0], bt_open.weekly_gross_returns.iloc[0])

    def test_names_without_entry_price_excluded_before_weight_construction(self):
        """Regression-style check for the Oglegrinden-documented bug class:
        filtering tradability *after* building weights can leave gross
        exposure silently short of target. Here T0 has no valid entry price
        (NaN on the entry date itself) and must be dropped before weights
        are built, so the *remaining* long-leg names still sum to the full
        gross_per_leg target (renormalized over the truly tradable set).
        """
        panel, signal, dates = self._two_week_setup()
        panel.adj_close.loc[pd.Timestamp("2024-01-08"), "T0"] = np.nan
        decisions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])
        params = PortfolioParams(n_buckets=2, gross_per_leg=0.5, per_name_cap=0.5, min_names_per_leg=2)
        bt = run_backtest(panel, signal, decisions, params, CostModel(0, 0), price_min=0, adv_min=0)
        w1 = bt.weights_by_date[decisions[0]]
        assert "T0" not in w1.index
        assert w1[w1 > 0].sum() == pytest.approx(0.5, abs=1e-6)


class TestBenchmarkWeeklyReturns:
    def test_matches_manual_close_to_close_computation(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-19")
        rng = np.random.default_rng(1)
        spy = pd.Series(100 + np.cumsum(rng.normal(size=len(dates))), index=dates)
        decisions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")])
        out = benchmark_weekly_returns(spy, dates, decisions)
        entry0, exit0 = pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")
        expected0 = spy[exit0] / spy[entry0] - 1.0
        assert out[decisions[0]] == pytest.approx(expected0)
