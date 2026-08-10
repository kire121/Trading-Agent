from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vridmomentet.config import PortfolioParams
from vridmomentet.portfolio import (
    assign_bucket,
    build_target_weights,
    cap_and_renormalize,
    inverse_vol_weights,
    select_legs,
    turnover,
)


class TestAssignBucket:
    def test_quintile_split_of_25_names_is_5_and_5_per_extreme_bucket(self):
        s = pd.Series(np.arange(25), index=[f"T{i}" for i in range(25)])
        buckets = assign_bucket(s, n_buckets=5)
        counts = buckets.value_counts().sort_index()
        assert list(counts.values) == [5, 5, 5, 5, 5]
        top_names = buckets[buckets == 4].index
        assert set(top_names) == {f"T{i}" for i in range(20, 25)}

    def test_nan_stays_nan_and_is_excluded_from_bucketing(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], index=list("abcde"))
        buckets = assign_bucket(s, n_buckets=5)
        assert np.isnan(buckets["c"])
        assert buckets.drop("c").notna().all()


class TestSelectLegs:
    def test_long_short_are_disjoint_and_from_extremes(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=100), index=[f"T{i}" for i in range(100)])
        longs, shorts = select_legs(s, n_buckets=5)
        assert set(longs).isdisjoint(shorts)
        # every long name's signal must exceed every short name's signal
        assert s[longs].min() > s[shorts].max()

    def test_full_replacement_is_stateless(self):
        """Calling select_legs twice on different signals must not carry any
        retention/hysteresis from the first call -- this is a "full
        replacement", not a hysteresis-banded rebalance.
        """
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=[f"T{i}" for i in range(10)])
        s2 = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], index=[f"T{i}" for i in range(10)])
        longs1, shorts1 = select_legs(s1, n_buckets=5)
        longs2, shorts2 = select_legs(s2, n_buckets=5)
        assert set(longs2) == set(shorts1)
        assert set(shorts2) == set(longs1)


class TestInverseVolWeights:
    def test_weights_sum_to_one_and_favor_low_vol_names(self):
        names = ["A", "B", "C"]
        vol = pd.Series({"A": 0.01, "B": 0.02, "C": 0.04})
        w = inverse_vol_weights(names, vol)
        assert w.sum() == pytest.approx(1.0)
        assert w["A"] > w["B"] > w["C"]

    def test_missing_or_zero_vol_name_is_dropped(self):
        names = ["A", "B", "C"]
        vol = pd.Series({"A": 0.01, "B": 0.0, "C": np.nan})
        w = inverse_vol_weights(names, vol)
        assert list(w.index) == ["A"]
        assert w["A"] == pytest.approx(1.0)


class TestCapAndRenormalize:
    def test_uncapped_case_is_unchanged(self):
        w = pd.Series({"A": 0.1, "B": 0.2, "C": -0.15})
        out = cap_and_renormalize(w, cap=0.5)
        pd.testing.assert_series_equal(out, w)

    def test_cap_is_never_violated(self):
        rng = np.random.default_rng(1)
        raw = pd.Series(rng.normal(size=20), index=[f"T{i}" for i in range(20)])
        raw = raw / raw.abs().sum() * 0.5  # gross 0.5, one name may dominate
        out = cap_and_renormalize(raw, cap=0.02)
        assert (out.abs() <= 0.02 + 1e-9).all()

    def test_infeasible_gross_does_not_violate_cap(self):
        """3 names, cap=0.1 each -> max achievable gross is 0.3, well below
        a 1.0 target; the achieved gross must fall short, not breach the cap.
        """
        w = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        out = cap_and_renormalize(w, cap=0.1)
        assert (out.abs() <= 0.1 + 1e-9).all()
        assert out.abs().sum() == pytest.approx(0.3, abs=1e-6)

    def test_capped_names_are_exactly_at_cap_remainder_redistributed(self):
        # gross target 0.5, cap 0.10: A is oversized, B..F evenly share the
        # rest; after capping A, redistributing the remainder among B..F
        # (0.40 / 5 = 0.08 each) stays comfortably under the cap, so this
        # is a clean single-iteration, fully-feasible case.
        w = pd.Series({"A": 0.30, "B": 0.04, "C": 0.04, "D": 0.04, "E": 0.04, "F": 0.04})
        out = cap_and_renormalize(w, cap=0.10)
        assert out["A"] == pytest.approx(0.10, abs=1e-6)
        for name in "BCDEF":
            assert out[name] == pytest.approx(0.08, abs=1e-6)
        assert out.abs().sum() == pytest.approx(0.5, abs=1e-6)

    def test_two_names_simultaneously_at_cap_both_get_capped_in_one_pass(self):
        """A and D both start at or above the cap; the algorithm must catch
        both in its first pass (not just the single largest name), then
        redistribute among the genuinely-free B, C.
        """
        w = pd.Series({"A": 0.30, "B": 0.05, "C": 0.05, "D": 0.10})  # gross 0.5, cap 0.1
        out = cap_and_renormalize(w, cap=0.10)
        assert out["A"] == pytest.approx(0.10, abs=1e-6)
        assert out["D"] == pytest.approx(0.10, abs=1e-6)
        assert (out.abs() <= 0.10 + 1e-9).all()
        # 4 names x 0.10 cap = 0.40 max achievable, below the 0.5 target:
        # infeasible gross, so the cap wins and gross legitimately falls short.
        assert out.abs().sum() == pytest.approx(0.40, abs=1e-6)


class TestBuildTargetWeights:
    def _synthetic(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        names = [f"T{i}" for i in range(n)]
        s = pd.Series(rng.normal(size=n), index=names)
        vol = pd.Series(rng.uniform(0.01, 0.05, size=n), index=names)
        return s, vol

    def test_market_neutral_exactly_by_construction(self):
        # 200 names -> 40/leg at quintile buckets; 40 * 2% cap = 80% max
        # achievable per leg, comfortably above the 50% target, so this is
        # a feasible case and the exact-neutrality claim is meaningful.
        s, vol = self._synthetic(n=200)
        params = PortfolioParams(n_buckets=5, gross_per_leg=0.5, per_name_cap=0.02, min_names_per_leg=5)
        port = build_target_weights(s, vol, params)
        assert port.long_weights.sum() == pytest.approx(0.5, abs=1e-9)
        assert port.short_weights.sum() == pytest.approx(-0.5, abs=1e-9)
        assert port.weights.sum() == pytest.approx(0.0, abs=1e-9)

    def test_per_name_cap_respected(self):
        s, vol = self._synthetic(n=20)
        params = PortfolioParams(n_buckets=5, gross_per_leg=0.5, per_name_cap=0.02, min_names_per_leg=2)
        port = build_target_weights(s, vol, params)
        assert (port.weights.abs() <= 0.02 + 1e-9).all()

    def test_too_few_names_returns_empty_portfolio(self):
        s = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        vol = pd.Series([0.01, 0.02, 0.03], index=["A", "B", "C"])
        params = PortfolioParams(n_buckets=5, min_names_per_leg=5)
        port = build_target_weights(s, vol, params)
        assert port.weights.empty
        assert port.long_names == []
        assert port.short_names == []

    def test_long_names_have_higher_signal_than_short_names(self):
        s, vol = self._synthetic()
        port = build_target_weights(s, vol)
        assert s[port.long_names].min() > s[port.short_names].max()


class TestTurnover:
    def test_full_replacement_gives_turnover_equal_to_sum_of_abs_weights(self):
        prev = pd.Series({"A": 0.25, "B": -0.25})
        new = pd.Series({"C": 0.25, "D": -0.25})
        # completely disjoint names -> turnover = sum(|prev|) + sum(|new|)
        assert turnover(prev, new) == pytest.approx(1.0)

    def test_unchanged_portfolio_has_zero_turnover(self):
        w = pd.Series({"A": 0.25, "B": -0.25})
        assert turnover(w, w) == pytest.approx(0.0)
