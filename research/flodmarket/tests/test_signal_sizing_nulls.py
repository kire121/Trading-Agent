import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.flodmarket import signal, sizing, nulls, costs  # noqa: E402


class WeeklyDecisionDatesTests(unittest.TestCase):
    def test_returns_last_trading_day_per_iso_week(self):
        dates = pd.bdate_range("2024-01-01", periods=20)  # Mon 2024-01-01
        wk = signal.weekly_decision_dates(dates)
        self.assertTrue(len(wk) >= 3)
        # Every returned date must be the max date in its own ISO week among `dates`.
        iso = pd.Series(dates, index=dates).index.isocalendar()
        for d in wk:
            key = (iso.loc[d, "year"], iso.loc[d, "week"])
            same_week = dates[(iso["year"] == key[0]) & (iso["week"] == key[1])]
            self.assertEqual(d, same_week.max())


class ComputeGTests(unittest.TestCase):
    def test_clips_to_unit_interval(self):
        S = pd.Series([10.0, -10.0, 1.0, -1.0, np.nan])
        g = signal.compute_g(S, z_star=2.0)
        self.assertAlmostEqual(g.iloc[0], 1.0)
        self.assertAlmostEqual(g.iloc[1], -1.0)
        self.assertAlmostEqual(g.iloc[2], 0.5)
        self.assertTrue(np.isnan(g.iloc[4]))


class SizingTests(unittest.TestCase):
    def test_gross_cap_never_exceeded(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        rng = np.random.default_rng(0)
        raw = pd.DataFrame(rng.normal(0, 5, (100, 10)), index=idx, columns=[f"T{i}" for i in range(10)])
        capped = sizing.apply_gross_cap(raw, k=1.0, gross_cap=2.0)
        gross = capped.abs().sum(axis=1)
        self.assertTrue((gross <= 2.0 + 1e-9).all())

    def test_solve_k_hits_target_vol_approximately(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        rng = np.random.default_rng(1)
        raw = pd.DataFrame(rng.normal(0, 1, (500, 5)), index=idx, columns=[f"T{i}" for i in range(5)])
        rets = pd.DataFrame(rng.normal(0, 0.01, (500, 5)), index=idx, columns=[f"T{i}" for i in range(5)])
        k = sizing.solve_k_for_target_vol(raw, rets, idx[0].isoformat(), idx[-1].isoformat(),
                                           target_vol=0.10, gross_cap=2.0)
        weights = sizing.apply_gross_cap(raw, k, 2.0)
        realized = sizing.portfolio_returns(weights, rets, apply_costs=False)
        realized_vol = realized.std() * np.sqrt(252)
        self.assertAlmostEqual(realized_vol, 0.10, delta=0.02)


class CostsTests(unittest.TestCase):
    def test_illiquid_gets_highest_cost(self):
        adv = pd.Series([1e9, 150e6, 0.0])
        bps = costs.half_spread_bps(adv)
        self.assertTrue(bps.iloc[0] < bps.iloc[1] < bps.iloc[2])


class NullsTests(unittest.TestCase):
    def test_block_permute_preserves_multiset(self):
        rng = np.random.default_rng(0)
        arr = np.arange(100, dtype=float)
        permuted = nulls.block_permute_1d(arr, block_len=7, rng=rng)
        self.assertEqual(sorted(permuted.tolist()), sorted(arr.tolist()))
        self.assertEqual(len(permuted), len(arr))

    def test_block_permute_within_ticker_no_cross_boundary_leak(self):
        idx = pd.MultiIndex.from_product([["AAA", "BBB"], pd.date_range("2020-01-01", periods=50, freq="B")],
                                          names=["ticker", "date"])
        vals = pd.Series(np.concatenate([np.arange(50, dtype=float), np.arange(1000, 1050, dtype=float)]), index=idx)
        rng = np.random.default_rng(0)
        permuted = nulls.block_permute_within_ticker(vals, block_len=10, rng=rng)
        aaa_vals = permuted.xs("AAA", level="ticker")
        bbb_vals = permuted.xs("BBB", level="ticker")
        self.assertTrue((aaa_vals < 100).all())
        self.assertTrue((bbb_vals >= 1000).all())
        self.assertEqual(sorted(aaa_vals.tolist()), list(range(50)))


if __name__ == "__main__":
    unittest.main()
