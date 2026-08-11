import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.metrics import (  # noqa: E402
    annualized_return,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    max_drawdown,
    newey_west_tstat_nodeps,
    sharpe_ratio,
    sortino_ratio,
    summary_stats,
)


class BasicMetricsTests(unittest.TestCase):
    def test_sharpe_ratio_zero_for_constant_returns(self):
        # 0.0 är exakt representerbart i binärt flyttal, så std blir exakt 0 --
        # inga flyttalsrester som (t.ex. med upprepat 0.01) kan ge ett
        # astronomiskt kvot-resultat istället för den avsedda 0.0-konventionen.
        r = pd.Series([0.0] * 20)
        self.assertEqual(sharpe_ratio(r), 0.0)  # std=0 -> 0.0 per konvention (ej NaN)

    def test_sharpe_ratio_positive_for_positive_drift(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.01, 0.02, 500))
        self.assertGreater(sharpe_ratio(r), 0.0)

    def test_max_drawdown_known_case(self):
        # +10%, -20%, +10%: toppen efter första steget, sedan en 20%-nedgång.
        r = pd.Series([0.10, -0.20, 0.10])
        dd = max_drawdown(r)
        self.assertAlmostEqual(dd, -0.20, places=6)

    def test_max_drawdown_never_positive(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.001, 0.01, 200))
        self.assertLessEqual(max_drawdown(r), 0.0)

    def test_sortino_ratio_no_downside_is_inf_or_zero(self):
        r = pd.Series([0.01, 0.02, 0.03])
        result = sortino_ratio(r)
        self.assertTrue(result == float("inf") or result == 0.0)

    def test_annualized_return_zero_length_is_zero(self):
        r = pd.Series([], dtype=float)
        self.assertEqual(annualized_return(r), 0.0)

    def test_summary_stats_has_expected_keys(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0.001, 0.01, 100))
        stats = summary_stats(r)
        for key in ("n_obs", "annualized_return", "annualized_vol", "sharpe", "sortino",
                    "max_drawdown", "skew", "kurtosis", "hit_rate"):
            self.assertIn(key, stats)


class DeflatedSharpeTests(unittest.TestCase):
    def test_expected_max_sharpe_requires_at_least_two_trials(self):
        with self.assertRaises(ValueError):
            expected_max_sharpe_under_null([0.1])

    def test_deflated_sharpe_ratio_runs_and_bounds_dsr_in_0_1(self):
        rng = np.random.default_rng(3)
        trial_sharpes = rng.normal(0, 0.05, 20).tolist()
        result = deflated_sharpe_ratio(
            observed_sharpe_per_period=0.08, trial_sharpes_per_period=trial_sharpes, n_obs=250,
        )
        self.assertGreaterEqual(result["dsr"], 0.0)
        self.assertLessEqual(result["dsr"], 1.0)

    def test_higher_observed_sharpe_gives_higher_dsr(self):
        rng = np.random.default_rng(4)
        trial_sharpes = rng.normal(0, 0.05, 20).tolist()
        low = deflated_sharpe_ratio(0.02, trial_sharpes, n_obs=250)
        high = deflated_sharpe_ratio(0.20, trial_sharpes, n_obs=250)
        self.assertGreater(high["dsr"], low["dsr"])


class NeweyWestNodepsTests(unittest.TestCase):
    def test_significant_mean_gives_large_tstat(self):
        rng = np.random.default_rng(5)
        x = rng.normal(1.0, 0.1, 200)  # klart skild från 0
        result = newey_west_tstat_nodeps(x, lags=4)
        self.assertGreater(abs(result["t_stat"]), 5.0)

    def test_too_short_series_returns_nan(self):
        result = newey_west_tstat_nodeps(np.array([1.0, 2.0]), lags=4)
        self.assertTrue(np.isnan(result["t_stat"]))


class NeweyWestStatsmodelsTests(unittest.TestCase):
    def test_matches_nodeps_direction_qualitatively(self):
        from lib.metrics import newey_west_tstat  # noqa: PLC0415 (lazy-imports statsmodels internally)
        rng = np.random.default_rng(6)
        r = pd.Series(rng.normal(0.5, 0.1, 200))
        try:
            t = newey_west_tstat(r)
        except ImportError:
            self.skipTest("statsmodels not installed")
            return
        self.assertGreater(t, 5.0)


if __name__ == "__main__":
    unittest.main()
