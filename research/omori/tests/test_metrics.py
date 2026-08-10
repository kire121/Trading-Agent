import numpy as np
import pytest

from research.omori import metrics


class TestSharpe:
    def test_zero_for_constant_returns(self):
        r = np.zeros(100)
        assert metrics.annualized_sharpe(r) == 0.0

    def test_known_value(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, size=2520)
        sr = metrics.annualized_sharpe(r)
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        assert sr == pytest.approx(expected)

    def test_positive_drift_gives_positive_sharpe(self):
        r = np.full(500, 0.0005)
        r[::2] += 1e-6  # break exact-zero variance
        assert metrics.annualized_sharpe(r) > 0


class TestDrawdown:
    def test_no_drawdown_for_monotonic_gains(self):
        r = np.full(100, 0.001)
        assert metrics.max_drawdown(r) == pytest.approx(0.0, abs=1e-9)

    def test_known_single_drop(self):
        r = np.array([0.0, -0.5, 0.0, 0.0])
        dd = metrics.max_drawdown(r)
        assert dd == pytest.approx(-0.5)


class TestDSR:
    def test_psr_half_at_benchmark(self):
        p = metrics.probabilistic_sharpe_ratio(sr_hat=1.0, benchmark_sr=1.0, n_obs=252)
        assert p == pytest.approx(0.5, abs=1e-6)

    def test_psr_increases_with_sr_hat(self):
        low = metrics.probabilistic_sharpe_ratio(0.5, 0.0, 252)
        high = metrics.probabilistic_sharpe_ratio(1.5, 0.0, 252)
        assert high > low

    def test_expected_max_sharpe_zero_variance(self):
        assert metrics.expected_max_sharpe([1.0]) == 0.0

    def test_deflated_sharpe_penalizes_many_trials(self):
        rng = np.random.default_rng(0)
        trials_few = rng.normal(0, 0.3, size=3)
        trials_many = rng.normal(0, 0.3, size=200)
        dsr_few = metrics.deflated_sharpe_ratio(1.0, 500, trials_few)
        dsr_many = metrics.deflated_sharpe_ratio(1.0, 500, trials_many)
        assert dsr_many["expected_max_sr"] >= dsr_few["expected_max_sr"]


class TestSummarize:
    def test_summarize_keys(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, size=500)
        out = metrics.summarize(r)
        assert set(["sharpe", "annualized_return", "annualized_vol", "max_drawdown",
                     "n_days", "skew", "kurtosis"]) <= set(out.keys())
