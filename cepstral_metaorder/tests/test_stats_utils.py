import numpy as np
import pytest
from scipy import stats as scipy_stats

from cepstral_metaorder import stats_utils as su


def test_newey_west_matches_naive_ttest_for_iid_data_at_zero_lags():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, 2000)
    nw = su.newey_west_mean_tstat(x, lags=0)
    naive_t, _ = scipy_stats.ttest_1samp(x, 0.0)
    assert nw["t_stat"] == pytest.approx(naive_t, rel=0.02)


def test_newey_west_tstat_is_large_for_a_clearly_nonzero_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 0.1, 500)
    nw = su.newey_west_mean_tstat(x, lags=5)
    assert nw["t_stat"] > 10


def test_newey_west_tstat_is_small_for_zero_mean_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 500)
    nw = su.newey_west_mean_tstat(x, lags=5)
    assert abs(nw["t_stat"]) < 3.0


def test_newey_west_se_grows_with_positive_autocorrelation():
    rng = np.random.default_rng(3)
    n = 1000
    eps = rng.normal(0, 1, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + eps[i]
    ar += 0.3  # nonzero mean so t-stats are meaningfully comparable

    nw_adjusted = su.newey_west_mean_tstat(ar, lags=10)
    nw_unadjusted = su.newey_west_mean_tstat(ar, lags=0)
    assert nw_adjusted["se"] > nw_unadjusted["se"]
    assert abs(nw_adjusted["t_stat"]) < abs(nw_unadjusted["t_stat"])


def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.001, 0.01, 500)
    trial_sharpes = rng.normal(0, 0.3, 9)
    dsr_few = su.deflated_sharpe_ratio(returns, n_trials=1, trial_sharpes=trial_sharpes, periods_per_year=252)
    dsr_many = su.deflated_sharpe_ratio(returns, n_trials=100, trial_sharpes=trial_sharpes, periods_per_year=252)
    assert dsr_many["sr_benchmark_per_period"] > dsr_few["sr_benchmark_per_period"]
    assert dsr_many["dsr"] <= dsr_few["dsr"]


def test_deflated_sharpe_is_low_for_pure_noise():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.01, 500)
    trial_sharpes = rng.normal(0, 0.3, 9)
    result = su.deflated_sharpe_ratio(returns, n_trials=9, trial_sharpes=trial_sharpes, periods_per_year=252)
    assert result["dsr"] < 0.7


def test_deflated_sharpe_is_high_for_strong_consistent_edge():
    rng = np.random.default_rng(6)
    returns = rng.normal(0.003, 0.01, 1000)  # daily SR ~0.3 => annualized ~4.8, a very strong edge
    # realistic cross-trial Sharpe dispersion at daily frequency (annualized
    # Sharpes clustered within roughly +-0.5 => daily std ~0.03), NOT the same
    # order of magnitude as the edge itself
    trial_sharpes = rng.normal(0, 0.03, 9)
    result = su.deflated_sharpe_ratio(returns, n_trials=9, trial_sharpes=trial_sharpes, periods_per_year=252)
    assert result["dsr"] > 0.9
