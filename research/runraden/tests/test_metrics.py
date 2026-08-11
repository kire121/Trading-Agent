import numpy as np
import pandas as pd

from metrics import deflated_sharpe_ratio, pc1_share, pooled_ic, sharpe_ratio


def test_pooled_ic_perfect_correlation():
    pred = pd.Series(np.arange(100, dtype=float))
    target = pred * 2 + 1
    assert np.isclose(pooled_ic(pred, target), 1.0)


def test_pooled_ic_handles_nans_and_small_samples():
    pred = pd.Series([np.nan] * 5)
    target = pd.Series([1.0] * 5)
    assert np.isnan(pooled_ic(pred, target))


def test_sharpe_ratio_scales_with_sqrt_periods():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, 500))
    sr = sharpe_ratio(r, periods_per_year=52)
    expected = r.mean() / r.std() * np.sqrt(52)
    assert np.isclose(sr, expected)


def test_pc1_share_single_factor_structure():
    rng = np.random.default_rng(0)
    n, k = 500, 10
    factor = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.01, (n, k))  # tiny noise relative to the common factor
    loadings = rng.normal(1, 0.1, k)
    X = np.outer(factor, loadings) + noise
    share = pc1_share(X)
    assert share > 0.9  # dominated by the single common factor


def test_pc1_share_no_common_structure():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 10))  # independent columns
    share = pc1_share(X)
    assert share < 0.5


def test_deflated_sharpe_ratio_more_trials_lowers_dsr():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, 300))
    dsr_1 = deflated_sharpe_ratio(r, n_trials=1)
    dsr_many = deflated_sharpe_ratio(r, n_trials=100)
    assert dsr_many["sr_benchmark_annual"] >= dsr_1["sr_benchmark_annual"]
    assert dsr_many["dsr"] <= dsr_1["dsr"]


def test_deflated_sharpe_ratio_short_series_returns_nan():
    r = pd.Series([0.01, -0.01, 0.02])
    out = deflated_sharpe_ratio(r, n_trials=27)
    assert np.isnan(out["dsr"])
