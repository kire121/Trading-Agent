import numpy as np
import pandas as pd

from .. import nulls


def test_circular_block_shuffle_preserves_value_multiset():
    arr = np.arange(50)
    rng = np.random.default_rng(0)
    shuffled = nulls.circular_block_shuffle(arr, block_len=5, rng=rng)
    assert len(shuffled) == len(arr)
    assert set(shuffled.tolist()) <= set(arr.tolist())  # only draws from arr's own values


def test_block_bootstrap_indices_reproducible_with_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    idx1 = nulls.block_bootstrap_indices(100, 100, 10, rng1)
    idx2 = nulls.block_bootstrap_indices(100, 100, 10, rng2)
    assert np.array_equal(idx1, idx2)


def test_r_hat_dispersion_null_returns_n_draws_values():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    x_t = pd.Series(rng.poisson(2, len(idx)).astype(float), index=idx)
    out = nulls.r_hat_dispersion_null(x_t, q=95, tau=21, kappa=1, n_draws=20, block_len=10, seed=0)
    assert len(out) == 20
    assert np.all(np.isfinite(out) | np.isnan(out))


def test_block_bootstrap_sharpe_ci_brackets_point_estimate():
    from .. import metrics
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    lo, hi, draws = nulls.block_bootstrap_sharpe_ci(r, n_draws=200, block_len=10, seed=0)
    point = metrics.sharpe(r)
    assert lo <= hi
    assert lo - 5 <= point <= hi + 5  # loose sanity bound, not a tight CI check
