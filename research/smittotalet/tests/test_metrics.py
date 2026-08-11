import numpy as np
import pandas as pd

from .. import metrics


def test_sharpe_of_constant_positive_return_is_high():
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    r = pd.Series(0.001, index=idx)
    r.iloc[::7] += 0.0001  # tiny noise so std != 0
    assert metrics.sharpe(r) > 0


def test_max_drawdown_is_negative_or_zero():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(-0.001, 0.02, 100), index=idx)
    dd = metrics.max_drawdown(r)
    assert dd <= 0


def test_deflated_sharpe_ratio_penalizes_more_trials():
    sr_trials_small = np.random.default_rng(0).normal(0, 0.05, 5)
    sr_trials_large = np.tile(sr_trials_small, 20)
    dsr_small = metrics.deflated_sharpe_ratio(0.1, 500, sr_trials_small)
    dsr_large = metrics.deflated_sharpe_ratio(0.1, 500, sr_trials_large)
    assert dsr_large["expected_max_sr"] >= dsr_small["expected_max_sr"]


def test_sub_period_sign_consistency_two_halves():
    idx = pd.date_range("2010-01-01", periods=1000, freq="B")
    r = pd.Series(0.001, index=idx)
    out = metrics.sub_period_sign_consistency(r, idx[0], idx[-1], n_periods=2)
    assert out["consistency"] == 1.0
