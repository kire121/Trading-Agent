import numpy as np
import pandas as pd
import pytest

from research.dammluckan import metrics


def test_sharpe_hand_computed():
    r = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(metrics.sharpe(r) - expected) < 1e-12


def test_expected_max_sharpe_requires_multiple_trials():
    with pytest.raises(ValueError):
        metrics.expected_max_sharpe([0.1])


def test_expected_max_sharpe_increases_with_trial_count():
    rng = np.random.default_rng(0)
    trials_small = rng.normal(0, 0.05, 5)
    trials_large = rng.normal(0, 0.05, 500)
    small = metrics.expected_max_sharpe(trials_small)
    large = metrics.expected_max_sharpe(trials_large)
    assert large > small  # more trials -> higher expected max under the null


def test_dsr_clear_outlier_scores_high():
    rng = np.random.default_rng(1)
    trials = rng.normal(0, 0.02, 50)
    sr_hat = 0.15  # far above the pack
    out = metrics.deflated_sharpe_ratio(sr_hat, n_obs=1000, sr_trials=trials)
    assert out["dsr_prob"] > 0.99


def test_dsr_typical_trial_scores_low():
    rng = np.random.default_rng(2)
    trials = rng.normal(0, 0.02, 50)
    sr_hat = float(np.median(trials))
    out = metrics.deflated_sharpe_ratio(sr_hat, n_obs=1000, sr_trials=trials)
    assert out["dsr_prob"] < 0.9


def test_pnl_quarter_concentration_single_quarter():
    dates = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(0.0, index=dates)
    r.iloc[10:15] = 0.05  # all PnL concentrated in one quarter
    share = metrics.pnl_quarter_concentration(r)
    assert share > 0.99


def test_pnl_quarter_concentration_evenly_spread():
    dates = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(0.0001, index=dates)  # flat positive drip, evenly spread
    share = metrics.pnl_quarter_concentration(r)
    assert share < 0.2


def test_sub_period_sign_consistency_all_positive():
    dates = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(0.0005, index=dates)
    res = metrics.sub_period_sign_consistency(r, dates[0], dates[-1], n_periods=4)
    assert res["n_inconsistent"] == 0


def test_event_concentration_flags_dominant_asset():
    from collections import namedtuple
    T = namedtuple("T", ["ticker"])
    trades = [T("SPY")] * 60 + [T("TLT")] * 40
    res = metrics.event_concentration(trades, ["SPY", "TLT", "GLD"])
    assert res["max_ticker"] == "SPY"
    assert abs(res["max_share"] - 0.6) < 1e-12
