import numpy as np
import pandas as pd
import pytest

from oglegrinden.stats import (
    sharpe_ratio,
    max_drawdown,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    block_bootstrap_gate_test,
    gated_vs_always_on,
    twin_gate_regression,
    max_pnl_concentration,
    subperiod_sign_check,
)


def test_sharpe_ratio_known_value():
    r = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.02])
    sr = sharpe_ratio(r, periods_per_year=52, annualize=False)
    assert np.isclose(sr, r.mean() / r.std(ddof=1))


def test_sharpe_ratio_zero_vol_is_zero_not_inf():
    r = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(r) == 0.0


def test_max_drawdown_simple_path():
    # wealth path: 1 -> 1.1 -> 0.99 -> 1.05
    r = pd.Series([0.10, -0.10, 0.0606060606])
    dd = max_drawdown(r)
    wealth = (1 + r).cumprod()
    expected = float((wealth / wealth.cummax() - 1).min())
    assert np.isclose(dd, expected)
    assert dd < 0


def test_deflated_sharpe_ratio_high_when_best_trial_clearly_stands_out():
    rng = np.random.default_rng(0)
    # 29 "noise" trials with Sharpe ~N(0, 0.05), one clear winner at 1.2
    noise_trials = rng.normal(0, 0.02, 29)
    trials = np.append(noise_trials, 1.2)
    result = deflated_sharpe_ratio(observed_sharpe_per_period=1.2, trial_sharpes_per_period=trials, n_obs=500)
    assert result["dsr"] > 0.99


def test_deflated_sharpe_ratio_low_when_best_is_typical_of_the_pack():
    rng = np.random.default_rng(0)
    trials = rng.normal(0.05, 0.05, 30)
    observed = float(np.median(trials))
    result = deflated_sharpe_ratio(observed_sharpe_per_period=observed, trial_sharpes_per_period=trials, n_obs=500)
    assert result["dsr"] < 0.9


def test_expected_max_sharpe_increases_with_number_of_trials():
    rng = np.random.default_rng(1)
    small = rng.normal(0, 0.05, 5)
    large = rng.normal(0, 0.05, 200)
    sr0_small = expected_max_sharpe_under_null(small)["sr0"]
    sr0_large = expected_max_sharpe_under_null(large)["sr0"]
    assert sr0_large > sr0_small  # more trials -> higher bar from luck alone


def test_block_bootstrap_gate_test_random_gate_gives_high_p_value():
    """If the gate carries no real information (independent of returns),
    a randomly-placed gate with the same ON-fraction should do about as
    well as the actual gate on average -> p-value should not be small."""
    rng = np.random.default_rng(2)
    n = 300
    idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    returns = pd.Series(rng.normal(0.001, 0.02, n), index=idx)
    gate = pd.Series(rng.random(n) < 0.4, index=idx)  # independent of returns
    result = block_bootstrap_gate_test(gate, returns, n_boot=200, block_size=13, seed=3)
    assert result["p_value"] > 0.05


def test_block_bootstrap_gate_test_perfect_foresight_gate_gives_low_p_value():
    """A gate that is ON exactly during the best weeks should crush the
    bootstrap null (which preserves ON-fraction/persistence but not the
    correlation with returns)."""
    rng = np.random.default_rng(4)
    n = 300
    idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    returns = pd.Series(rng.normal(0.0, 0.02, n), index=idx)
    threshold = returns.quantile(0.6)
    gate = returns > threshold  # picks the best ~40% of weeks by construction
    result = block_bootstrap_gate_test(gate, returns, n_boot=200, block_size=13, seed=5)
    assert result["p_value"] < 0.05


def test_gated_vs_always_on_flags_beats_correctly():
    idx = pd.date_range("2010-01-01", periods=100, freq="W-FRI")
    gated = pd.Series(0.001, index=idx)
    baseline = pd.Series(0.0005, index=idx)
    # add a touch of noise so std isn't degenerate zero for one series
    gated = gated + np.sin(np.arange(100)) * 1e-5
    baseline = baseline + np.cos(np.arange(100)) * 1e-5
    result = gated_vs_always_on(gated, baseline)
    assert result["gated"]["annualized_return"] > result["always_on"]["annualized_return"]


def test_twin_gate_regression_recovers_known_positive_coefficient():
    rng = np.random.default_rng(6)
    n = 400
    idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    L = pd.Series(rng.normal(0, 1, n), index=idx)
    rho_bar = pd.Series(rng.normal(0, 1, n), index=idx)
    ar = pd.Series(rng.normal(0, 1, n), index=idx)
    true_b = 0.02
    y = true_b * L + 0.001 * rho_bar - 0.001 * ar + rng.normal(0, 0.01, n)
    y = pd.Series(y, index=idx)

    reg = twin_gate_regression(y, L, rho_bar, ar)
    assert reg["b_L"] > 0
    assert reg["p_L"] < 0.05


def test_twin_gate_regression_null_when_L_carries_no_information():
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
    L = pd.Series(rng.normal(0, 1, n), index=idx)
    rho_bar = pd.Series(rng.normal(0, 1, n), index=idx)
    ar = pd.Series(rng.normal(0, 1, n), index=idx)
    y = pd.Series(rng.normal(0, 0.01, n), index=idx)  # pure noise, no relation to L

    reg = twin_gate_regression(y, L, rho_bar, ar)
    assert reg["p_L"] > 0.05


def test_subperiod_sign_check_handles_series_with_different_indices():
    """forward_pnl spans every decision Friday, but the smoothed topology
    signals only exist for weeks where the universe was large enough --
    a shorter, differently-shaped index. subperiod_sign_check must not
    assume they share an index (regression test for a boolean-mask bug
    that crashed on real data: mask built from forward_pnl's 1750-week
    index applied to a 1043-week signal series)."""
    rng = np.random.default_rng(8)
    full_idx = pd.date_range("2000-01-01", periods=800, freq="W-FRI")
    short_idx = full_idx[300:]  # signal only available for the back 500 weeks
    forward_pnl = pd.Series(rng.normal(0, 0.01, len(full_idx)), index=full_idx)
    L = pd.Series(rng.normal(0, 1, len(short_idx)), index=short_idx)
    rho_bar = pd.Series(rng.normal(0, 1, len(short_idx)), index=short_idx)
    ar = pd.Series(rng.normal(0, 1, len(short_idx)), index=short_idx)

    boundaries = [
        (pd.Timestamp("2000-01-01"), pd.Timestamp("2005-12-31")),  # entirely before signal exists
        (pd.Timestamp("2010-01-01"), pd.Timestamp("2015-12-31")),  # overlaps signal
    ]
    results = subperiod_sign_check(forward_pnl, L, rho_bar, ar, boundaries)
    assert results[0]["insufficient_data"] is True
    assert results[1]["insufficient_data"] is False
    assert "b_L" in results[1]


def test_max_pnl_concentration_flags_single_dominant_window():
    idx = pd.date_range("2010-01-01", periods=50, freq="W-FRI")
    r = pd.Series(0.001, index=idx)
    r.iloc[20:28] = 0.5 / 8  # one dominant 8-week window
    result = max_pnl_concentration(r, window=8)
    assert result["share"] > 0.5
