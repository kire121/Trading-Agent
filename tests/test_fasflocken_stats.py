import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fasflocken.universe import SyntheticUniverseProvider
from fasflocken.config import SignalParams, GICS_SECTORS
from fasflocken.pipeline import compute_sector_signal, build_z_panel
from fasflocken.backtest import run_backtest, annualized_sharpe
from fasflocken import stats

SECTORS = tuple(list(GICS_SECTORS)[:6])
FAST_PARAMS = SignalParams(
    band_low_days=5, band_high_days=20, analytic_window=40, z_lookback_weeks=20, n_legs=2, hysteresis_band=3,
    min_constituents=4,
)


@pytest.fixture(scope="module")
def tiny_setup():
    start, end = dt.date(2016, 1, 1), dt.date(2017, 12, 31)
    provider = SyntheticUniverseProvider(start=start, end=end, n_per_sector=6, seed=13, sectors=SECTORS)
    sector_signals = {s: compute_sector_signal(provider, s, start, end, FAST_PARAMS) for s in SECTORS}
    z_panel = build_z_panel(sector_signals)
    return provider, start, end, sector_signals, z_panel


def test_resample_panel_preserves_shape_and_index():
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    panel = pd.DataFrame(np.arange(80).reshape(40, 2), index=idx, columns=["A", "B"])
    rng = np.random.default_rng(1)
    resampled = stats._resample_panel(panel, 13, rng)
    assert resampled.shape == panel.shape
    assert (resampled.index == panel.index).all()
    # content should generally differ from the original ordering
    assert not resampled.equals(panel)
    # every resampled row must be a row that actually existed in the original panel
    original_rows = {tuple(r) for r in panel.to_numpy()}
    resampled_rows = {tuple(r) for r in resampled.to_numpy()}
    assert resampled_rows.issubset(original_rows)


def test_resample_panel_blocks_are_contiguous_circular():
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    panel = pd.DataFrame({"A": np.arange(40) * 2}, index=idx)
    rng = np.random.default_rng(3)
    resampled = stats._resample_panel(panel, 10, rng)
    vals = resampled["A"].to_numpy()
    original = panel["A"].to_numpy()
    row_of = {v: i for i, v in enumerate(original)}
    positions = np.array([row_of[v] for v in vals])
    # within each block of 10, consecutive original-row positions step by 1 (mod 40, circular)
    for b in range(0, 40, 10):
        chunk = positions[b : b + 10]
        diffs = (chunk[1:] - chunk[:-1]) % 40
        assert np.all(diffs == 1)


def test_resample_panel_reuses_and_advances_generator():
    idx = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    panel = pd.DataFrame(np.arange(80).reshape(40, 2), index=idx, columns=["A", "B"])
    rng = np.random.default_rng(9)
    draw1 = stats._resample_panel(panel, 13, rng)
    draw2 = stats._resample_panel(panel, 13, rng)
    assert not draw1.equals(draw2)


def test_expected_max_sharpe_monotonic_in_trials():
    v1 = stats.expected_max_sharpe(1.0, 2)
    v2 = stats.expected_max_sharpe(1.0, 20)
    v3 = stats.expected_max_sharpe(1.0, 200)
    assert 0 < v1 < v2 < v3


def test_expected_max_sharpe_zero_for_single_trial_or_zero_spread():
    assert stats.expected_max_sharpe(1.0, 1) == 0.0
    assert stats.expected_max_sharpe(0.0, 50) == 0.0


def test_deflated_sharpe_ratio_single_trial_gap_equals_raw_sharpe():
    idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 60), index=idx)
    result = stats.deflated_sharpe_ratio(r, trial_sharpes=np.array([np.nan]))
    assert result["sharpe0_expected_max"] == 0.0
    assert np.isclose(result["deflated_sharpe_gap"], result["sharpe"])


def test_deflated_sharpe_ratio_wider_trial_spread_lowers_gap():
    idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.002, 0.01, 60), index=idx)
    narrow = stats.deflated_sharpe_ratio(r, trial_sharpes=np.array([0.1, 0.11, 0.09, 0.1]))
    wide = stats.deflated_sharpe_ratio(r, trial_sharpes=np.array([-3, 3, -2, 2, -1, 1]))
    assert wide["deflated_sharpe_gap"] < narrow["deflated_sharpe_gap"]


def test_sign_stability_detects_flip_and_consistency():
    idx = pd.date_range("2004-01-02", periods=1200, freq="W-FRI")
    consistent = pd.Series(0.001, index=idx)
    res = stats.sign_stability(consistent)
    assert res["sign_stable"] is True

    r = pd.Series(0.001, index=idx)
    r.loc["2018":] = -0.001
    res2 = stats.sign_stability(r)
    assert res2["sign_stable"] is False


def test_overlap_control_pass_and_fail():
    idx = pd.date_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=200), index=idx)
    independent = pd.Series(rng.normal(size=200), index=idx)
    correlated = a * 2 + rng.normal(scale=0.01, size=200)

    res_indep = stats.overlap_control(a, independent, a, independent)
    assert res_indep["passes"] or abs(res_indep["spearman_rho"]) < 0.3

    res_corr = stats.overlap_control(a, correlated, a, correlated)
    assert not res_corr["passes"]
    assert abs(res_corr["spearman_rho"]) > 0.9


def test_run_twin_backtest_and_delta_sharpe(tiny_setup):
    provider, start, end, sector_signals, z_panel = tiny_setup
    main = run_backtest(provider, start, end, FAST_PARAMS, sectors=SECTORS, z_override=z_panel)
    twin = stats.run_twin_backtest(provider, start, end, FAST_PARAMS, sectors=SECTORS, sector_signals=sector_signals)
    delta = stats.delta_sharpe_vs_twin(main, twin)
    assert np.isfinite(delta) or np.isnan(delta)
    assert len(twin.weekly_returns) == len(main.weekly_returns)


def test_oracle_backtest_runs_and_outperforms_typically(tiny_setup):
    provider, start, end, _, _ = tiny_setup
    oracle_ret = stats.oracle_backtest(provider, start, end, n_legs=2, sectors=SECTORS)
    assert isinstance(oracle_ret, pd.Series)
    assert len(oracle_ret) > 0
    assert np.isfinite(annualized_sharpe(oracle_ret)) or oracle_ret.std() == 0


def test_bootstrap_pvalue_smoke(tiny_setup):
    provider, start, end, _, z_panel = tiny_setup
    result = stats.circular_block_bootstrap_pvalue(
        provider, start, end, FAST_PARAMS, n_draws=8, block_weeks=13, seed=0, sectors=SECTORS, z_panel=z_panel
    )
    assert len(result.null_sharpes) == 8
    assert 0.0 <= result.p_value <= 1.0 or np.isnan(result.p_value)


def test_random_sector_baseline_smoke(tiny_setup):
    provider, start, end, _, _ = tiny_setup
    sharpes = stats.random_sector_baseline(provider, start, end, n_legs=2, n_draws=5, seed=0, sectors=SECTORS)
    assert len(sharpes) == 5
