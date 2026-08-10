from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vridmomentet import stats
from vridmomentet.config import RejectionThresholds
from vridmomentet.data import Panel
from vridmomentet.universe import MembershipInterval, PointInTimeMembership


class TestBasicStats:
    def test_sharpe_of_zero_vol_series_is_zero_not_inf(self):
        r = pd.Series([0.01] * 10)
        assert stats.sharpe_ratio(r) == 0.0

    def test_sharpe_scales_with_sqrt_periods_per_year(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.02, size=200))
        sr_weekly = stats.sharpe_ratio(r, periods_per_year=52, annualize=False)
        sr_annual = stats.sharpe_ratio(r, periods_per_year=52, annualize=True)
        assert sr_annual == pytest.approx(sr_weekly * np.sqrt(52))

    def test_max_drawdown_is_non_positive(self):
        r = pd.Series([0.05, -0.10, 0.02, -0.03, 0.08])
        assert stats.max_drawdown(r) <= 0.0

    def test_summary_stats_keys(self):
        r = pd.Series(np.random.default_rng(1).normal(size=100))
        s = stats.summary_stats(r)
        for k in ["n_obs", "sharpe", "sortino", "max_drawdown", "annualized_return", "hit_rate"]:
            assert k in s


class TestDeflatedSharpeRatio:
    def test_clearly_standout_trial_gives_high_dsr(self):
        rng = np.random.default_rng(2)
        noise_trials = list(rng.normal(0, 0.05, size=29))
        trials = noise_trials + [0.35]  # one outlier trial (periodic Sharpe)
        result = stats.deflated_sharpe_ratio(observed_sharpe_per_period=0.35, trial_sharpes_per_period=trials, n_obs=500)
        assert result["dsr"] > 0.99

    def test_typical_of_the_pack_gives_low_dsr(self):
        rng = np.random.default_rng(3)
        trials = list(rng.normal(0, 0.05, size=30))
        observed = float(np.median(trials))
        result = stats.deflated_sharpe_ratio(observed_sharpe_per_period=observed, trial_sharpes_per_period=trials, n_obs=500)
        assert result["dsr"] < 0.9

    def test_requires_at_least_two_trials(self):
        with pytest.raises(ValueError):
            stats.expected_max_sharpe_under_null([0.1])


class TestBlockBootstrapSharpe:
    def test_independent_noise_gives_roughly_uniform_pvalues_on_average(self):
        """A single draw's p-value is itself noisy (by design -- a valid test
        rejects a true null ~5% of the time), so this checks calibration
        across many independent zero-mean draws instead of asserting a
        threshold on one: the *median* p-value under a true null should sit
        well above 0.05, not cluster near significance.
        """
        p_values = []
        for seed in range(15):
            rng = np.random.default_rng(seed)
            r = pd.Series(rng.normal(0, 0.02, size=300))
            result = stats.block_bootstrap_sharpe_pvalue(r, n_boot=200, block_size=8.0, seed=seed + 1000)
            p_values.append(result["p_value"])
        assert np.median(p_values) > 0.2

    def test_strong_genuine_drift_gives_low_p_value(self):
        rng = np.random.default_rng(6)
        r = pd.Series(rng.normal(0.02, 0.02, size=300))  # strong, real Sharpe ~1/week
        result = stats.block_bootstrap_sharpe_pvalue(r, n_boot=300, block_size=8.0, seed=7)
        assert result["p_value"] < 0.05

    def test_too_few_observations_returns_nan_not_crash(self):
        r = pd.Series([0.01, -0.02, 0.03])
        result = stats.block_bootstrap_sharpe_pvalue(r, n_boot=50)
        assert np.isnan(result["p_value"])


class TestBeatsAllTwins:
    def test_true_when_primary_sharpe_exceeds_every_twin(self):
        primary = pd.Series(np.random.default_rng(8).normal(0.01, 0.02, size=100))
        twins = {
            "a": pd.Series(np.random.default_rng(9).normal(0.0, 0.02, size=100)),
            "b": pd.Series(np.random.default_rng(10).normal(0.0, 0.03, size=100)),
        }
        result = stats.beats_all_twins(primary, twins)
        assert result["beats_all_twins"] is True

    def test_false_when_one_twin_wins(self):
        rng = np.random.default_rng(11)
        primary = pd.Series(rng.normal(0.0, 0.02, size=200))
        twins = {"strong": pd.Series(rng.normal(0.02, 0.01, size=200))}
        result = stats.beats_all_twins(primary, twins)
        assert result["beats_all_twins"] is False
        assert result["per_twin"]["strong"]["primary_beats_twin"] is False


class TestSignStability:
    def test_stable_when_all_subperiods_share_sign(self):
        idx = pd.date_range("2003-01-01", periods=600, freq="W-FRI")
        r = pd.Series(0.001, index=idx)  # constant positive
        subperiods = {
            "p1": (dt.date(2003, 1, 1), dt.date(2005, 12, 31)),
            "p2": (dt.date(2006, 1, 1), dt.date(2008, 12, 31)),
        }
        result = stats.sign_stability(r, subperiods)
        assert result["sign_stable"] is True

    def test_unstable_when_signs_flip(self):
        idx = pd.date_range("2003-01-01", periods=600, freq="W-FRI")
        r = pd.Series(0.001, index=idx)
        r.loc["2006":"2008"] = -0.001
        subperiods = {
            "p1": (dt.date(2003, 1, 1), dt.date(2005, 12, 31)),
            "p2": (dt.date(2006, 1, 1), dt.date(2008, 12, 31)),
        }
        result = stats.sign_stability(r, subperiods)
        assert result["sign_stable"] is False


class TestPnlConcentration:
    def test_evenly_spread_pnl_has_low_concentration(self):
        r = pd.Series([0.01] * 100)
        result = stats.max_pnl_concentration(r, window=8)
        assert result["share"] == pytest.approx(0.08, abs=0.01)

    def test_single_burst_dominates(self):
        r = pd.Series([0.0] * 100)
        r.iloc[50:58] = 0.05
        result = stats.max_pnl_concentration(r, window=8)
        assert result["share"] > 0.9


class TestEvaluateRejection:
    def _passing_shuffle(self, fraction=0.25):
        # Comfortably above the 5% nominal null rate / 10% shuffle_p_max
        # threshold -- reads as genuine excess order information.
        return {"by_block_size": {1: {"fraction_exceeding_95th_pct_null": fraction}, 4: {"fraction_exceeding_95th_pct_null": fraction}}}

    def test_all_criteria_pass_gives_no_rejection(self):
        dsr = {"z": 2.0}
        bootstrap = {"p_value": 0.01}
        twins = {"beats_all_twins": True}
        concentration = {"share": 0.1}
        sign = {"sign_stable": True}
        result = stats.evaluate_rejection(dsr, bootstrap, self._passing_shuffle(), twins, concentration, sign, pead_delta_sharpe=0.5)
        assert result["reject"] is False

    def test_single_failing_criterion_triggers_rejection(self):
        dsr = {"z": -1.0}  # fails
        bootstrap = {"p_value": 0.01}
        twins = {"beats_all_twins": True}
        concentration = {"share": 0.1}
        sign = {"sign_stable": True}
        result = stats.evaluate_rejection(dsr, bootstrap, self._passing_shuffle(), twins, concentration, sign, pead_delta_sharpe=0.5)
        assert result["reject"] is True
        assert result["reasons"]["dsr_oos_z_leq_threshold"] is True
        assert result["reasons"]["bootstrap_p_geq_threshold"] is False

    def test_shuffle_null_at_nominal_rate_triggers_rejection(self):
        """The brief's own primary pre-registered null (Huvudnull) must
        actually gate the verdict: a shuffle-null exceedance fraction at or
        below the nominal ~5% false-positive rate (i.e. no detectable
        excess order information) has to be wired into `reasons`, not just
        reported and ignored.
        """
        dsr = {"z": 2.0}
        bootstrap = {"p_value": 0.01}
        twins = {"beats_all_twins": True}
        concentration = {"share": 0.1}
        sign = {"sign_stable": True}
        no_signal_shuffle = {"by_block_size": {1: {"fraction_exceeding_95th_pct_null": 0.05}}}
        result = stats.evaluate_rejection(dsr, bootstrap, no_signal_shuffle, twins, concentration, sign, pead_delta_sharpe=0.5)
        assert result["reject"] is True
        assert result["reasons"]["shuffle_null_shows_no_excess_order_information"] is True

    def test_missing_shuffle_data_is_treated_as_a_failure_not_silently_ignored(self):
        dsr = {"z": 2.0}
        bootstrap = {"p_value": 0.01}
        twins = {"beats_all_twins": True}
        concentration = {"share": 0.1}
        sign = {"sign_stable": True}
        result = stats.evaluate_rejection(dsr, bootstrap, {}, twins, concentration, sign, pead_delta_sharpe=0.5)
        assert result["reject"] is True
        assert result["reasons"]["shuffle_null_shows_no_excess_order_information"] is True


def _synthetic_panel(n_days=250, n_names=10, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    names = [f"T{i}" for i in range(n_names)]
    close = pd.DataFrame({n: 100 * np.cumprod(1 + rng.normal(scale=0.01, size=n_days)) for n in names}, index=dates)
    volume = pd.DataFrame({n: rng.uniform(1e6, 2e6, size=n_days) for n in names}, index=dates)
    start, end = dates.min().date(), dates.max().date() + dt.timedelta(days=1)
    membership = PointInTimeMembership(
        [MembershipInterval(ticker=t, source="sp500", start=start, end=end, point_in_time=True) for t in names]
    )
    return Panel(close=close, adj_close=close.copy(), adj_open=close.copy(), volume=volume, membership=membership)


class TestShuffleNullCheck:
    def test_runs_and_returns_expected_shape(self):
        panel = _synthetic_panel()
        result = stats.shuffle_null_check(panel, window=20, n_windows_sample=15, n_reps=50, block_sizes=(1, 4), seed=0)
        assert set(result["by_block_size"].keys()) == {1, 4}
        for block_size, res in result["by_block_size"].items():
            assert 0 <= res["fraction_exceeding_95th_pct_null"] <= 1 or np.isnan(res["fraction_exceeding_95th_pct_null"])

    def test_pure_noise_gives_roughly_nominal_false_positive_rate(self):
        """On pure iid noise (no genuine order information), the fraction of
        sampled windows whose real |A| exceeds its own shuffle-null's 95th
        percentile should be in the right ballpark of 5% -- not exactly 5%
        (small sample), but nowhere near systematically elevated.
        """
        panel = _synthetic_panel(n_days=400, n_names=20, seed=42)
        result = stats.shuffle_null_check(panel, window=20, n_windows_sample=60, n_reps=200, block_sizes=(1,), seed=1)
        frac = result["by_block_size"][1]["fraction_exceeding_95th_pct_null"]
        assert frac < 0.30  # generous upper bound; true rate should be close to 0.05


def _synthetic_panel_with_volume_spike(spike_idx: int, spike_ticker: str = "T0", n_days=200, n_names=3, seed=5) -> tuple[Panel, pd.Timestamp]:
    """Panel derives dollar_volume/adv/vol60 once at construction time, so a
    spike must be baked into the `volume` frame *before* building the
    Panel -- mutating panel.volume post-construction does not propagate to
    its already-computed dollar_volume snapshot.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    names = [f"T{i}" for i in range(n_names)]
    close = pd.DataFrame({n: 100 * np.cumprod(1 + rng.normal(scale=0.01, size=n_days)) for n in names}, index=dates)
    volume = pd.DataFrame({n: rng.uniform(1e6, 2e6, size=n_days) for n in names}, index=dates)
    spike_date = dates[spike_idx]
    volume.loc[spike_date, spike_ticker] *= 50  # a massive, obvious one-day spike
    start, end = dates.min().date(), dates.max().date() + dt.timedelta(days=1)
    membership = PointInTimeMembership(
        [MembershipInterval(ticker=t, source="sp500", start=start, end=end, point_in_time=True) for t in names]
    )
    panel = Panel(close=close, adj_close=close.copy(), adj_open=close.copy(), volume=volume, membership=membership)
    return panel, spike_date


class TestPeadExclusionMask:
    def test_flags_window_containing_a_volume_spike(self):
        panel, spike_date = _synthetic_panel_with_volume_spike(150)
        mask = stats.pead_exclusion_mask(panel, window=20, spike_lookback=100, spike_z_threshold=4.0)
        # The spike day itself, and the ~20 days after it (whose trailing
        # window still contains the spike), should be flagged for T0.
        assert bool(mask.loc[spike_date, "T0"])
        assert bool(mask.loc[panel.dates[155], "T0"])

    def test_does_not_flag_unrelated_names(self):
        panel, spike_date = _synthetic_panel_with_volume_spike(150)
        mask = stats.pead_exclusion_mask(panel, window=20, spike_lookback=100, spike_z_threshold=4.0)
        assert not bool(mask.loc[spike_date, "T1"])

    def test_apply_exclusion_mask_sets_flagged_entries_to_nan(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        s = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        mask = pd.DataFrame({"A": [False, False, True, False, False], "B": [False] * 5}, index=idx)
        out = stats.apply_exclusion_mask(s, mask)
        assert np.isnan(out.loc[idx[2], "A"])
        assert out.loc[idx[2], "B"] == 3.0


class TestDiversificationStats:
    def test_beta_one_when_strategy_equals_benchmark(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="W")
        rng = np.random.default_rng(9)
        bench = pd.Series(rng.normal(0, 0.02, size=100), index=idx)
        result = stats.diversification_stats(bench, bench, None)
        assert result["beta_to_benchmark"] == pytest.approx(1.0, abs=1e-6)
        assert result["corr_to_tidspilen"] is None

    def test_zero_beta_when_uncorrelated(self):
        idx = pd.date_range("2020-01-01", periods=2000, freq="D")
        rng = np.random.default_rng(10)
        bench = pd.Series(rng.normal(size=2000), index=idx)
        strat = pd.Series(rng.normal(size=2000), index=idx)  # independent draw
        result = stats.diversification_stats(strat, bench, None)
        assert abs(result["beta_to_benchmark"]) < 0.15
