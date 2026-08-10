import numpy as np
import pandas as pd
import pytest

from cepstral_metaorder import validation as val
from cepstral_metaorder import signal as sig
from cepstral_metaorder.synthetic import make_symbol_dense
from cepstral_metaorder.config import SPEC


def test_existence_permutation_test_calibrated_near_nominal_on_pure_noise():
    rng = np.random.default_rng(0)
    dense = make_symbol_dense(rng, n_days=30, inject_tau=None)
    vw = sig.volume_wide(dense)
    uw = sig.detrend(vw)
    result = val.existence_permutation_test({"NOISE": uw}, n_perm=100, sample_size=9, seed=1)
    # pure noise: real exceedance should be in the right ballpark of nominal 5%,
    # nowhere near the >=15% bar that would indicate a real phenomenon
    assert result["exceedance_fraction"] < result["required_fraction"]
    assert result["passed"] is False


def test_existence_permutation_test_detects_injected_periodicity():
    rng = np.random.default_rng(2)
    dense = make_symbol_dense(rng, n_days=40, inject_tau=13, inject_from_day=21,
                               inject_strength=2.2, direction_bias=1.0)
    vw = sig.volume_wide(dense)
    uw = sig.detrend(vw)
    # restrict to the injected days only, where the burst is actually present
    uw_injected = uw.loc[uw.index[21:]]
    result = val.existence_permutation_test({"TARGET": uw_injected}, n_perm=100, sample_size=15, seed=3)
    assert result["exceedance_fraction"] > result["required_fraction"]
    assert result["passed"] is True


def test_pc1_share_high_for_one_common_factor():
    rng = np.random.default_rng(4)
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    common = rng.normal(size=60)
    panel = pd.DataFrame({f"S{i}": common + rng.normal(0, 0.01, 60) for i in range(20)}, index=dates)
    assert val.pc1_share(panel) > 0.9


def test_pc1_share_low_for_independent_names():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    panel = pd.DataFrame({f"S{i}": rng.normal(size=200) for i in range(30)}, index=dates)
    assert val.pc1_share(panel) < SPEC.validation.pc1_share_max


def test_fama_macbeth_detects_true_incremental_signal():
    rng = np.random.default_rng(6)
    rows = []
    for day in range(120):
        n = 40
        z = rng.normal(size=n)
        control = rng.normal(size=n)
        y = 0.02 * z + 0.0 * control + rng.normal(0, 0.05, n)  # y truly depends on z, not on control
        for i in range(n):
            rows.append({"date": day, "y": y[i], "Z": z[i], "control": control[i]})
    panel = pd.DataFrame(rows)
    result = val.fama_macbeth_screen(panel, y_col="y", z_col="Z", control_cols=["control"], nw_lags=3)
    assert result["passed"] is True
    assert result["nw"]["t_stat"] > SPEC.validation.min_abs_newey_west_t


def test_fama_macbeth_rejects_when_z_has_no_true_incremental_power():
    rng = np.random.default_rng(7)
    rows = []
    for day in range(120):
        n = 40
        z = rng.normal(size=n)          # pure noise, unrelated to y
        control = rng.normal(size=n)
        y = 0.02 * control + rng.normal(0, 0.05, n)
        for i in range(n):
            rows.append({"date": day, "y": y[i], "Z": z[i], "control": control[i]})
    panel = pd.DataFrame(rows)
    result = val.fama_macbeth_screen(panel, y_col="y", z_col="Z", control_cols=["control"], nw_lags=3)
    assert result["passed"] is False
    assert abs(result["nw"]["t_stat"]) < SPEC.validation.min_abs_newey_west_t


def test_sign_consistency_passes_with_one_bad_subperiod_out_of_four():
    rng = np.random.default_rng(8)
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    rows = []
    for i, d in enumerate(dates):
        chunk = i // 50  # 4 chunks of 50 days
        n = 30
        direction = rng.choice([-1.0, 1.0], size=n)
        sign_for_chunk = -1.0 if chunk == 0 else 1.0  # subperiod 0 is inconsistent, 1-3 are consistent
        y = sign_for_chunk * direction * 0.01 + rng.normal(0, 0.02, n)
        for j in range(n):
            rows.append({"date": d, "D": direction[j], "y": y[j]})
    panel = pd.DataFrame(rows)
    result = val.sign_consistency_by_subperiod(panel, direction_col="D", y_col="y", n_subperiods=4)
    assert result["n_missing_consistency"] == 1
    assert result["passed"] is True


def test_sign_consistency_fails_with_two_bad_subperiods_out_of_four():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2024-01-02", periods=200, freq="B")
    rows = []
    for i, d in enumerate(dates):
        chunk = i // 50
        n = 30
        direction = rng.choice([-1.0, 1.0], size=n)
        sign_for_chunk = -1.0 if chunk in (0, 1) else 1.0  # two bad subperiods
        y = sign_for_chunk * direction * 0.01 + rng.normal(0, 0.02, n)
        for j in range(n):
            rows.append({"date": d, "D": direction[j], "y": y[j]})
    panel = pd.DataFrame(rows)
    result = val.sign_consistency_by_subperiod(panel, direction_col="D", y_col="y", n_subperiods=4)
    assert result["n_missing_consistency"] == 2
    assert result["passed"] is False


def test_twin_horse_race_ignores_dead_twins():
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    main = pd.Series(np.random.default_rng(10).normal(0.001, 0.01, 100), index=dates)
    # a "twin" that would appear beaten (lower Sharpe) but fails liveness -- must not count
    dead_twin = pd.Series(np.random.default_rng(11).normal(-0.01, 0.01, 100), index=dates)
    # a live twin that actually beats main
    strong_twin = pd.Series(np.random.default_rng(12).normal(0.01, 0.005, 100), index=dates)

    liveness = {"dead": {"alive": False}, "strong": {"alive": True}}
    result = val.twin_horse_race(main, {"dead": dead_twin, "strong": strong_twin}, liveness)
    assert result["passed"] is False  # loses to the live "strong" twin
    assert result["twins"]["dead"]["beaten_by_main"] is True  # main did beat it, but irrelevant
    assert result["twins"]["strong"]["beaten_by_main"] is False


def test_verdict_short_circuits_on_step0_failure():
    step0_existence = {"passed": False}
    step0_breadth = {"passed": True}
    v = val.verdict(step0_existence, step0_breadth, step1={"passed": True}, step2={"anything": True})
    assert v["rejected"] is True
    assert v["stage_reached"] == "step0"


def test_verdict_short_circuits_on_step1_failure_before_step2():
    step0_existence = {"passed": True}
    step0_breadth = {"passed": True}
    step1 = {"passed": False, "nw": {"t_stat": 0.5}}
    v = val.verdict(step0_existence, step0_breadth, step1, step2=None)
    assert v["rejected"] is True
    assert v["stage_reached"] == "step1"


def test_block_shuffle_signal_preserves_each_symbols_own_value_set():
    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"S_bar": rng.normal(size=40), "D": rng.normal(size=40)}, index=dates)
    shuffled = val.block_shuffle_signal({"A": frame}, block_size=5, seed=1)["A"]
    assert len(shuffled) == len(frame)
    # same multiset of values, just reordered in blocks -- not literally
    # the same day-to-day pairing
    assert sorted(shuffled["S_bar"].values) == pytest.approx(sorted(frame["S_bar"].values))
    assert not shuffled["S_bar"].reset_index(drop=True).equals(frame["S_bar"].reset_index(drop=True))


def test_block_shuffle_signal_uses_different_seeds_to_get_different_orders():
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"S_bar": rng.normal(size=60), "D": rng.normal(size=60)}, index=dates)
    a = val.block_shuffle_signal({"A": frame}, block_size=5, seed=10)["A"]
    b = val.block_shuffle_signal({"A": frame}, block_size=5, seed=20)["A"]
    assert not a["S_bar"].reset_index(drop=True).equals(b["S_bar"].reset_index(drop=True))


def test_random_matched_signal_samples_direction_magnitude_from_real_distribution():
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    real_d = np.array([0.5] * 50 + [0.9] * 50)  # only two possible magnitudes in the real signal
    random_sig = val.random_matched_signal(["A", "B"], list(dates), real_d, seed=3)
    for sym in ["A", "B"]:
        mags = random_sig[sym]["D"].abs().round(6).unique()
        assert set(mags) <= {0.5, 0.9}
        # both signs should appear given 100 draws
        assert (random_sig[sym]["D"] > 0).any() and (random_sig[sym]["D"] < 0).any()
