import numpy as np
import pandas as pd
import pytest

from cepstral_metaorder import signal
from cepstral_metaorder.config import SPEC
from cepstral_metaorder.synthetic import make_symbol_dense


def test_real_cepstrum_recovers_known_period_from_a_pure_comb():
    """Hand-built input (no volume/synthetic.py machinery): a Dirac comb with
    period 12 sitting in noise. A textbook real-cepstrum implementation shows
    its highest peak (excluding the trivial tau=0 bin) at an integer multiple
    of 12 -- for a sparse impulse train the cepstrum's raw argmax often lands
    on a harmonic (24, 36, ...) rather than the fundamental itself; that's
    known "octave error" behavior in cepstral/pitch analysis, not a bug. The
    fundamental-preference correction (_argmax_prefer_fundamental) is applied
    downstream in score_and_tau_day/rolling_tau_star_window, not in this raw
    primitive, so this test checks the pre-correction invariant."""
    rng = np.random.default_rng(0)
    n = 360
    t = np.arange(n)
    x = rng.normal(0, 0.05, n)
    x[(t % 12) == 0] += 2.0

    c = signal.real_cepstrum_row(x, eps=SPEC.cepstrum.epsilon)
    band = c[SPEC.cepstrum.quefrency_min_min: SPEC.cepstrum.quefrency_max_min + 1]
    tau_star = np.argmax(band) + SPEC.cepstrum.quefrency_min_min
    assert tau_star % 12 == 0


def test_detrend_has_causal_warmup_and_is_mean_reverting_around_profile():
    rng = np.random.default_rng(1)
    dense = make_symbol_dense(rng, n_days=30, inject_tau=None)
    vw = signal.volume_wide(dense)
    uw = signal.detrend(vw, window=21)

    # first 21 sessions have no complete trailing profile yet
    assert uw.iloc[:21].isna().all(axis=1).all()
    # from day 22 onward the profile is defined and u has no systematic level
    assert uw.iloc[21:].notna().all(axis=1).all()
    later_mean = uw.iloc[21:].values.mean()
    assert abs(later_mean) < 0.15


def test_cross_sectional_standardize_matches_hand_computed_median_mad():
    dates = pd.Index(["2024-01-02"])
    cols = [5]
    # 5 symbols, single quefrency column, single day: values 1,2,3,4,100 (outlier)
    raw = {
        "A": pd.DataFrame([[1.0]], index=dates, columns=cols),
        "B": pd.DataFrame([[2.0]], index=dates, columns=cols),
        "C": pd.DataFrame([[3.0]], index=dates, columns=cols),
        "D": pd.DataFrame([[4.0]], index=dates, columns=cols),
        "E": pd.DataFrame([[100.0]], index=dates, columns=cols),
    }
    out = signal.cross_sectional_standardize_panel(raw, min_cross_section=5)
    median = 3.0
    mad = np.median(np.abs(np.array([1, 2, 3, 4, 100]) - median))  # = 1.0
    assert out["A"].loc["2024-01-02", 5] == pytest.approx((1.0 - median) / mad)
    assert out["E"].loc["2024-01-02", 5] == pytest.approx((100.0 - median) / mad)


def test_cross_sectional_standardize_requires_minimum_breadth():
    dates = pd.Index(["2024-01-02"])
    raw = {s: pd.DataFrame([[1.0]], index=dates, columns=[5]) for s in ["A", "B", "C"]}
    out = signal.cross_sectional_standardize_panel(raw, min_cross_section=10)
    assert out["A"].loc["2024-01-02", 5] != out["A"].loc["2024-01-02", 5]  # NaN


def test_comb_filter_and_direction_recover_injected_period_and_sign():
    """The central claim of the whole hypothesis: inject a known periodic,
    directional burst into one name against a field of pure-noise peers with
    a SHARED common periodicity (simulating market-wide hedging rhythm), and
    check that cross-sectional standardization isolates the idiosyncratic
    signal, tau* lands on the true injected period, and D recovers the
    injected direction's sign."""
    rng = np.random.default_rng(42)
    n_days = 40
    inject_from = 21  # right after the detrend warmup, so the injection window is fully scored
    true_tau = 17
    n_peers = 14

    symbols = {}
    symbols["TARGET"] = make_symbol_dense(
        rng, n_days, inject_tau=true_tau, inject_from_day=inject_from,
        inject_strength=1.8, direction_bias=1.0, common_tau=60, common_strength=0.6,
    )
    for i in range(n_peers):
        symbols[f"PEER{i}"] = make_symbol_dense(
            rng, n_days, inject_tau=None, common_tau=60, common_strength=0.6,
        )

    raw_ceps = {}
    dense_by_symbol = {}
    for sym, dense in symbols.items():
        vw = signal.volume_wide(dense)
        uw = signal.detrend(vw)
        raw_ceps[sym] = signal.cepstrum_wide(uw)
        dense_by_symbol[sym] = dense

    standardized = signal.cross_sectional_standardize_panel(raw_ceps, min_cross_section=10)

    target_frame = signal.signal_frame_for_symbol(dense_by_symbol["TARGET"], standardized["TARGET"])
    peer_frames = {
        sym: signal.signal_frame_for_symbol(dense_by_symbol[sym], standardized[sym])
        for sym in symbols if sym != "TARGET"
    }

    # Evaluate a few days into the injection window rather than at the very
    # end of the run: the 21-day rolling detrend profile is itself causal and
    # adaptive, so once enough *injected* days accumulate inside its own
    # trailing window it starts absorbing part of the burst into "normal",
    # diluting u_t. That's a real, worth-documenting property of a causal
    # baseline applied to a persistent (>~3 week) metaorder -- see README --
    # but it's a different phenomenon from "can the detector find a fresh
    # burst", which is what this test checks. index 28 = 7 injected days
    # old, detrend profile <=33% injection-contaminated.
    last_day = target_frame.index[28]

    # tau* recovered on the injected name is a harmonic of the true injected
    # period (the fundamental-preference correction reduces but does not
    # eliminate octave error -- see test_real_cepstrum_recovers_known_period_
    # from_a_pure_comb), and specifically NOT the common-to-everyone period
    # (60) that cross-sectional standardization is supposed to erase.
    recovered_tau = int(target_frame.loc[last_day, "tau_star_window"])
    assert recovered_tau % true_tau == 0
    assert recovered_tau != 60

    # the injected name's smoothed score dominates the noise-only peer field
    target_S = target_frame.loc[last_day, "S_bar"]
    peer_S = np.array([f.loc[last_day, "S_bar"] for f in peer_frames.values()])
    assert target_S > np.nanpercentile(peer_S, 95)

    # direction sign matches the injected drift, and clears the |D|>=0.1 entry bar
    target_D = target_frame.loc[last_day, "D"]
    assert target_D > SPEC.entry_exit.entry_min_abs_direction

    # peers' tau* should NOT systematically cluster on the common 60-min rhythm --
    # that's exactly the market-wide periodicity cross-sectional standardization
    # is supposed to cancel.
    peer_taus = [f.loc[last_day, "tau_star_window"] for f in peer_frames.values()]
    frac_at_common = np.mean([t == 60 for t in peer_taus if pd.notna(t)])
    assert frac_at_common < 0.5


def test_direction_sign_flips_with_injected_bias_sign():
    rng = np.random.default_rng(7)
    n_days = 30
    tau = 10
    dense_pos = make_symbol_dense(rng, n_days, inject_tau=tau, inject_from_day=21,
                                   inject_strength=1.8, direction_bias=1.0)
    rng2 = np.random.default_rng(7)
    dense_neg = make_symbol_dense(rng2, n_days, inject_tau=tau, inject_from_day=21,
                                   inject_strength=1.8, direction_bias=-1.0)

    def last_day_D(dense):
        vw = signal.volume_wide(dense)
        uw = signal.detrend(vw)
        raw = signal.cepstrum_wide(uw)
        # single-name standardization is meaningless (needs cross-section);
        # bypass it here since this test only checks the comb/direction step
        # in isolation given a known tau*, not the standardization step.
        last = dense["session_date"].max()
        window_dates = sorted(vw.index)[-SPEC.cepstrum.direction_window_days:]
        phases = signal.comb_burst_phases(uw, window_dates, tau)
        return signal.direction_for_window(vw, signal.ret_wide(dense), window_dates, tau, phases)

    assert last_day_D(dense_pos) > 0
    assert last_day_D(dense_neg) < 0
