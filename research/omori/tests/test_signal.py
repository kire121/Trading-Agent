import numpy as np
import pytest

from research.omori import config, signal


def synthetic_e_path(p, c, k=1.0, tau_max=20, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    s = np.arange(1, tau_max + 1, dtype=float)
    e = k * (s + c) ** (-p)
    if noise:
        e = e * (1.0 + rng.normal(0, noise, size=len(e)))
    return e


class TestFitOmori:
    def test_recovers_known_p_and_c_noiseless(self):
        for true_p, true_c in [(0.8, 0.0), (1.2, 1.0), (0.5, 2.0)]:
            e = synthetic_e_path(true_p, true_c, tau_max=15)
            fit = signal.fit_omori(15, e)
            assert fit.identified
            assert fit.p_hat == pytest.approx(true_p, abs=1e-6)
            assert fit.c_hat == true_c

    def test_recovers_p_approximately_under_noise(self):
        e = synthetic_e_path(0.9, 1.0, tau_max=20, noise=0.05, seed=1)
        fit = signal.fit_omori(20, e)
        assert fit.identified
        assert fit.p_hat == pytest.approx(0.9, abs=0.15)

    def test_full_shrinkage_below_min_positive_days(self):
        e = np.array([0.5, 0.3, -0.1])  # only 2 positive-excess days
        fit = signal.fit_omori(3, e)
        assert not fit.identified
        assert fit.n_pos == 2
        assert np.isnan(fit.p_hat)

    def test_unidentified_when_excess_not_decaying(self):
        # monotonically INCREASING excess volume -> no valid decaying (p>0)
        # fit under any candidate c; must surface as unidentified, not a
        # silently wrong negative-p number.
        e = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fit = signal.fit_omori(6, e)
        assert not fit.identified

    def test_flat_zero_excess_is_unidentified(self):
        e = np.zeros(10)
        fit = signal.fit_omori(10, e)
        assert not fit.identified
        assert fit.n_pos == 0

    def test_causality_only_uses_available_tau(self):
        e = synthetic_e_path(0.8, 1.0, tau_max=20)
        fit_partial = signal.fit_omori(6, e)
        fit_full = signal.fit_omori(20, e)
        assert fit_partial.n_pos <= fit_full.n_pos


class TestShrink:
    def test_full_shrinkage_returns_prior(self):
        fit = signal.FitResult(p_hat=np.nan, k_hat=np.nan, c_hat=np.nan, n_pos=0, identified=False)
        assert signal.shrink(fit, prior_p=0.7, kappa=5.0) == 0.7

    def test_large_n_e_dominates_over_prior(self):
        fit = signal.FitResult(p_hat=2.0, k_hat=1.0, c_hat=0.0, n_pos=1000, identified=True)
        p_tilde = signal.shrink(fit, prior_p=0.5, kappa=5.0)
        assert p_tilde == pytest.approx(2.0, abs=0.01)

    def test_zero_n_e_but_identified_gives_pure_prior_weighted_kappa(self):
        fit = signal.FitResult(p_hat=2.0, k_hat=1.0, c_hat=0.0, n_pos=0, identified=True)
        p_tilde = signal.shrink(fit, prior_p=0.5, kappa=5.0)
        assert p_tilde == pytest.approx(0.5)

    def test_shrinkage_formula_exact(self):
        fit = signal.FitResult(p_hat=1.5, k_hat=1.0, c_hat=0.0, n_pos=10, identified=True)
        p_tilde = signal.shrink(fit, prior_p=0.6, kappa=4.0)
        expected = (10 * 1.5 + 4.0 * 0.6) / (10 + 4.0)
        assert p_tilde == pytest.approx(expected)


class TestTauExit:
    def test_self_consistent_with_model_equation(self):
        """Plugging tau_exit back into e(tau)=K(tau+c)^-p should reproduce
        theta * e(1) exactly (the derivation's own self-consistency)."""
        for p, c, theta in [(0.8, 0.0, 0.25), (1.2, 2.0, 0.15), (0.5, 1.0, 0.35)]:
            tex = signal.tau_exit(c, p, theta=theta, floor=0, cap=1000)
            e_at_1 = (1 + c) ** (-p)
            e_at_tex = (tex + c) ** (-p)
            assert e_at_tex / e_at_1 == pytest.approx(theta, rel=1e-6)

    def test_floor_and_cap_clip(self):
        assert signal.tau_exit(0.0, 100.0, theta=0.25, floor=3, cap=20) == 3
        assert signal.tau_exit(0.0, 0.001, theta=0.25, floor=3, cap=20) == 20

    def test_falls_back_to_cap_when_undefined(self):
        assert signal.tau_exit(np.nan, 0.5, cap=20) == 20
        assert signal.tau_exit(0.0, np.nan, cap=20) == 20


class TestSignal:
    def test_g_of_p_capped_at_one(self):
        assert signal.g_of_p(0.1, p_star=0.6) == 1.0

    def test_g_of_p_decreasing_in_p_tilde(self):
        vals = [signal.g_of_p(p, p_star=0.6) for p in [0.3, 0.6, 1.0, 2.0]]
        assert vals == sorted(vals, reverse=True)

    def test_traded_signal_sign_matches_direction(self):
        assert signal.traded_signal(1.0, 0.5, 0.6) > 0
        assert signal.traded_signal(-1.0, 0.5, 0.6) < 0
