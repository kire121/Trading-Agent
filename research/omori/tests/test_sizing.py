import numpy as np

from research.omori import config, sizing


class TestRawTargetWeight:
    def test_full_tilt_when_p_tilde_at_or_below_p_star(self):
        w = sizing.raw_target_weight(1.0, p_tilde=0.5, p_star=0.6, sigma_hat_daily=0.01,
                                      vol_target=0.004)
        assert w == 0.4  # tilt=1.0 * (0.004/0.01)

    def test_tilt_shrinks_as_p_tilde_exceeds_p_star(self):
        w_at_star = sizing.raw_target_weight(1.0, 0.6, 0.6, 0.01, 0.004)
        w_beyond = sizing.raw_target_weight(1.0, 1.2, 0.6, 0.01, 0.004)
        assert abs(w_beyond) < abs(w_at_star)

    def test_direction_flips_sign(self):
        w_long = sizing.raw_target_weight(1.0, 0.5, 0.6, 0.01, 0.004)
        w_short = sizing.raw_target_weight(-1.0, 0.5, 0.6, 0.01, 0.004)
        assert w_long == -w_short

    def test_invalid_sigma_returns_zero(self):
        assert sizing.raw_target_weight(1.0, 0.5, 0.6, 0.0, 0.004) == 0.0
        assert sizing.raw_target_weight(1.0, 0.5, 0.6, np.nan, 0.004) == 0.0


class TestGrossCap:
    def test_no_scaling_when_under_cap(self):
        w = np.array([0.3, -0.2, 0.1])
        out = sizing.apply_gross_cap(w, gross_cap=1.5)
        np.testing.assert_array_equal(out, w)

    def test_scales_down_proportionally_when_over_cap(self):
        w = np.array([1.0, -1.0, 1.0])  # gross = 3.0
        out = sizing.apply_gross_cap(w, gross_cap=1.5)
        np.testing.assert_allclose(np.abs(out).sum(), 1.5)
        # direction/relative proportions preserved
        np.testing.assert_allclose(out / w, np.full(3, 0.5))

    def test_never_scales_up(self):
        w = np.array([0.1, -0.1])
        out = sizing.apply_gross_cap(w, gross_cap=1.5)
        np.testing.assert_array_equal(out, w)
