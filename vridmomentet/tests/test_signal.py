"""Validation of the core Levy-area estimator against known-answer and
exact-identity cases -- the same role irreversibility_lab/tests/test_estimators.py
plays for its irreversibility estimators (synthetic processes with a
*provably known* property, not just "looks reasonable").
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from vridmomentet.signal import (
    cross_sectional_zscore,
    direction_factor,
    formation_return,
    levy_area_of_windows,
    rolling_levy_area,
    signed_dollar_volume,
)


def _shoelace_raw(p_tilde: np.ndarray, v_tilde: np.ndarray) -> float:
    """Shoelace formula applied literally to the raw (non-anchored) z-scored
    series, exactly as written in the brief: A = 1/2 sum(P~_{s-1} dV~_s -
    V~_{s-1} dP~_s). Used only to demonstrate the boundary-term gap vs. the
    anchored/double-sum form -- not the production estimator.
    """
    dp = np.diff(p_tilde)
    dv = np.diff(v_tilde)
    return 0.5 * float(np.sum(p_tilde[:-1] * dv - v_tilde[:-1] * dp))


def _double_sum_raw(p_tilde: np.ndarray, v_tilde: np.ndarray) -> float:
    """A = 1/2 sum_{s'<s} (dP~_{s'} dV~_s - dP~_s dV~_{s'}), the brief's
    "identity that forces honesty", evaluated directly as a double sum
    (O(n^2), only used to validate the O(n) production formula).
    """
    dp = np.diff(p_tilde)
    dv = np.diff(v_tilde)
    n = len(dp)
    total = 0.0
    for sp in range(n):
        for s in range(sp + 1, n):
            total += dp[sp] * dv[s] - dp[s] * dv[sp]
    return 0.5 * total


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


class TestShoelaceVsDoubleSumIdentity:
    def test_boundary_term_gap_for_unanchored_series(self):
        """Confirms the exact relationship derived in signal.py's module
        docstring: shoelace(raw z-scored path) = double_sum(path) +
        1/2 * (P~_0 * V~_n - V~_0 * P~_n). This is *not* generally zero,
        which is exactly why the production code anchors the path first.
        """
        rng = np.random.default_rng(0)
        r = rng.normal(size=20)
        u = rng.normal(size=20)
        p_tilde = _zscore(np.cumsum(r))
        v_tilde = _zscore(np.cumsum(u))

        shoelace = _shoelace_raw(p_tilde, v_tilde)
        double_sum = _double_sum_raw(p_tilde, v_tilde)
        boundary_term = 0.5 * (p_tilde[0] * v_tilde[-1] - v_tilde[0] * p_tilde[-1])

        assert shoelace == pytest.approx(double_sum + boundary_term, abs=1e-9)
        # And, for a generic random window, that boundary term is not
        # negligible relative to the double-sum value itself -- anchoring
        # is a substantive choice, not a rounding nicety.
        assert abs(boundary_term) > 1e-6

    def test_anchored_shoelace_equals_double_sum_exactly(self):
        """The identity the brief states literally holds once the window is
        anchored at its own first point (P^ = P~ - P~_0 etc.) -- this is
        what levy_area_of_windows actually computes.
        """
        rng = np.random.default_rng(1)
        for _ in range(20):
            n = rng.integers(5, 30)
            r = rng.normal(size=n)
            u = rng.normal(size=n)
            p_tilde = _zscore(np.cumsum(r))
            v_tilde = _zscore(np.cumsum(u))
            p_hat = p_tilde - p_tilde[0]
            v_hat = v_tilde - v_tilde[0]

            anchored_shoelace = _shoelace_raw(p_hat, v_hat)
            double_sum = _double_sum_raw(p_tilde, v_tilde)  # increments are shift-invariant
            production = float(levy_area_of_windows(r[None, :], u[None, :])[0])

            assert anchored_shoelace == pytest.approx(double_sum, abs=1e-9)
            assert production == pytest.approx(double_sum, abs=1e-9)


class TestTimeReversalAntisymmetry:
    def test_raw_unnormalized_shoelace_flips_sign_and_magnitude_exactly_under_reversal(self):
        """W(r, u) := shoelace(cumsum(r), cumsum(u)), with NO z-scoring, flips
        sign (and matches magnitude) exactly under day-order reversal, for
        any (r, u) -- checked here to floating-point precision. This is the
        unnormalized object the brief's "identity that forces honesty" is
        about; see signal.py's module docstring for why normalizing by the
        window's own std (the actual production step) weakens this from an
        exact identity to an approximate one.
        """
        rng = np.random.default_rng(2)
        for _ in range(20):
            n = rng.integers(4, 30)
            r = rng.normal(size=n)
            u = rng.normal(size=n)
            w_fwd = _shoelace_raw(np.cumsum(r), np.cumsum(u))
            w_rev = _shoelace_raw(np.cumsum(r[::-1]), np.cumsum(u[::-1]))
            assert w_rev == pytest.approx(-w_fwd, abs=1e-9)

    def test_production_area_sign_flips_exactly_but_magnitude_only_approximately(self):
        """For the actual (within-window z-scored, anchored) production
        signal A = B / (std(P) * std(V)), reversal flips the SIGN exactly
        (both denominators are always positive, so the sign is inherited
        directly from B's exact sign flip) -- but NOT the magnitude, because
        std(P) is itself a mildly order-dependent statistic of the
        cumulative path (std(cumsum(r)) != std(cumsum(reverse(r))) in
        general). Both halves of this claim are load-bearing: the exact
        sign flip is what signal.py's docstring promises; the magnitude
        mismatch is exactly why stats.py verifies E[A]=0 under the shuffle
        null empirically (Monte Carlo) rather than asserting it as an
        algebraic identity.
        """
        rng = np.random.default_rng(2)
        r = rng.normal(size=25)
        u = rng.normal(size=25)
        a_fwd = float(levy_area_of_windows(r[None, :], u[None, :])[0])
        a_rev = float(levy_area_of_windows(r[None, ::-1], u[None, ::-1])[0])

        assert np.sign(a_rev) == pytest.approx(-np.sign(a_fwd))
        assert a_rev != pytest.approx(-a_fwd, abs=1e-9)  # magnitude genuinely differs
        assert a_rev == pytest.approx(-a_fwd, rel=0.10)  # ...but only mildly so


class TestExactZeroMeanUnderShuffle:
    def test_raw_unnormalized_shoelace_has_exactly_zero_mean_over_all_permutations(self):
        """Exact combinatorial proof (not Monte Carlo) for small n: the brief
        claims E[A] = 0 exactly under exchangeability of day-order. Average
        W(r, u) = shoelace(cumsum(r), cumsum(u)) (no z-scoring) over
        literally all n! permutations of the day order -- must be exactly
        zero. (Equivalently, by test_anchored_shoelace_equals_double_sum_exactly,
        this is the same claim as E[sum_{s'<s}(r_s' u_s - r_s u_s')] = 0 over
        the window's n-1 cumsum-derived increments, which holds because for
        any fixed pair of increment-positions, a uniformly random day-order
        permutation makes the two cross terms exchangeable in expectation.)
        """
        rng = np.random.default_rng(3)
        n = 6  # 6! = 720, small enough to brute-force exactly
        r = rng.normal(size=n)
        u = rng.normal(size=n)

        total = 0.0
        count = 0
        for perm in itertools.permutations(range(n)):
            idx = np.array(perm)
            total += _shoelace_raw(np.cumsum(r[idx]), np.cumsum(u[idx]))
            count += 1
        assert count == 720
        mean_over_all_permutations = total / count
        assert mean_over_all_permutations == pytest.approx(0.0, abs=1e-10)

    def test_production_area_has_approximately_zero_mean_under_random_shuffle(self):
        """The exact zero-mean result above is for the raw double sum B, not
        the normalized/anchored A actually used as the signal (E[B]=0 does
        not imply E[A]=0 -- a ratio of dependent random variables). This is
        the weaker, empirical claim that E[A]=0 is a good approximation:
        over many random day-order permutations of a fixed (r, u) window,
        the mean of the production Levy area should be statistically
        indistinguishable from zero relative to its own spread. (This is
        exactly what stats.py's pre-registered shuffle-null test relies on;
        this test pins down that the reliance is justified.)
        """
        rng = np.random.default_rng(4)
        n = 20
        r = rng.normal(size=n)
        u = rng.normal(size=n)

        n_reps = 2000
        perms = np.stack([rng.permutation(n) for _ in range(n_reps)])
        areas = levy_area_of_windows(
            np.take_along_axis(np.broadcast_to(r, (n_reps, n)), perms, axis=1),
            np.take_along_axis(np.broadcast_to(u, (n_reps, n)), perms, axis=1),
        )
        mean_area = float(np.mean(areas))
        se_area = float(np.std(areas, ddof=1) / np.sqrt(n_reps))
        assert abs(mean_area) < 3.0 * se_area


class TestKnownAnswerRotationDirection:
    def test_volume_leading_price_gives_positive_q(self):
        """Construct a window where signed volume clearly moves first and
        price follows with a lag -- q_i = -A_i must come out positive
        ("volume leder priset").
        """
        n = 20
        t = np.arange(n)
        u = np.sin(2 * np.pi * t / n)          # volume leads: one full cycle
        r = np.sin(2 * np.pi * (t - 4) / n) * 0.01  # price follows ~4 days later, smaller scale
        q = -float(levy_area_of_windows(r[None, :], u[None, :])[0])
        assert q > 0

    def test_price_leading_volume_gives_negative_q(self):
        """Mirror case: price moves first, volume follows -- q_i must be negative."""
        n = 20
        t = np.arange(n)
        r = np.sin(2 * np.pi * t / n) * 0.01
        u = np.sin(2 * np.pi * (t - 4) / n)
        q = -float(levy_area_of_windows(r[None, :], u[None, :])[0])
        assert q < 0

    def test_perfectly_synchronized_path_has_near_zero_area(self):
        """If u is a pure scalar multiple of r (u_s = k * r_s for a constant
        k), then V = cumsum(u) = k * cumsum(r) = k * P exactly -- (P, V)
        traces a straight line through the origin, not a loop, so it
        encloses zero area. Note: an *additive* offset (u = k*r + c) would
        NOT give a straight line -- cumsum(u) would pick up an extra c*s
        drift term that cumsum(r) does not share, so this must be pure
        scaling, no shift.
        """
        n = 20
        rng = np.random.default_rng(5)
        r = rng.normal(size=n) * 0.01
        u = 3.0 * r
        a = float(levy_area_of_windows(r[None, :], u[None, :])[0])
        assert a == pytest.approx(0.0, abs=1e-8)


class TestDegenerateInputs:
    def test_zero_variance_window_gives_nan_not_crash(self):
        r = np.zeros(20)
        u = np.linspace(-1, 1, 20)
        a = levy_area_of_windows(r[None, :], u[None, :])
        assert np.isnan(a[0])

    def test_nan_in_window_propagates_to_nan_not_crash(self):
        r = np.random.default_rng(6).normal(size=20)
        u = np.random.default_rng(7).normal(size=20)
        u[5] = np.nan
        a = levy_area_of_windows(r[None, :], u[None, :])
        assert np.isnan(a[0])


class TestRollingPanelWrapper:
    def test_rolling_levy_area_matches_single_window_kernel(self):
        rng = np.random.default_rng(8)
        n_days, window = 60, 20
        r = pd.DataFrame({"AAA": rng.normal(size=n_days), "BBB": rng.normal(size=n_days)})
        u = pd.DataFrame({"AAA": rng.normal(size=n_days), "BBB": rng.normal(size=n_days)})

        rolled = rolling_levy_area(r, u, window)

        assert rolled.iloc[: window - 1].isna().all().all()
        # Spot-check a handful of causal windows against the direct kernel.
        for t in [window - 1, window + 5, n_days - 1]:
            for col in ["AAA", "BBB"]:
                r_win = r[col].values[t - window + 1 : t + 1]
                u_win = u[col].values[t - window + 1 : t + 1]
                expected = float(levy_area_of_windows(r_win[None, :], u_win[None, :])[0])
                assert rolled[col].iloc[t] == pytest.approx(expected, abs=1e-9)

    def test_no_lookahead_prefix_invariance(self):
        """The Levy area at date t must depend only on data through t: cutting
        the series short at some later date must not change earlier values
        (a direct causality/no-lookahead check, in the spirit of
        irreversibility_lab's backtest-mechanics tests).
        """
        rng = np.random.default_rng(9)
        n_days, window = 80, 20
        r = pd.Series(rng.normal(size=n_days), name="X").to_frame()
        u = pd.Series(rng.normal(size=n_days), name="X").to_frame()

        full = rolling_levy_area(r, u, window)
        truncated = rolling_levy_area(r.iloc[:50], u.iloc[:50], window)

        pd.testing.assert_series_equal(full["X"].iloc[:50], truncated["X"], check_names=False)


class TestCrossSectionalZScore:
    def test_winsorization_caps_outliers_before_zscoring(self):
        row = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 1000.0]], columns=list("abcde"))
        z = cross_sectional_zscore(row, 0.01, 0.99)
        # The extreme outlier must be clipped, not dominate the z-score scale.
        assert z["e"].iloc[0] < 10.0

    def test_nan_stays_nan(self):
        row = pd.DataFrame([[1.0, 2.0, np.nan, 4.0, 5.0]], columns=list("abcde"))
        z = cross_sectional_zscore(row, 0.01, 0.99)
        assert np.isnan(z["c"].iloc[0])
        assert not z["a"].isna().iloc[0]

    def test_row_mean_zero_when_no_winsorization_bites(self):
        rng = np.random.default_rng(10)
        row = pd.DataFrame(rng.normal(size=(1, 50)))
        z = cross_sectional_zscore(row, 0.0, 1.0)
        assert z.iloc[0].mean() == pytest.approx(0.0, abs=1e-9)
        assert z.iloc[0].std(ddof=1) == pytest.approx(1.0, abs=1e-9)


class TestDirectionFactor:
    def test_sign_transform(self):
        r = pd.DataFrame({"A": [0.01] * 25 + [-0.01] * 5})
        d = direction_factor(r, window=20, transform="sign", tanh_scale_days=20)
        assert d["A"].iloc[19] == 1.0

    def test_tanh_transform_bounded_and_same_sign_as_formation_return(self):
        rng = np.random.default_rng(11)
        r = pd.DataFrame({"A": rng.normal(scale=0.01, size=60)})
        d = direction_factor(r, window=20, transform="tanh", tanh_scale_days=20)
        r_n = formation_return(r, window=20)
        valid = d["A"].notna() & r_n["A"].notna() & (r_n["A"] != 0)
        assert (d["A"][valid].abs() <= 1.0).all()
        assert np.all(np.sign(d["A"][valid]) == np.sign(r_n["A"][valid]))


class TestSignedDollarVolume:
    def test_sign_matches_return_sign(self):
        from vridmomentet.data import Panel
        from vridmomentet.universe import PointInTimeMembership

        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        rng = np.random.default_rng(12)
        close = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(size=100))}, index=idx)
        adj_close = close.copy()
        adj_open = close.copy()
        volume = pd.DataFrame({"A": rng.integers(1_000, 100_000, size=100).astype(float)}, index=idx)
        panel = Panel(close=close, adj_close=adj_close, adj_open=adj_open, volume=volume,
                      membership=PointInTimeMembership([]))
        u = signed_dollar_volume(panel)
        r = panel.log_returns
        valid = u["A"].notna() & r["A"].notna() & (r["A"] != 0)
        assert np.all(np.sign(u["A"][valid]) == np.sign(r["A"][valid]))
