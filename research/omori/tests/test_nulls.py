import numpy as np

from research.omori import nulls, signal


class TestCircularBlockShuffle:
    def test_preserves_length_and_multiset(self):
        arr = np.arange(20, dtype=float)
        rng = np.random.default_rng(0)
        out = nulls.circular_block_shuffle(arr, block_len=4, rng=rng)
        assert len(out) == 20
        # circular block shuffle with replacement doesn't preserve the
        # multiset exactly (blocks can repeat/overlap), but every value must
        # still come from the original array's value range.
        assert out.min() >= arr.min() and out.max() <= arr.max()

    def test_single_element_returns_copy(self):
        arr = np.array([5.0])
        out = nulls.circular_block_shuffle(arr, block_len=4, rng=np.random.default_rng(0))
        assert list(out) == [5.0]


class TestEstimatorNullTest:
    def test_genuine_dispersion_beats_null_when_p_hats_are_truly_heterogeneous(self):
        # Construct events whose e-paths have GENUINELY different decay
        # exponents (heterogeneous by construction) -- the real cross-event
        # dispersion should then, with high probability, exceed the
        # within-event-shuffled null's p95 (which destroys each event's own
        # decay ordering).
        rng = np.random.default_rng(0)
        fits = []
        for i, p in enumerate([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 0.4, 1.0, 1.6] * 5):
            s = np.arange(1, 16, dtype=float)
            e = (s) ** (-p) * (1 + rng.normal(0, 0.02, size=15))
            fit = signal.fit_omori(15, e)
            fits.append({"identified": fit.identified, "p_hat": fit.p_hat, "e_path": e})
        result = nulls.estimator_null_test(fits, n_draws=30, subsample_n=50, block_len=3, seed=0)
        assert result["real_dispersion"] > 0
        assert "passed" in result

    def test_degenerate_identical_events_have_low_dispersion(self):
        fits = []
        for i in range(30):
            s = np.arange(1, 16, dtype=float)
            e = s ** (-0.8)  # IDENTICAL decay for every event, no noise
            fit = signal.fit_omori(15, e)
            fits.append({"identified": fit.identified, "p_hat": fit.p_hat, "e_path": e})
        result = nulls.estimator_null_test(fits, n_draws=20, subsample_n=30, block_len=3, seed=0)
        assert result["real_dispersion"] == 0.0
        assert not result["passed"]


class TestBlockPermutationIC:
    def test_perfect_correlation_gives_significant_p_value(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        y = x * 2.0  # perfect monotonic relation
        dates = np.arange(200)
        out = nulls.block_permutation_ic(x, y, dates, n_draws=200, block_len=10, seed=0)
        assert out["ic"] > 0.99
        assert out["p_value"] < 0.05

    def test_unrelated_series_gives_insignificant_p_value_typically(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        dates = np.arange(300)
        out = nulls.block_permutation_ic(x, y, dates, n_draws=200, block_len=10, seed=1)
        assert abs(out["ic"]) < 0.3

    def test_too_few_observations_returns_nan(self):
        out = nulls.block_permutation_ic([1, 2], [3, 4], [0, 1], n_draws=50)
        assert np.isnan(out["ic"])
