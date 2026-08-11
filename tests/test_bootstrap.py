import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from lib.bootstrap import (  # noqa: E402
    circular_block_bootstrap_1d,
    circular_block_bootstrap_columns,
    circular_block_bootstrap_rows,
    empirical_pvalue,
    stationary_block_bootstrap_indices,
    synthetic_price_path,
    synthetic_return_path,
)


class CircularBlockBootstrapTests(unittest.TestCase):
    def test_1d_preserves_length_and_value_set(self):
        x = np.arange(20, dtype=float)
        rng = np.random.default_rng(0)
        out = circular_block_bootstrap_1d(x, block_size=4, rng=rng)
        self.assertEqual(len(out), len(x))
        self.assertTrue(set(out.tolist()).issubset(set(x.tolist())))

    def test_1d_deterministic_given_same_rng_state(self):
        x = np.arange(20, dtype=float)
        out1 = circular_block_bootstrap_1d(x, block_size=4, rng=np.random.default_rng(42))
        out2 = circular_block_bootstrap_1d(x, block_size=4, rng=np.random.default_rng(42))
        self.assertTrue(np.array_equal(out1, out2))

    def test_columns_independent_per_column(self):
        mat = np.column_stack([np.arange(30, dtype=float), np.arange(100, 130, dtype=float)])
        rng = np.random.default_rng(1)
        out = circular_block_bootstrap_columns(mat, block_size=5, rng=rng)
        self.assertEqual(out.shape, mat.shape)
        self.assertTrue(set(out[:, 0].tolist()).issubset(set(mat[:, 0].tolist())))
        self.assertTrue(set(out[:, 1].tolist()).issubset(set(mat[:, 1].tolist())))

    def test_rows_preserves_cross_section(self):
        mat = np.column_stack([np.arange(30, dtype=float), np.arange(100, 130, dtype=float)])
        rng = np.random.default_rng(2)
        out = circular_block_bootstrap_rows(mat, block_size=5, rng=rng)
        self.assertEqual(out.shape, mat.shape)
        # Varje rad ska fortfarande vara ett äkta (col0, col1)-par från originalet
        # (samma dags tvärsnitt bevaras, bara tidsordningen scramblas).
        original_pairs = {tuple(row) for row in mat}
        for row in out:
            self.assertIn(tuple(row), original_pairs)


class StationaryBootstrapTests(unittest.TestCase):
    def test_indices_within_bounds_and_correct_length(self):
        rng = np.random.default_rng(0)
        idx = stationary_block_bootstrap_indices(n=50, block_size=5.0, rng=rng)
        self.assertEqual(len(idx), 50)
        self.assertTrue((idx >= 0).all() and (idx < 50).all())


class SyntheticPathTests(unittest.TestCase):
    def test_return_path_length_and_membership(self):
        returns = np.array([0.01, -0.02, 0.005, 0.0, -0.01, 0.02])
        rng = np.random.default_rng(0)
        path = synthetic_return_path(returns, length=100, rng=rng, block=3, kind="circular")
        self.assertEqual(len(path), 100)
        self.assertTrue(set(np.round(path, 10).tolist()).issubset(set(np.round(returns, 10).tolist())))

    def test_price_path_is_positive_and_correct_length(self):
        returns = np.array([0.01, -0.02, 0.005, 0.0, -0.01, 0.02])
        rng = np.random.default_rng(0)
        prices = synthetic_price_path(returns, length=50, rng=rng, block=3, start_price=100.0, kind="stationary")
        self.assertEqual(len(prices), 51)  # +1 för startpriset (cumsum-konventionen)
        self.assertTrue((prices > 0).all())

    def test_unknown_kind_raises(self):
        returns = np.array([0.01, -0.02])
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            synthetic_return_path(returns, length=10, rng=rng, block=2, kind="not_a_real_kind")


class EmpiricalPvalueTests(unittest.TestCase):
    def test_never_returns_zero(self):
        null_draws = np.array([0.0, 0.1, 0.2, 0.3])
        p = empirical_pvalue(observed=10.0, null_draws=null_draws)  # inget null-drag når observed
        self.assertGreater(p, 0.0)

    def test_bounds(self):
        null_draws = np.linspace(-1, 1, 100)
        p = empirical_pvalue(observed=0.0, null_draws=null_draws)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
