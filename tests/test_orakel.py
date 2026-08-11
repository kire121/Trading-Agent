import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.orakel import rearrangement_oracle  # noqa: E402


class RearrangementOracleTests(unittest.TestCase):
    def test_same_multiset_as_input_values(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        rng = np.random.default_rng(0)
        values = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
        target = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
        oracle = rearrangement_oracle(values, target)
        self.assertTrue(np.allclose(np.sort(oracle.to_numpy()), np.sort(values.to_numpy())))

    def test_maximizes_sum_of_products_vs_random_pairings(self):
        # Rearrangement-olikheten: sum(values*target) ska vara MAXIMAL för
        # orakel-parningen jämfört med varje slumpmässig permutation av samma
        # multiset mot samma target-ordning.
        idx = pd.date_range("2020-01-01", periods=40, freq="B")
        rng = np.random.default_rng(1)
        values = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
        target = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
        oracle = rearrangement_oracle(values, target)
        oracle_sum = float((oracle * target).sum())

        for _ in range(20):
            shuffled = pd.Series(rng.permutation(values.to_numpy()), index=values.index)
            shuffled_sum = float((shuffled * target).sum())
            self.assertLessEqual(shuffled_sum, oracle_sum + 1e-9)

    def test_ascending_pairing(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        values = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0], index=idx)
        # target redan stigande i indexordning -> ranken matchar indexordningen
        # exakt, så orakel-parningen ska ge values sorterat stigande, oavsett
        # values ursprungliga ordning.
        target = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
        oracle = rearrangement_oracle(values, target)
        self.assertTrue((oracle.to_numpy() == np.array([1.0, 2.0, 3.0, 4.0, 5.0])).all())

    def test_fewer_than_two_points_returns_empty(self):
        idx = pd.date_range("2020-01-01", periods=1, freq="B")
        values = pd.Series([1.0], index=idx)
        target = pd.Series([2.0], index=idx)
        oracle = rearrangement_oracle(values, target)
        self.assertEqual(len(oracle), 0)


if __name__ == "__main__":
    unittest.main()
