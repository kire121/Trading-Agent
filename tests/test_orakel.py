import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.orakel import rearrangement_oracle, rearrangement_oracle_returns  # noqa: E402


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


class RearrangementOracleReturnsTests(unittest.TestCase):
    """Regressionstest för rearrangement_oracle_returns' två kompositions-
    lägen. Fixturet återanvänder RearrangementOracleTests.test_ascending_pairing
    ovan (target redan stigande i indexordning -> oraklet ger values
    sorterat stigande) så att båda lägenas förväntade utdata kan räknas ut
    för hand, utan att tautologiskt återanvända samma kompositionsformel i
    testet som i implementationen."""

    def _fixture(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        values = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0], index=idx)
        target = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
        # oracle (per rearrangement_oracle, se test_ascending_pairing) blir
        # [1, 2, 3, 4, 5] i indexordning eftersom target redan är stigande.
        return values, target

    def test_multiplicative_matches_smittotalets_oracle_cap_test_formula(self):
        # Smittotalets oracle_cap_test: oracle_returns = base_week * oracle_g(...),
        # dvs. target * rearrangement_oracle(values, target).
        values, target = self._fixture()
        result = rearrangement_oracle_returns(values, target, mode="multiplicative")
        expected = np.array([10.0 * 1.0, 20.0 * 2.0, 30.0 * 3.0, 40.0 * 4.0, 50.0 * 5.0])
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-12)
        # Explicit reference to the un-composed primitive, guarding against a
        # future refactor silently changing what "multiplicative" composes.
        oracle = rearrangement_oracle(values, target)
        np.testing.assert_allclose(result.to_numpy(), (target * oracle).to_numpy(), atol=1e-12)

    def test_multiplicative_is_default_mode(self):
        values, target = self._fixture()
        default_result = rearrangement_oracle_returns(values, target)
        explicit_result = rearrangement_oracle_returns(values, target, mode="multiplicative")
        pd.testing.assert_series_equal(default_result, explicit_result)

    def test_additive_matches_timglasets_clock_oracle_test_formula(self):
        # research/timglaset/oracle.py::clock_oracle_test: oracle_returns =
        # T1 + rearrangement_oracle(overlay, T1), dvs. target +
        # rearrangement_oracle(values, target) -- overlayn är en
        # skillnadsserie, inte en multiplikativ tilt (spec §0).
        values, target = self._fixture()
        result = rearrangement_oracle_returns(values, target, mode="additive")
        expected = np.array([10.0 + 1.0, 20.0 + 2.0, 30.0 + 3.0, 40.0 + 4.0, 50.0 + 5.0])
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-12)

    def test_additive_and_multiplicative_diverge_on_the_same_inputs(self):
        # Guards against a copy-paste bug collapsing the two modes to the
        # same composition.
        values, target = self._fixture()
        additive = rearrangement_oracle_returns(values, target, mode="additive")
        multiplicative = rearrangement_oracle_returns(values, target, mode="multiplicative")
        self.assertFalse(np.allclose(additive.to_numpy(), multiplicative.to_numpy()))

    def test_unknown_mode_raises(self):
        values, target = self._fixture()
        with self.assertRaises(ValueError):
            rearrangement_oracle_returns(values, target, mode="geometric")


if __name__ == "__main__":
    unittest.main()
