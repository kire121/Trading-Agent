import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.twins import quantile_map_to, twin_is_alive  # noqa: E402


class QuantileMapToTests(unittest.TestCase):
    # Portad från research/smittotalet/tests/test_twins.py (samma invarianter
    # som originalets shippade tester, se lib/twins.py proveniens-header).

    def test_preserves_reference_distribution(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        rng = np.random.default_rng(0)
        reference = pd.Series(rng.uniform(0.3, 1.3, len(idx)), index=idx)
        raw = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
        mapped = quantile_map_to(reference, raw)
        ref_sorted = np.sort(reference.reindex(mapped.dropna().index).to_numpy())
        mapped_sorted = np.sort(mapped.dropna().to_numpy())
        self.assertTrue(np.allclose(ref_sorted, mapped_sorted, atol=1e-6))

    def test_preserves_raw_ordering(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        reference = pd.Series(np.linspace(0.3, 1.3, 50), index=idx)
        raw = pd.Series(np.arange(50), index=idx)  # strikt växande
        mapped = quantile_map_to(reference, raw)
        self.assertTrue((mapped.diff().dropna() >= 0).all())

    def test_fewer_than_two_overlapping_points_is_all_nan(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        reference = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan], index=idx)
        raw = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        mapped = quantile_map_to(reference, raw)
        self.assertTrue(mapped.isna().all())


class TwinIsAliveTests(unittest.TestCase):
    def test_healthy_twin_is_alive(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({
            "score": rng.normal(0, 1, len(idx)),
            "direction": rng.choice([-1.0, 1.0], len(idx)),
        }, index=idx)
        checks = twin_is_alive(frame)
        self.assertTrue(checks["alive"])
        self.assertTrue(checks["coverage_ok"])
        self.assertTrue(checks["score_dispersion_ok"])
        self.assertTrue(checks["direction_balance_ok"])

    def test_constant_score_is_not_alive(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        frame = pd.DataFrame({"score": [1.0] * len(idx), "direction": [1.0, -1.0] * 50}, index=idx)
        checks = twin_is_alive(frame)
        self.assertFalse(checks["score_dispersion_ok"])
        self.assertFalse(checks["alive"])

    def test_direction_collapsed_to_one_sign_is_not_alive(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"score": rng.normal(0, 1, len(idx)), "direction": [1.0] * len(idx)}, index=idx)
        checks = twin_is_alive(frame)
        self.assertFalse(checks["direction_balance_ok"])
        self.assertFalse(checks["alive"])


if __name__ == "__main__":
    unittest.main()
