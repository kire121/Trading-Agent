# Ny testsvit för lib/synth_ohlc.py (promoverad från
# research/flodmarket/synth.py -- den branchen hade ingen egen dedikerad
# testfil för denna modul, se docs/INSTRUKTION.md avsnitt 7).
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.synth_ohlc import simulate_panel, panel_to_multiindex, plant_effect  # noqa: E402


class SimulatePanelTests(unittest.TestCase):
    def test_shapes_and_asset_count(self):
        panel = simulate_panel(n_assets=6, n_days=100, seed=1)
        for field in ("O", "H", "L", "C"):
            self.assertEqual(panel[field].shape, (100, 6))
        self.assertEqual(len(panel["assets"]), 6)
        self.assertEqual(len(panel["sigma_year"]), 6)
        self.assertEqual(len(panel["dates"]), 100)

    def test_high_low_bracket_open_close(self):
        panel = simulate_panel(n_assets=4, n_days=50, seed=2)
        O, H, L, C = panel["O"], panel["H"], panel["L"], panel["C"]
        self.assertTrue((H >= O).all().all())
        self.assertTrue((H >= C).all().all())
        self.assertTrue((L <= O).all().all())
        self.assertTrue((L <= C).all().all())
        self.assertTrue((H >= L).all().all())

    def test_deterministic_given_seed(self):
        p1 = simulate_panel(n_assets=3, n_days=30, seed=42)
        p2 = simulate_panel(n_assets=3, n_days=30, seed=42)
        pd.testing.assert_frame_equal(p1["O"], p2["O"])
        pd.testing.assert_frame_equal(p1["C"], p2["C"])

    def test_different_seed_gives_different_path(self):
        p1 = simulate_panel(n_assets=3, n_days=30, seed=1)
        p2 = simulate_panel(n_assets=3, n_days=30, seed=2)
        self.assertFalse(p1["C"].equals(p2["C"]))

    def test_sigma_year_cycles_through_levels(self):
        panel = simulate_panel(n_assets=8, n_days=10, seed=3)
        levels = sorted(set(panel["sigma_year"].values()))
        self.assertEqual(levels, [0.05, 0.10, 0.20, 0.40])

    def test_all_prices_positive(self):
        panel = simulate_panel(n_assets=5, n_days=200, seed=4)
        for field in ("O", "H", "L", "C"):
            self.assertTrue((panel[field] > 0).all().all())


class PanelToMultiindexTests(unittest.TestCase):
    def test_round_trips_values_and_shape(self):
        panel = simulate_panel(n_assets=3, n_days=20, seed=5)
        mi = panel_to_multiindex(panel)
        for field in ("O", "H", "L", "C"):
            self.assertEqual(len(mi[field]), 3 * 20)
            self.assertEqual(list(mi[field].index.names), ["ticker", "date"])

        asset = panel["assets"][0]
        date = panel["dates"][0]
        self.assertAlmostEqual(mi["O"].loc[(asset, date)], panel["O"].loc[date, asset])
        self.assertAlmostEqual(mi["C"].loc[(asset, date)], panel["C"].loc[date, asset])

    def test_no_ticker_boundary_crossing_in_sort_order(self):
        panel = simulate_panel(n_assets=4, n_days=10, seed=6)
        mi = panel_to_multiindex(panel)
        tickers_seen = mi["C"].index.get_level_values("ticker")
        # sorterad -> varje tickers block är sammanhängande
        change_points = (tickers_seen != tickers_seen.to_series().shift(1)).sum()
        self.assertEqual(change_points, 4)  # en övergång per ticker (inkl. första)


class PlantEffectTests(unittest.TestCase):
    def test_does_not_mutate_input_panel(self):
        panel = simulate_panel(n_assets=3, n_days=50, seed=7)
        C_before = panel["C"].copy()
        plant_effect(panel, seed=7, theta=0.05, mixed_sign=False)
        pd.testing.assert_frame_equal(panel["C"], C_before)

    def test_zero_theta_leaves_close_unchanged(self):
        panel = simulate_panel(n_assets=3, n_days=50, seed=8)
        biased = plant_effect(panel, seed=8, theta=0.0, mixed_sign=False)
        pd.testing.assert_frame_equal(biased["C"], panel["C"])

    def test_nonzero_theta_changes_close_for_some_bars(self):
        panel = simulate_panel(n_assets=3, n_days=50, seed=9)
        biased = plant_effect(panel, seed=9, theta=0.3, mixed_sign=False)
        self.assertFalse(biased["C"].equals(panel["C"]))
        # close förblir inom [L,H] (klampad) för varje bar
        self.assertTrue((biased["C"] >= panel["L"] - 1e-9).all().all())
        self.assertTrue((biased["C"] <= panel["H"] + 1e-9).all().all())

    def test_z_factor_present_with_unit_variance_per_asset(self):
        panel = simulate_panel(n_assets=3, n_days=300, seed=10)
        biased = plant_effect(panel, seed=10, theta=0.1, mixed_sign=False)
        self.assertIn("z", biased)
        for asset in panel["assets"]:
            # ddof=0: plant_effect normaliserar med numpy-arrayens .std()
            # (populationsvarians, ddof=0), inte pandas default ddof=1.
            self.assertAlmostEqual(biased["z"][asset].std(ddof=0), 1.0, places=6)

    def test_mixed_sign_differs_from_single_sign_z(self):
        panel = simulate_panel(n_assets=2, n_days=300, seed=11)
        single = plant_effect(panel, seed=11, theta=0.1, mixed_sign=False)
        mixed = plant_effect(panel, seed=11, theta=0.1, mixed_sign=True)
        self.assertFalse(single["z"].equals(mixed["z"]))
        self.assertTrue(mixed["mixed_sign"])
        self.assertFalse(single["mixed_sign"])

    def test_theta_and_mixed_sign_recorded_on_output(self):
        panel = simulate_panel(n_assets=2, n_days=30, seed=12)
        biased = plant_effect(panel, seed=12, theta=0.07, mixed_sign=True)
        self.assertAlmostEqual(biased["theta"], 0.07)
        self.assertTrue(biased["mixed_sign"])


if __name__ == "__main__":
    unittest.main()
