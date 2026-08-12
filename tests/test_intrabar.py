# Promoverad från research/flodmarket/tests/test_intrabar.py, branch
# claude/strategy-spec-implementation-qy84em, commit
# eca52d9e42ce949c35eb4c46894e6b17568f9cc2, vid Flodmärkets
# levande-komponenter-promovering. Importvägen ändrad
# (research.flodmarket.intrabar -> lib.intrabar); fe_demean anropas redan
# med explicit window= i originalet, så ingen testlogik ändrad.
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.intrabar import shadow_stats, rolling_tstat, fe_demean  # noqa: E402


def _single_bar(O, H, L, C):
    idx = pd.date_range("2020-01-02", periods=1, freq="B")
    return (pd.Series([O], index=idx), pd.Series([H], index=idx),
            pd.Series([L], index=idx), pd.Series([C], index=idx))


class ShadowStatsIdentityTests(unittest.TestCase):
    # Exakta testfall från docs/flodmarket_forregistrering.md avsnitt 12.3.

    def test_case1_basic_bar(self):
        O, H, L, C = _single_bar(100, 110, 95, 108)
        out = shadow_stats(O, H, L, C)
        self.assertAlmostEqual(out["U"].iloc[0], 2 / 15, places=5)
        self.assertAlmostEqual(out["D"].iloc[0], 5 / 15, places=5)
        self.assertAlmostEqual(out["b"].iloc[0], 8 / 15, places=5)
        self.assertAlmostEqual(out["s"].iloc[0], 0.2, places=5)

    def test_case2_antisymmetry(self):
        O1, H1, L1, C1 = _single_bar(100, 110, 95, 108)
        s1 = shadow_stats(O1, H1, L1, C1)["s"].iloc[0]

        O2, H2, L2, C2 = _single_bar(100, 105, 90, 92)
        out2 = shadow_stats(O2, H2, L2, C2)
        self.assertAlmostEqual(out2["U"].iloc[0], 5 / 15, places=5)
        self.assertAlmostEqual(out2["D"].iloc[0], 2 / 15, places=5)
        self.assertAlmostEqual(out2["s"].iloc[0], -0.2, places=5)
        self.assertAlmostEqual(out2["s"].iloc[0], -s1, places=10)

    def test_case3_scale_and_adjustment_invariance(self):
        base = (100, 110, 95, 108)
        O0, H0, L0, C0 = _single_bar(*base)
        s0 = shadow_stats(O0, H0, L0, C0)

        for f in (3.0, 0.37, 1000.0):
            O, H, L, C = _single_bar(*(v * f for v in base))
            out = shadow_stats(O, H, L, C)
            self.assertAlmostEqual(out["U"].iloc[0], s0["U"].iloc[0], places=8)
            self.assertAlmostEqual(out["D"].iloc[0], s0["D"].iloc[0], places=8)
            self.assertAlmostEqual(out["b"].iloc[0], s0["b"].iloc[0], places=8)
            self.assertAlmostEqual(out["s"].iloc[0], s0["s"].iloc[0], places=8)

    def test_case4_clamp(self):
        O, H, L, C = _single_bar(112, 110, 95, 108)
        out = shadow_stats(O, H, L, C)
        self.assertTrue(bool(out["flag_clamped"].iloc[0]))
        self.assertAlmostEqual(out["U"].iloc[0], 0.0, places=8)
        self.assertAlmostEqual(out["D"].iloc[0], 13 / 15, places=5)

    def test_case5_zero_range_is_nan_no_exception(self):
        O, H, L, C = _single_bar(100, 100, 100, 100)
        try:
            out = shadow_stats(O, H, L, C)
        except Exception as e:  # pragma: no cover
            self.fail(f"shadow_stats raised on H=L: {e!r}")
        self.assertTrue(np.isnan(out["s"].iloc[0]))
        self.assertTrue(bool(out["flag_zerorange"].iloc[0]))

    def test_flat_bar_ohlc_equal_is_nan(self):
        O, H, L, C = _single_bar(100, 100, 100, 100)
        out = shadow_stats(O, H, L, C)
        self.assertTrue(np.isnan(out["U"].iloc[0]))
        self.assertTrue(np.isnan(out["D"].iloc[0]))
        self.assertTrue(np.isnan(out["b"].iloc[0]))
        self.assertTrue(np.isnan(out["s"].iloc[0]))


class ShadowStatsPanelTests(unittest.TestCase):
    def test_synthopen_flag_within_ticker_not_across_boundary(self):
        idx = pd.MultiIndex.from_tuples(
            [("AAA", pd.Timestamp("2020-01-02")), ("AAA", pd.Timestamp("2020-01-03")),
             ("BBB", pd.Timestamp("2020-01-02")), ("BBB", pd.Timestamp("2020-01-03"))],
            names=["ticker", "date"],
        )
        O = pd.Series([100.0, 101.0, 50.0, 101.0], index=idx)  # BBB dag2 O==AAA dag1 C nästan av en slump
        H = pd.Series([102.0, 103.0, 52.0, 103.0], index=idx)
        L = pd.Series([99.0, 100.0, 49.0, 100.0], index=idx)
        C = pd.Series([101.0, 102.0, 51.0, 102.0], index=idx)
        out = shadow_stats(O, H, L, C)
        # AAA dag2 O(101.0) == AAA dag1 C(101.0) -> syntetisk open flaggad
        self.assertTrue(bool(out.loc[("AAA", pd.Timestamp("2020-01-03")), "flag_synthopen"]))
        # BBB dag2 O(101.0) != BBB dag1 C(51.0) -> ej flaggad, även om den
        # råkar matcha AAA:s föregående close (får inte läcka över tickergränsen)
        self.assertFalse(bool(out.loc[("BBB", pd.Timestamp("2020-01-03")), "flag_synthopen"]))
        # Första baren för varje ticker har ingen föregående close -> aldrig flaggad
        self.assertFalse(bool(out.loc[("AAA", pd.Timestamp("2020-01-02")), "flag_synthopen"]))


class RollingTstatTests(unittest.TestCase):
    def test_keff_below_min_valid_is_nan(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        s = pd.Series([0.1] * 10, index=idx)
        s.iloc[2:] = np.nan  # bara 2 av 10 giltiga -> K_eff=2 < 0.8*10=8
        S = rolling_tstat(s, K=10, min_valid=0.8)
        self.assertTrue(np.isnan(S.iloc[-1]))

    def test_sufficient_keff_computes_tstat(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.05, 0.2, 10), index=idx)
        S = rolling_tstat(s, K=10, min_valid=0.8)
        window = s.to_numpy()
        expected = window.mean() / (window.std(ddof=1) / np.sqrt(10))
        self.assertAlmostEqual(S.iloc[-1], expected, places=8)

    def test_zero_std_window_is_nan_not_inf(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        s = pd.Series([0.05] * 10, index=idx)  # nollvarians
        S = rolling_tstat(s, K=10, min_valid=0.8)
        self.assertTrue(np.isnan(S.iloc[-1]))


class FeDemeanTests(unittest.TestCase):
    def test_disjoint_window_and_pit(self):
        idx = pd.date_range("2000-01-03", periods=40 + 252 + 10, freq="B")
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 0.1, len(idx)), index=idx)
        K = 40
        s_tilde = fe_demean(s, K, window=252)
        # Fullt fönster tillgängligt vid position i = K + 252 - 1 (0-indexerat)
        i = K + 252 - 1
        expected_mean = s.iloc[i - K - 251: i - K + 1].mean()
        self.assertAlmostEqual(s.iloc[i] - s_tilde.iloc[i], expected_mean, places=8)
        # Innan fullt fönster finns tillgängligt -> NaN (inget partiellt fönster)
        self.assertTrue(np.isnan(s_tilde.iloc[0]))

    def test_window_argument_is_required(self):
        # Generalisering vid promoveringen: inget Flodmärket-specifikt
        # default (window=252) kvar i signaturen -- anroparen måste skicka
        # det uttryckligen.
        idx = pd.date_range("2020-01-01", periods=20, freq="B")
        s = pd.Series(np.zeros(len(idx)), index=idx)
        with self.assertRaises(TypeError):
            fe_demean(s, 5)  # saknar window


if __name__ == "__main__":
    unittest.main()
