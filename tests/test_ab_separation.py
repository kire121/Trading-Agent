# Ny testsvit för lib/ab_separation.py (promoverad + kraftkalibreringsfixad
# från research/flodmarket/ab_separation.py, se modulens header för
# DÖDSORSAK/graven Flodmärket 2026-08-12). FlodmarketDeathRegressionTest
# längst ned är regressionstestet krävt av uppdraget: 10x1200x50 => AssertionError.
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.ab_separation import (  # noqa: E402
    assert_percentile_resolution,
    null_percentile,
    run_null_calibration,
    run_planted_effect_check,
    meta_achievability_check,
)


class AssertPercentileResolutionTests(unittest.TestCase):
    def test_p99_with_50_draws_raises(self):
        # 50*(1-0.99) = 0.5 < 5 -- exakt Flodmärkets fall.
        with self.assertRaises(AssertionError):
            assert_percentile_resolution(50, 0.99)

    def test_p99_with_500_draws_passes(self):
        # 500*(1-0.99) = 5.0 >= 5 -- den specmandaterade riktiga skalan
        # (K1.3/T3, STEG1_N_NULL_DRAWS=500 i research/flodmarket/config.py).
        assert_percentile_resolution(500, 0.99)  # no raise

    def test_p95_with_50_draws_raises(self):
        # 50*(1-0.95) = 2.5 < 5 -- samma brist gäller theta=0-kalibreringens
        # p95-tröskel, inte bara p99.
        with self.assertRaises(AssertionError):
            assert_percentile_resolution(50, 0.95)

    def test_p95_with_exactly_100_draws_passes(self):
        # 100*(1-0.95) = 5.0 >= 5 -- gränsfallet.
        assert_percentile_resolution(100, 0.95)  # no raise

    def test_invalid_quantile_raises_value_error(self):
        with self.assertRaises(ValueError):
            assert_percentile_resolution(1000, 1.5)
        with self.assertRaises(ValueError):
            assert_percentile_resolution(1000, 0.0)

    def test_zero_draws_raises(self):
        with self.assertRaises(AssertionError):
            assert_percentile_resolution(0, 0.95)

    def test_error_message_names_required_n(self):
        with self.assertRaises(AssertionError) as ctx:
            assert_percentile_resolution(50, 0.99)
        self.assertIn("500", str(ctx.exception))


class NullPercentileTests(unittest.TestCase):
    def test_matches_numpy_percentile_when_resolution_sufficient(self):
        arr = np.arange(1, 501, dtype=float)  # 500 dragningar
        got = null_percentile(arr, 0.99)
        expected = float(np.percentile(arr, 99.0))
        self.assertAlmostEqual(got, expected, places=8)

    def test_raises_when_insufficient(self):
        arr = np.arange(1, 11, dtype=float)  # 10 dragningar
        with self.assertRaises(AssertionError):
            null_percentile(arr, 0.99)

    def test_nonfinite_values_filtered_before_resolution_check(self):
        # 300 giltiga + 700 NaN -> filtreringen lämnar bara 300 giltiga
        # dragningar kvar; 300*(1-0.99)=3.0 < 5 -- ska fela trots att den
        # RÅA längden (1000) hade räckt.
        arr = np.concatenate([np.arange(300, dtype=float), np.full(700, np.nan)])
        with self.assertRaises(AssertionError):
            null_percentile(arr, 0.99)

    def test_nonfinite_values_filtered_but_still_enough(self):
        arr = np.concatenate([np.arange(600, dtype=float), np.full(50, np.nan)])
        # 600 giltiga kvar efter filtrering -> 600*0.01=6>=5, ska klara sig.
        got = null_percentile(arr, 0.99)
        expected = float(np.percentile(np.arange(600, dtype=float), 99.0))
        self.assertAlmostEqual(got, expected, places=8)


class RunNullCalibrationTests(unittest.TestCase):
    def test_well_calibrated_estimator_hits_target_rate(self):
        # estimator och nolla dras från EXAKT samma fördelning -> andelen
        # som slår sin egen p95-tröskel ska ligga nära 5%.
        def estimator_fn(seed):
            rng = np.random.default_rng(seed)
            return float(rng.normal(0, 1))

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed + 999_999)
            return rng.normal(0, 1, n_draws)

        result = run_null_calibration(estimator_fn, null_fn, n_outer_sims=300,
                                       n_inner_null_draws=200, seed_base=1000,
                                       null_percentile_q=0.95, target_exceedance_rate=0.05,
                                       tolerance=0.03)
        self.assertEqual(result["n_valid"], 300)
        self.assertTrue(result["passes"], msg=result)

    def test_insufficient_inner_draws_raises(self):
        def estimator_fn(seed):
            return 0.1

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed)
            return rng.normal(0, 1, n_draws)

        with self.assertRaises(AssertionError):
            run_null_calibration(estimator_fn, null_fn, n_outer_sims=5,
                                  n_inner_null_draws=50, seed_base=1,
                                  null_percentile_q=0.95)


class RunPlantedEffectCheckTests(unittest.TestCase):
    def test_strong_effect_passes(self):
        def estimator_fn(seed):
            return 0.5

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed)
            return rng.normal(0, 0.01, n_draws)

        result = run_planted_effect_check(estimator_fn, null_fn, n_inner_null_draws=500,
                                           seed=1, ic_min=0.02, null_percentile_q=0.99)
        self.assertTrue(result["passes"])

    def test_weak_effect_fails_without_raising(self):
        def estimator_fn(seed):
            return 0.001  # under ic_min

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed)
            return rng.normal(0, 0.01, n_draws)

        result = run_planted_effect_check(estimator_fn, null_fn, n_inner_null_draws=500,
                                           seed=1, ic_min=0.02, null_percentile_q=0.99)
        self.assertFalse(result["passes"])

    def test_insufficient_draws_raises(self):
        def estimator_fn(seed):
            return 0.5

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed)
            return rng.normal(0, 0.01, n_draws)

        with self.assertRaises(AssertionError):
            run_planted_effect_check(estimator_fn, null_fn, n_inner_null_draws=50,
                                      seed=1, ic_min=0.02, null_percentile_q=0.99)


class MetaAchievabilityCheckTests(unittest.TestCase):
    def test_high_power_scenario_passes(self):
        def estimator_fn(seed):
            return 0.5  # långt över ic_min och över nollan, varje gång

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed)
            return rng.normal(0, 0.01, n_draws)

        result = meta_achievability_check(estimator_fn, null_fn, n_inner_null_draws=500,
                                           ic_min=0.02, null_percentile_q=0.99, n_meta_reps=20,
                                           seed_base=1, min_pass_rate=0.80)
        self.assertEqual(result["pass_rate"], 1.0)
        self.assertTrue(result["passes"])

    def test_low_power_scenario_fails(self):
        # Observerad statistika är brusig och ligger nära tröskeln -> klarar
        # bara ibland. Inte deterministiskt 100%, ska landa klart under 80%.
        def estimator_fn(seed):
            rng = np.random.default_rng(seed + 5_000_000)
            return float(rng.normal(0.021, 0.02))  # nära ic_min=0.02, mycket brus

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed)
            return rng.normal(0, 0.02, n_draws)

        result = meta_achievability_check(estimator_fn, null_fn, n_inner_null_draws=500,
                                           ic_min=0.02, null_percentile_q=0.99, n_meta_reps=30,
                                           seed_base=1, min_pass_rate=0.80)
        self.assertLess(result["pass_rate"], 0.80)
        self.assertFalse(result["passes"])

    def test_insufficient_draws_raises_before_any_meta_rep_completes(self):
        def estimator_fn(seed):
            return 0.5

        def null_fn(seed, n_draws):
            rng = np.random.default_rng(seed)
            return rng.normal(0, 0.01, n_draws)

        with self.assertRaises(AssertionError):
            meta_achievability_check(estimator_fn, null_fn, n_inner_null_draws=50,
                                      ic_min=0.02, null_percentile_q=0.99, n_meta_reps=10,
                                      seed_base=1)


class FlodmarketDeathRegressionTest(unittest.TestCase):
    """Reproducerar exakt den batteriskala som fällde Flodmärket (graven
    2026-08-12, results/flodmarket/AVVIKELSER.md avsnitt 8 samt
    REPORT.md): 10 syntetiska tillgångar x 1200 dagar, 50 inre
    nolldragningar, en planterad enkeltecken-effekt utvärderad mot en
    p99-tröskel.

    Estimatorn/nollan här är en förenklad, pooled (icke-veckobeslutad)
    Spearman-IC byggd direkt på de tre promoverade modulerna
    (lib.synth_ohlc, lib.intrabar, lib.ab_separation) -- inte en
    fullständig återskapning av research/flodmarket/signal.py:s
    veckobeslutspipeline eller nulls.py:s block-permutation (ingen av dem
    promoverades hit, se docs/INSTRUKTION.md avsnitt 7). Den delar ändå
    exakt den skala (10x1200x50) och den p99-baserade grindmekanismen som
    faktiskt utlöste dödsorsaken -- och därmed exakt den brist denna fix
    stänger.
    """

    N_ASSETS = 10
    N_DAYS = 1200
    N_INNER_NULL_DRAWS = 50  # Flodmärkets faktiska, deklarerade skala (AVVIKELSER.md avsnitt 8)
    K = 40
    THETA = 0.02

    @staticmethod
    def _pooled_ic(g_wide, fwd_wide):
        from scipy import stats as scipy_stats
        g_flat = g_wide.stack()
        r_flat = fwd_wide.stack()
        common = pd.concat([g_flat, r_flat], axis=1, keys=["g", "r"]).dropna()
        if len(common) < 20:
            return float("nan")
        return float(scipy_stats.spearmanr(common["g"], common["r"]).correlation)

    def _estimator_fn(self, seed):
        from lib.synth_ohlc import simulate_panel, plant_effect, panel_to_multiindex
        from lib.intrabar import shadow_stats, rolling_tstat

        base = simulate_panel(n_assets=self.N_ASSETS, n_days=self.N_DAYS, seed=seed)
        biased = plant_effect(base, seed=seed, theta=self.THETA, mixed_sign=False)
        mi = panel_to_multiindex(biased)
        shadow = shadow_stats(mi["O"], mi["H"], mi["L"], mi["C"])
        S = rolling_tstat(shadow["s"], self.K, min_valid=0.8)
        g = (S / 2.0).clip(lower=-1.0, upper=1.0)
        g_wide = g.unstack("ticker")
        raw_fwd = base["C"].pct_change(5).shift(-5)
        biased_fwd = raw_fwd + self.THETA * biased["z"]
        return self._pooled_ic(g_wide, biased_fwd)

    def _null_fn(self, seed, n_draws):
        # Enkel (icke-block-)permutation av s inom varje tillgång --
        # tillräckligt för att generera en nollfördelning för DETTA
        # regressionstest; den riktiga block-permutationen (nulls.py)
        # promoverades inte hit (utanför omfattningen).
        from lib.synth_ohlc import simulate_panel, plant_effect, panel_to_multiindex
        from lib.intrabar import shadow_stats, rolling_tstat

        base = simulate_panel(n_assets=self.N_ASSETS, n_days=self.N_DAYS, seed=seed)
        biased = plant_effect(base, seed=seed, theta=self.THETA, mixed_sign=False)
        mi = panel_to_multiindex(biased)
        shadow = shadow_stats(mi["O"], mi["H"], mi["L"], mi["C"])
        s = shadow["s"]
        raw_fwd = base["C"].pct_change(5).shift(-5)
        biased_fwd = raw_fwd + self.THETA * biased["z"]

        rng = np.random.default_rng(seed + 10_000_019)
        draws = np.empty(n_draws)
        for i in range(n_draws):
            s_perm = s.groupby(level="ticker", group_keys=False).apply(
                lambda ser: pd.Series(rng.permutation(ser.to_numpy()), index=ser.index))
            S_perm = rolling_tstat(s_perm, self.K, min_valid=0.8)
            g_perm = (S_perm / 2.0).clip(lower=-1.0, upper=1.0).unstack("ticker")
            draws[i] = self._pooled_ic(g_perm, biased_fwd)
        return draws

    def test_reproduces_flodmarket_death_case(self):
        from lib.ab_separation import run_planted_effect_check

        with self.assertRaises(AssertionError) as ctx:
            run_planted_effect_check(self._estimator_fn, self._null_fn,
                                      n_inner_null_draws=self.N_INNER_NULL_DRAWS,
                                      seed=20260811, ic_min=0.02, null_percentile_q=0.99)
        self.assertIn("n_draws=50", str(ctx.exception))
        self.assertIn("n_draws >= 500", str(ctx.exception))

    def test_same_scale_with_spec_mandated_500_draws_no_longer_raises(self):
        # Samma åtgärd som (a) i uppdraget beskriver: "fler dragningar".
        # 500 är exakt den specmandaterade skalan (STEG1_N_NULL_DRAWS i
        # research/flodmarket/config.py) -- använder en snabb syntetisk
        # nolla här (inte den fulla panelpipelinen) eftersom det enda som
        # testas är att UPPLÖSNINGSGRINDEN släpper igenom vid tillräcklig
        # skala, inte pipelinens faktiska IC-utfall.
        from lib.ab_separation import run_planted_effect_check

        def fast_null_fn(seed, n_draws):
            rng = np.random.default_rng(seed + 10_000_019)
            return rng.normal(0, 0.02, n_draws)

        result = run_planted_effect_check(self._estimator_fn, fast_null_fn,
                                           n_inner_null_draws=500,
                                           seed=20260811, ic_min=0.02, null_percentile_q=0.99)
        self.assertIn("passes", result)  # ingen exception -- grinden själv släpper igenom


if __name__ == "__main__":
    unittest.main()
