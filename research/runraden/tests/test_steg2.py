import numpy as np

import config
from metrics import sharpe_ratio
from positions import build_positions
from steg2 import (
    build_oos_panel,
    k3_cost_and_twin_check,
    k4_sign_flip_and_isolation_check,
    oracle_positions,
    run_grid,
)
from synth import make_word_effect_panel


def test_dsr_n_trials_matches_spec():
    assert config.N_GRID_CELLS == 27
    assert config.DSR_N_TRIALS == 27 + 4 + 0


def test_oracle_positions_beat_or_match_any_real_signal():
    prices = make_word_effect_panel(("+", "+", "+", "+", "+"), effect=0.05, n_assets=6,
                                     n_years=3, seed=11)
    panel = build_oos_panel(prices, vol_window=20)
    df = panel.dropna(subset=["next_week_return"]).copy()
    df["sigma_60d"] = df["sigma_60d"].fillna(0.01)
    oracle_pos = oracle_positions(df, config.TARGET_GROSS_VOL, config.MAX_GROSS,
                                   config.MAX_GROSS_PER_NAME)
    # Oracle always trades in the direction of the realised outcome -> non-negative
    # PnL contribution every single week (weight and outcome share a sign).
    contrib = oracle_pos["weight"] * oracle_pos["next_week_return"]
    assert (contrib >= -1e-12).all()


def test_run_grid_produces_27_cells_with_finite_or_nan_sharpe():
    prices = make_word_effect_panel(("+", "+", "+", "+", "+"), effect=0.04, n_assets=6,
                                     n_years=4, seed=12)
    results = run_grid(prices, burn_in_years=1, cost_bp=2.0)
    assert len(results) == config.N_GRID_CELLS
    keys = {(r["kappa"], r["vol_window"], r["no_trade_band"]) for r in results}
    assert len(keys) == config.N_GRID_CELLS
    for r in results:
        assert np.isnan(r["sharpe"]) or np.isfinite(r["sharpe"])


def test_k3_and_k4_run_end_to_end_on_synthetic_grid():
    prices = make_word_effect_panel(("+", "+", "+", "+", "+"), effect=0.08, n_assets=8,
                                     n_years=5, seed=13)
    results = run_grid(prices, burn_in_years=1, cost_bp=2.0)
    primary_key = (config.PRIMARY_KAPPA, config.PRIMARY_VOL_WINDOW, config.PRIMARY_NO_TRADE_BAND)
    primary = next((r for r in results
                     if (r["kappa"], r["vol_window"], r["no_trade_band"]) == primary_key), None)
    assert primary is not None

    twin_sharpes = {"T2": 0.1, "T3": -0.1}
    k3 = k3_cost_and_twin_check(primary, twin_sharpes, prices, burn_in_years=1)
    assert set(["dsr", "stressed_sharpe", "loses_to_a_twin", "sign_flip_under_stress",
                "passed"]).issubset(k3.keys())

    k4 = k4_sign_flip_and_isolation_check(results, primary_key)
    assert "passed" in k4
