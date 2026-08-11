"""Deterministic, hand-constructed-input tests for kill_criteria.py's
pass/fail direction logic (K1a is instead covered by an integration test
against the full walk-forward pipeline, since its null needs a real refit).
"""
import numpy as np
import pandas as pd

import config
from kill_criteria import (
    k1a_redundancy_screen,
    k1b_common_factor_screen,
    k1c_leave_one_cell_out,
    k1c_max_quarter_pnl_share,
    k1c_sign_consistency,
    k2_incremental_value,
)
from synth import make_iid_panel, make_word_effect_panel
from targets import attach_targets
from words import build_asset_week_panel


# --- K1a: full integration, planted effect should clear the N1 null -------
def test_k1a_passes_with_a_strong_planted_effect():
    # Deliberately a *mixed*-sign trigger word: an all-same-sign word (e.g.
    # "+++++") is invariant under within-week letter permutation (nothing to
    # shuffle among identical elements), which would make N1 a no-op for
    # exactly the rows that matter and silently invalidate the null.
    trigger = ("+", "+", "-", "+", "-")
    prices = make_word_effect_panel(trigger, effect=0.06, n_assets=10, n_years=6, seed=21)
    panel = build_asset_week_panel(prices)
    panel = attach_targets(panel, prices, vol_window=60)
    df_5d = panel[panel["table_id"] == "5d"].reset_index(drop=True)
    df_4d = panel[panel["table_id"] == "4d"].reset_index(drop=True)

    out = k1a_redundancy_screen(df_5d, df_4d, kappa=10.0, burn_in_years=1, n_draws=25, seed=1)
    assert out["real_ic"] > out["null_p95"]
    assert out["passed"] is True


# --- K2: incremental IC direction ------------------------------------------
def _combined(additive_pred, ghat, z_next, word=None):
    n = len(z_next)
    return pd.DataFrame({
        "additive_pred": additive_pred, "ghat": ghat, "full_pred": additive_pred + ghat,
        "z_next": z_next, "word": word if word is not None else ["W"] * n,
    })


def test_k2_passes_when_ghat_adds_real_predictive_value():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(0, 1, n)
    additive_pred = 0.05 * z + rng.normal(0, 1, n)   # weak additive signal
    ghat = 0.8 * z + rng.normal(0, 0.2, n)           # strong extra signal in ghat
    out = k2_incremental_value(_combined(additive_pred, ghat, z))
    assert out["incremental_ic"] > 0
    assert out["passed"] is True


def test_k2_fails_when_ghat_is_pure_noise():
    rng = np.random.default_rng(1)
    n = 2000
    z = rng.normal(0, 1, n)
    additive_pred = 0.3 * z + rng.normal(0, 1, n)
    ghat = rng.normal(0, 1, n)  # uncorrelated with z
    out = k2_incremental_value(_combined(additive_pred, ghat, z))
    assert out["passed"] is False
    assert out["incremental_ic"] <= 0


# --- K1c: leave-one-cell-out concentration ---------------------------------
def test_leave_one_cell_out_flags_concentration_in_a_single_cell():
    rng = np.random.default_rng(2)
    n_per_cell = 200
    words, ghat, z = [], [], []
    # One "hot" cell with both a strong ghat *magnitude* (the selection
    # criterion the implementation uses for "top |g| cells") and a strong
    # real relationship to z...
    hot_ghat = rng.normal(0, 5, n_per_cell)
    hot_z = 0.9 * hot_ghat + rng.normal(0, 0.5, n_per_cell)
    words += ["HOT"] * n_per_cell
    ghat += list(hot_ghat)
    z += list(hot_z)
    # ...and several "cold" cells with no relationship at all.
    for c in range(6):
        cold_ghat = rng.normal(0, 1, n_per_cell)
        cold_z = rng.normal(0, 1, n_per_cell)
        words += [f"COLD{c}"] * n_per_cell
        ghat += list(cold_ghat)
        z += list(cold_z)
    out = k1c_leave_one_cell_out(_combined(np.zeros(len(z)), np.array(ghat), np.array(z), word=words))
    assert out["max_single_cell_share"] > 0.5
    assert out["passed"] is False


def test_leave_one_cell_out_passes_when_effect_is_spread_out():
    rng = np.random.default_rng(3)
    n_per_cell = 200
    words, ghat, z = [], [], []
    for c in range(8):
        g = rng.normal(0, 1, n_per_cell)
        zz = 0.5 * g + rng.normal(0, 0.9, n_per_cell)  # same modest relationship, every cell
        words += [f"CELL{c}"] * n_per_cell
        ghat += list(g)
        z += list(zz)
    out = k1c_leave_one_cell_out(_combined(np.zeros(len(z)), np.array(ghat), np.array(z), word=words))
    assert out["max_single_cell_share"] < 0.5
    assert out["passed"] is True


# --- K1c: max-quarter PnL share --------------------------------------------
def _positions(weights, next_returns, dates):
    return pd.DataFrame({
        "asset": ["A"] * len(dates), "t_signal": dates, "execution_date": dates,
        "weight": weights, "next_week_return": next_returns,
    })


def test_max_quarter_pnl_share_flags_a_single_dominant_quarter():
    dates = pd.date_range("2020-01-06", periods=52, freq="7D")
    returns = np.full(52, 0.0001)
    returns[10] = 5.0  # one enormous week dwarfs everything else
    out = k1c_max_quarter_pnl_share(_positions(np.ones(52), returns, dates), cost_bp=0.0)
    assert out["max_quarter_share"] > 0.5
    assert out["passed"] is False


def test_max_quarter_pnl_share_passes_when_spread_evenly():
    dates = pd.date_range("2020-01-06", periods=52, freq="7D")
    returns = np.full(52, 0.001)  # identical contribution every week -> spread across quarters
    out = k1c_max_quarter_pnl_share(_positions(np.ones(52), returns, dates), cost_bp=0.0)
    assert out["max_quarter_share"] <= 0.5
    assert out["passed"] is True


# --- K1c: sign consistency across IS halves --------------------------------
def _word_table(words, z_next, t_target_end):
    return pd.DataFrame({"word": words, "z_next": z_next, "t_target_end": t_target_end})


TARGET_WORD = ("+", "-", "+", "-", "+")


def _random_word(rng):
    # A rich (32-word) vocabulary, not just a two-word A/B split: with only
    # two distinct words the additive model's 6 free parameters (alpha +
    # 5 day-position betas) can fit both cell means *exactly* via the
    # additive part alone (2 unique design-matrix rows, 6 parameters is
    # under-determined), leaving ~0 residual and masking any true
    # word-cell effect regardless of kappa. A realistic spread of words
    # makes the additive fit non-degenerate, like the real IS panel.
    return tuple(rng.choice(["+", "-"], size=5))


def _sign_consistency_panel(seed, flip_second_half):
    rng = np.random.default_rng(seed)
    n = 3000
    dates = pd.date_range("2010-01-01", periods=n, freq="7D")
    words, z = [], []
    for i, d in enumerate(dates):
        w = _random_word(rng)
        first_half = i < n // 2
        if w == TARGET_WORD:
            effect = 0.5 if (first_half or not flip_second_half) else -0.5
        else:
            effect = 0.0
        words.append(w)
        z.append(effect + rng.normal(0, 0.3))
    return _word_table(words, z, dates)


def test_sign_consistency_passes_when_top_cell_sign_is_stable():
    tbl = _sign_consistency_panel(seed=10, flip_second_half=False)
    empty_4d = pd.DataFrame(columns=["word", "z_next", "t_target_end"])
    out = k1c_sign_consistency(tbl, empty_4d, kappa=10.0)
    assert out["passed"] is True


def test_sign_consistency_fails_when_sign_flips_between_halves():
    tbl = _sign_consistency_panel(seed=11, flip_second_half=True)
    empty_4d = pd.DataFrame(columns=["word", "z_next", "t_target_end"])
    out = k1c_sign_consistency(tbl, empty_4d, kappa=10.0)
    assert out["passed"] is False


# --- K1b: common-factor screen ---------------------------------------------
def _k1b_inputs(ghat_matrix, dates, assets, sharpe_real, sharpe_t4):
    rows = []
    for i, d in enumerate(dates):
        for j, a in enumerate(assets):
            rows.append({"execution_date": d, "asset": a, "ghat": ghat_matrix[i, j],
                         "t_signal": d, "next_week_return": 0.0})
    combined = pd.DataFrame(rows)

    def _fake_positions(sharpe_target):
        n = len(dates)
        rng = np.random.default_rng(abs(hash(sharpe_target)) % (2**32))
        rets = rng.normal(sharpe_target / np.sqrt(52), 1.0, n)
        return pd.DataFrame({"asset": ["X"] * n, "t_signal": dates, "execution_date": dates,
                              "weight": np.ones(n), "next_week_return": rets})

    return combined, _fake_positions(sharpe_real), _fake_positions(sharpe_t4)


def test_k1b_not_anomalous_passes_regardless_of_t4():
    rng = np.random.default_rng(6)
    dates = pd.date_range("2015-01-05", periods=100, freq="7D")
    assets = [f"A{i}" for i in range(10)]
    ghat_matrix = rng.normal(0, 1, (100, 10))  # independent columns, no common factor
    combined, pos_real, pos_t4 = _k1b_inputs(ghat_matrix, dates, assets, sharpe_real=-1.0, sharpe_t4=2.0)
    out = k1b_common_factor_screen(combined, pos_real, pos_t4, cost_bp=0.0, n_draws=30, seed=1)
    assert out["passed"] is True


def test_k1b_anomalous_and_loses_to_t4_fails():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-05", periods=100, freq="7D")
    assets = [f"A{i}" for i in range(10)]
    factor = rng.normal(0, 1, 100)
    ghat_matrix = np.outer(factor, np.ones(10)) + rng.normal(0, 0.01, (100, 10))  # dominant common factor
    combined, pos_real, pos_t4 = _k1b_inputs(ghat_matrix, dates, assets, sharpe_real=-2.0, sharpe_t4=3.0)
    out = k1b_common_factor_screen(combined, pos_real, pos_t4, cost_bp=0.0, n_draws=30, seed=1)
    assert out["anomalous_common_factor"] is True
    assert out["beats_t4"] is False
    assert out["passed"] is False
