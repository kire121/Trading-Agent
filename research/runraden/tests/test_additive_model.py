import numpy as np

from additive_model import WalkForwardAdditiveModel
from metrics import pooled_ic
from synth import make_iid_panel, make_word_effect_panel
from targets import attach_targets
from words import build_asset_week_panel


def _build_panel(prices):
    panel = build_asset_week_panel(prices)
    panel = attach_targets(panel, prices, vol_window=60)
    return panel


def test_walkforward_recovers_planted_word_effect():
    trigger = ("+", "+", "+", "+", "+")
    prices = make_word_effect_panel(trigger, effect=0.06, n_assets=10, n_years=7,
                                     vol=0.01, seed=1)
    panel = _build_panel(prices)
    df_5d = panel[panel["table_id"] == "5d"].reset_index(drop=True)

    model = WalkForwardAdditiveModel(word_len=5, kappa=10.0, burn_in_years=1)
    model.fit_walkforward(df_5d)
    scored = model.score(df_5d)
    combined = df_5d.join(scored)

    # The trigger cell's shrunk effect should show up positive and be the
    # largest (or near-largest) among all seen cells in most late-sample refits.
    last_cell_map = model.cell_maps[-1]
    assert trigger in last_cell_map
    assert last_cell_map[trigger] > 0
    ranked = sorted(last_cell_map.items(), key=lambda kv: -kv[1])
    assert ranked[0][0] == trigger

    ic_full = pooled_ic(combined["full_pred"], combined["z_next"])
    ic_additive = pooled_ic(combined["additive_pred"], combined["z_next"])
    assert ic_full > ic_additive  # g adds incremental OOF predictive value


def test_walkforward_iid_panel_has_no_reliable_word_effect():
    prices = make_iid_panel(n_assets=8, n_years=6, vol=0.01, seed=2)
    panel = _build_panel(prices)
    df_5d = panel[panel["table_id"] == "5d"].reset_index(drop=True)

    model = WalkForwardAdditiveModel(word_len=5, kappa=300.0, burn_in_years=1)
    model.fit_walkforward(df_5d)
    scored = model.score(df_5d)
    combined = df_5d.join(scored)

    ic = pooled_ic(combined["ghat"], combined["z_next"])
    # No planted structure -> OOF IC of the word effect should be small in
    # magnitude (a loose bound; this is a stochastic sanity check, not an
    # exact-zero assertion).
    assert abs(ic) < 0.15


def test_design_matrix_and_scoring_shapes():
    prices = make_iid_panel(n_assets=3, n_years=4, vol=0.01, seed=3)
    panel = _build_panel(prices)
    df_5d = panel[panel["table_id"] == "5d"].reset_index(drop=True)
    model = WalkForwardAdditiveModel(word_len=5, kappa=300.0, burn_in_years=1)
    model.fit_walkforward(df_5d)
    scored = model.score(df_5d)
    assert list(scored.columns) == ["additive_pred", "ghat", "full_pred"]
    assert len(scored) == len(df_5d)
    # Rows before the first refit date should be NaN (burn-in respected).
    first_refit = model.refit_dates[0]
    pre_burn_in = df_5d["t_signal"] < first_refit
    assert scored.loc[pre_burn_in, "additive_pred"].isna().all()
