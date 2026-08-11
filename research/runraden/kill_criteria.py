"""Steg 0 (K1a/b/c) and Steg 1 (K2) kill-criteria evaluation on the IS panel.

Every threshold that the spec leaves unpinned (K1c's sign-consistency /
leave-one-cell-out / max-quarter-PnL-share cutoffs) is set to a documented
default in config.py-adjacent constants below and called out in the README
"Design choices & assumptions" section -- these are operational choices,
not part of the hypothesis pre-registration text itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from additive_model import WalkForwardAdditiveModel, design_matrix, ols_beta
from metrics import pooled_ic, oof_r2, pc1_share, sharpe_ratio, portfolio_weekly_returns
from nulls import n1_pooled_null_distribution, circular_block_bootstrap_columns
from positions import build_positions
from shrinkage import cell_shrinkage
import twins as twins_mod

# K1c operational thresholds (documented defaults, not spec-pinned)
K1C_TOP_CELL_FRACTION = 0.25       # "top |g| cells" = top quartile by |ghat|
K1C_MIN_SIGN_CONSISTENCY = 0.60    # >=60% of top cells must agree in sign across IS halves
K1C_MAX_SINGLE_CELL_IC_SHARE = 0.50  # no single top cell may explain >50% of pooled IC
K1C_MAX_QUARTER_PNL_SHARE = 0.50   # no single quarter may explain >50% of total PnL

# K1b operational threshold
K1B_PC1_NULL_PERCENTILE = 95


def fit_and_score_table(df_table: pd.DataFrame, word_len: int, kappa: float,
                         burn_in_years: int) -> tuple[WalkForwardAdditiveModel, pd.DataFrame]:
    model = WalkForwardAdditiveModel(word_len, kappa=kappa, burn_in_years=burn_in_years)
    model.fit_walkforward(df_table)
    scored = model.score(df_table)
    return model, df_table.join(scored)


def pooled_scored_panel(df_5d: pd.DataFrame, df_4d: pd.DataFrame, kappa: float,
                         burn_in_years: int) -> tuple[dict, pd.DataFrame]:
    models = {}
    parts = []
    for word_len, df_table, name in ((5, df_5d, "5d"), (4, df_4d, "4d")):
        if df_table.empty:
            continue
        model, scored = fit_and_score_table(df_table, word_len, kappa, burn_in_years)
        models[name] = model
        parts.append(scored)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return models, combined


# ---------------------------------------------------------------------------
# K1a: mandatory redundancy screen -- pooled OOF-IC of the word model vs the
# N1 (within-week permutation, full refit) null. Dead if real IC <= p95(null).
# ---------------------------------------------------------------------------
def k1a_redundancy_screen(df_5d: pd.DataFrame, df_4d: pd.DataFrame, kappa: float,
                           burn_in_years: int, n_draws: int = 200,
                           seed: int | None = None) -> dict:
    seed = config.RANDOM_SEED if seed is None else seed
    _, combined = pooled_scored_panel(df_5d, df_4d, kappa, burn_in_years)
    real_ic = pooled_ic(combined["ghat"], combined["z_next"]) if not combined.empty else np.nan

    def stat_fn(permuted_combined: pd.DataFrame) -> float:
        if permuted_combined.empty:
            return np.nan
        return pooled_ic(permuted_combined["ghat"], permuted_combined["z_next"])

    tables = [(df_5d, 5), (df_4d, 4)]
    null_ics = n1_pooled_null_distribution(tables, kappa, burn_in_years, stat_fn,
                                            n_draws=n_draws, seed=seed)
    p95 = float(np.nanpercentile(null_ics, 95))
    passed = bool(not np.isnan(real_ic) and real_ic > p95)
    return {
        "real_ic": real_ic, "null_p95": p95, "null_mean": float(np.nanmean(null_ics)),
        "null_std": float(np.nanstd(null_ics)), "n_draws": n_draws, "passed": passed,
    }


# ---------------------------------------------------------------------------
# K1b: PC1 share of the ghat panel vs a common-factor-decoupled block null.
# If the real PC1 share is anomalously high (> null p95), the interaction
# effect looks like disguised market timing, and the pipeline may only
# continue if the strategy beats T4 (the market-timing twin) net of costs.
# ---------------------------------------------------------------------------
def _ghat_matrix(combined: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    piv = combined.pivot_table(index="execution_date", columns="asset", values="ghat", aggfunc="mean")
    return piv.to_numpy(dtype=float), list(piv.index), list(piv.columns)


def k1b_common_factor_screen(combined: pd.DataFrame, positions_real: pd.DataFrame,
                              positions_t4: pd.DataFrame, cost_bp: float,
                              n_draws: int = 200, block_size: int = 8,
                              seed: int | None = None) -> dict:
    seed = config.RANDOM_SEED if seed is None else seed
    mat, _, _ = _ghat_matrix(combined)
    valid_cols = ~np.all(np.isnan(mat), axis=0)
    mat = mat[:, valid_cols]
    real_share = pc1_share(mat)

    rng = np.random.default_rng(seed)
    null_shares = np.full(n_draws, np.nan)
    for d in range(n_draws):
        resampled = circular_block_bootstrap_columns(np.nan_to_num(mat, nan=0.0), block_size, rng)
        null_shares[d] = pc1_share(resampled)
    p95 = float(np.nanpercentile(null_shares, K1B_PC1_NULL_PERCENTILE))
    anomalous = bool(not np.isnan(real_share) and real_share > p95)

    real_net = portfolio_weekly_returns(positions_real, cost_bp=cost_bp)
    t4_net = portfolio_weekly_returns(positions_t4, cost_bp=cost_bp)
    sharpe_real = sharpe_ratio(real_net)
    sharpe_t4 = sharpe_ratio(t4_net)
    beats_t4 = bool(not np.isnan(sharpe_real) and not np.isnan(sharpe_t4) and sharpe_real > sharpe_t4)

    passed = bool((not anomalous) or beats_t4)
    return {
        "real_pc1_share": real_share, "null_p95": p95, "anomalous_common_factor": anomalous,
        "sharpe_real": sharpe_real, "sharpe_t4": sharpe_t4, "beats_t4": beats_t4,
        "n_draws": n_draws, "passed": passed,
    }


# ---------------------------------------------------------------------------
# K1c: ubiquity translation -- sign consistency across IS halves,
# leave-one-cell-out concentration, and max-quarter PnL share.
# ---------------------------------------------------------------------------
def _static_cell_map(df_table: pd.DataFrame, word_len: int, kappa: float) -> dict:
    df = df_table.dropna(subset=["z_next"])
    if df.empty:
        return {}
    X = design_matrix(df["word"].tolist(), word_len)
    y = df["z_next"].to_numpy(dtype=float)
    beta = ols_beta(X, y)
    resid = y - X @ beta
    return cell_shrinkage(df["word"].tolist(), resid, kappa)


def k1c_sign_consistency(df_5d: pd.DataFrame, df_4d: pd.DataFrame, kappa: float) -> dict:
    results = {}
    for word_len, df_table, name in ((5, df_5d, "5d"), (4, df_4d, "4d")):
        if df_table.empty or df_table["z_next"].dropna().empty:
            continue
        df = df_table.dropna(subset=["z_next"]).sort_values("t_target_end")
        mid = df["t_target_end"].iloc[len(df) // 2]
        half1, half2 = df[df["t_target_end"] < mid], df[df["t_target_end"] >= mid]
        map1, map2 = _static_cell_map(half1, word_len, kappa), _static_cell_map(half2, word_len, kappa)
        common = sorted(set(map1) & set(map2), key=lambda w: -abs(map1[w]))
        k = max(1, int(len(common) * K1C_TOP_CELL_FRACTION))
        top = common[:k]
        if not top:
            continue
        agree = sum(1 for w in top if np.sign(map1[w]) == np.sign(map2[w]) and map1[w] != 0)
        results[name] = {"n_top_cells": len(top), "sign_consistency": agree / len(top)}
    if not results:
        return {"passed": False, "per_table": {}}
    passed = all(r["sign_consistency"] >= K1C_MIN_SIGN_CONSISTENCY for r in results.values())
    return {"per_table": results, "passed": passed}


def k1c_leave_one_cell_out(combined: pd.DataFrame) -> dict:
    df = combined.dropna(subset=["ghat", "z_next"])
    if df.empty:
        return {"passed": False, "max_single_cell_share": np.nan}
    full_ic = pooled_ic(df["ghat"], df["z_next"])
    cell_abs_g = df.groupby("word")["ghat"].apply(lambda s: s.abs().mean())
    k = max(1, int(len(cell_abs_g) * K1C_TOP_CELL_FRACTION))
    top_cells = cell_abs_g.sort_values(ascending=False).index[:k]

    if np.isnan(full_ic) or full_ic == 0:
        return {"passed": False, "max_single_cell_share": np.nan, "full_ic": full_ic}

    max_share = 0.0
    for w in top_cells:
        sub = df[df["word"] != w]
        ic_wo = pooled_ic(sub["ghat"], sub["z_next"])
        if np.isnan(ic_wo):
            continue
        drop = full_ic - ic_wo
        share = drop / full_ic if full_ic != 0 else np.nan
        if not np.isnan(share):
            max_share = max(max_share, share)
    passed = bool(max_share <= K1C_MAX_SINGLE_CELL_IC_SHARE)
    return {"passed": passed, "max_single_cell_share": max_share, "full_ic": full_ic}


def k1c_max_quarter_pnl_share(positions_real: pd.DataFrame, cost_bp: float) -> dict:
    net = portfolio_weekly_returns(positions_real, cost_bp=cost_bp)
    if net.empty:
        return {"passed": False, "max_quarter_share": np.nan}
    q = net.groupby(net.index.to_period("Q")).sum()
    total = q.sum()
    if total == 0 or np.isnan(total):
        return {"passed": False, "max_quarter_share": np.nan}
    shares = (q / total) if total > 0 else (q / total)
    max_share = float(shares.abs().max())
    passed = bool(max_share <= K1C_MAX_QUARTER_PNL_SHARE)
    return {"passed": passed, "max_quarter_share": max_share, "n_quarters": len(q)}


def k1c_ubiquity(df_5d: pd.DataFrame, df_4d: pd.DataFrame, combined: pd.DataFrame,
                  positions_real: pd.DataFrame, kappa: float, cost_bp: float) -> dict:
    sign_check = k1c_sign_consistency(df_5d, df_4d, kappa)
    loco_check = k1c_leave_one_cell_out(combined)
    quarter_check = k1c_max_quarter_pnl_share(positions_real, cost_bp)
    passed = bool(sign_check["passed"] and loco_check["passed"] and quarter_check["passed"])
    return {
        "sign_consistency": sign_check, "leave_one_cell_out": loco_check,
        "max_quarter_pnl_share": quarter_check, "passed": passed,
    }


# ---------------------------------------------------------------------------
# K2: OOF incremental IC (and R^2, as a deviance proxy) for g beyond the
# additive model. Dead if incremental IC <= 0.
# ---------------------------------------------------------------------------
def k2_incremental_value(combined: pd.DataFrame) -> dict:
    df = combined.dropna(subset=["additive_pred", "ghat", "z_next"])
    ic_additive = pooled_ic(df["additive_pred"], df["z_next"])
    ic_full = pooled_ic(df["full_pred"], df["z_next"])
    r2_additive = oof_r2(df["additive_pred"], df["z_next"])
    r2_full = oof_r2(df["full_pred"], df["z_next"])
    incremental_ic = ic_full - ic_additive if not (np.isnan(ic_full) or np.isnan(ic_additive)) else np.nan
    incremental_r2 = r2_full - r2_additive if not (np.isnan(r2_full) or np.isnan(r2_additive)) else np.nan
    passed = bool(not np.isnan(incremental_ic) and incremental_ic > 0)
    return {
        "ic_additive": ic_additive, "ic_full": ic_full, "incremental_ic": incremental_ic,
        "r2_additive": r2_additive, "r2_full": r2_full, "incremental_r2": incremental_r2,
        "passed": passed,
    }
