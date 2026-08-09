"""Robustness grid: W x threshold x estimator, with period breakdowns for
the IS/OOS split and the three pre-registered sub-periods."""

import numpy as np
import pandas as pd

from . import config, signal, backtest, variants

SUBPERIODS = config.SUBPERIODS


def slice_performance(portfolio_return, turnover, start, end):
    r = portfolio_return.loc[start:end]
    if len(r) < 20:
        return {"sharpe": np.nan, "cagr": np.nan, "max_drawdown": np.nan, "n_days": len(r)}
    equity = (1.0 + r).cumprod()
    t = turnover.loc[start:end] if turnover is not None else None
    return backtest.performance_summary(r, equity, t)


def evaluate_config(px, ret, W, threshold, estimator, z_panel_daily=None):
    res = variants.irreversibility_weekly_weights(
        px, ret, W=W, threshold=threshold, estimator=estimator, z_panel_daily=z_panel_daily
    )
    bt = backtest.run_backtest(px, ret, res["weekly_weights"])
    row = {"W": W, "threshold": threshold, "estimator": estimator}
    row["full"] = backtest.performance_summary(bt["portfolio_return"], bt["equity"], bt["turnover"])
    row["is"] = slice_performance(bt["portfolio_return"], bt["turnover"], config.IS_START, config.IS_END)
    row["oos"] = slice_performance(bt["portfolio_return"], bt["turnover"], config.OOS_START, None)
    row["subperiods"] = {
        name: slice_performance(bt["portfolio_return"], bt["turnover"], s, e)
        for name, (s, e) in SUBPERIODS.items()
    }
    return row, bt, res


def run_full_grid(px, ret, W_grid=config.W_GRID, threshold_grid=config.Z_THRESHOLD_GRID,
                   estimators=("hvg", "ordinal", "psi")):
    """Builds each (W, estimator) irreversibility panel ONCE (the expensive
    step) and reuses it across the threshold grid (cheap: regime
    classification + backtest only)."""
    rows = []
    panel_cache = {}
    bt_cache = {}
    for W in W_grid:
        for est in estimators:
            i_panel = signal.compute_irreversibility_panel(ret, W=W, estimator=est)
            z_panel = signal.rolling_zscore(i_panel, history=config.Z_HISTORY)
            panel_cache[(W, est)] = z_panel
            for thr in threshold_grid:
                row, bt, res = evaluate_config(px, ret, W, thr, est, z_panel_daily=z_panel)
                rows.append(row)
                bt_cache[(W, thr, est)] = (bt, res)
    return rows, panel_cache, bt_cache


def flatten_grid_rows(rows):
    flat = []
    for row in rows:
        base = {"W": row["W"], "threshold": row["threshold"], "estimator": row["estimator"]}
        for period_key in ("full", "is", "oos"):
            for stat_key, val in row[period_key].items():
                base[f"{period_key}_{stat_key}"] = val
        for sp_name, sp_stats in row["subperiods"].items():
            for stat_key, val in sp_stats.items():
                base[f"sp_{sp_name}_{stat_key}"] = val
        flat.append(base)
    return pd.DataFrame(flat)


def sign_consistency_check(grid_df):
    """For each (W, threshold): does the sign of the *IS* Sharpe agree in
    >= 2 of 3 estimators, AND does the majority-sign estimator's Sharpe keep
    the same sign across all three pre-registered sub-periods?
    """
    out = []
    for (W, thr), sub in grid_df.groupby(["W", "threshold"]):
        sub = sub.set_index("estimator")
        signs = np.sign(sub["is_sharpe"])
        maj_sign = signs.mode()
        maj_sign = maj_sign.iloc[0] if len(maj_sign) else np.nan
        agree = int((signs == maj_sign).sum())
        sp_cols = [c for c in sub.columns if c.startswith("sp_") and c.endswith("_sharpe")]
        sp_signs_consistent = {}
        for est in sub.index:
            vals = sub.loc[est, sp_cols]
            s = np.sign(vals.astype(float))
            sp_signs_consistent[est] = bool((s.dropna() == maj_sign).all()) if maj_sign == maj_sign else False
        out.append({
            "W": W, "threshold": thr, "majority_sign": maj_sign,
            "n_estimators_agreeing": agree, "passes_2of3": agree >= 2,
            "subperiod_consistent_by_estimator": sp_signs_consistent,
            "passes_subperiod_consistency": any(sp_signs_consistent.values()),
        })
    return pd.DataFrame(out)
