"""Grid = q in {90,95} x tau in {10,21} x kappa in {1,2} (8 cells), plus a
+-50% Omori-kernel-exponent robustness check OUTSIDE the grid (not a DSR
trial -- a sensitivity check on the one frozen, non-gridded parameter).
"""
import numpy as np
import pandas as pd

from . import backtest
from . import config
from . import metrics


def run_grid(panel, base_weights: pd.DataFrame, base_returns: pd.Series, is_start, is_end,
             oos_start, oos_end, apply_costs: bool = True) -> pd.DataFrame:
    rows = []
    for cell in config.GRID:
        result = backtest.run_cell(panel, cell.q, cell.tau, cell.kappa, base_weights, base_returns, apply_costs)
        tilted = result["tilted_returns"]
        is_ret = tilted.loc[is_start:is_end]
        oos_ret = tilted.loc[oos_start:oos_end]
        rows.append({
            "q": cell.q, "tau": cell.tau, "kappa": cell.kappa,
            "is_sharpe": metrics.sharpe(is_ret),
            "oos_sharpe": metrics.sharpe(oos_ret),
            "is_ann_return": metrics.ann_return(is_ret),
            "oos_ann_return": metrics.ann_return(oos_ret),
        })
    return pd.DataFrame(rows)


def sign_stability(grid_df: pd.DataFrame, col: str = "is_sharpe") -> float:
    signs = np.sign(grid_df[col].dropna())
    if not len(signs):
        return np.nan
    return float((signs == signs.mode().iloc[0]).mean())


def dsr_from_grid(daily_returns: pd.Series, grid_df: pd.DataFrame, sharpe_col: str = "is_sharpe") -> dict:
    """Feed the grid's per-cell (per-period) Sharpes as the DSR trial pool,
    same convention as omori/grid.py::dsr_from_grid."""
    n_obs = daily_returns.dropna().shape[0]
    sr_hat = metrics.sharpe(daily_returns) / np.sqrt(config.TRADING_DAYS_YEAR)  # per-period scale
    sr_trials = (grid_df[sharpe_col].dropna() / np.sqrt(config.TRADING_DAYS_YEAR)).to_numpy()
    if len(sr_trials) < 2:
        return {"dsr_prob": np.nan, "dsr_excess": np.nan, "expected_max_sr": np.nan}
    return metrics.deflated_sharpe_ratio(sr_hat, n_obs, sr_trials)


def kernel_robustness_check(panel, base_weights, base_returns, cell, is_start, is_end,
                             pct: float = config.OMORI_KERNEL_ROBUSTNESS_PCT, apply_costs: bool = True):
    """Perturb the frozen Omori kernel exponent p by +-pct, rerun the
    default cell, report IS Sharpe under each perturbation. Outside the
    main grid -- not a DSR trial."""
    from . import events as events_mod
    from . import signal as signal_mod
    returns = panel.simple_returns()
    events, x_t = events_mod.build(returns, q=cell.q)
    out = {}
    for label, p in (
        ("p_low", config.OMORI_KERNEL_P * (1 - pct)),
        ("p_base", config.OMORI_KERNEL_P),
        ("p_high", config.OMORI_KERNEL_P * (1 + pct)),
    ):
        r_hat, g_t, _ = signal_mod.build(x_t, q=cell.q, tau=cell.tau, kappa=cell.kappa, p=p)
        _, tilted_returns, _ = backtest.apply_overlay(panel, base_weights, g_t, apply_costs)
        out[label] = {"p": p, "is_sharpe": metrics.sharpe(tilted_returns.loc[is_start:is_end])}
    return out
