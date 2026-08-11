"""Wire the pieces together: events -> R_hat/G_t -> overlay on the frozen
TSMOM base book, plus the oracle-cap upper bound.
"""
import numpy as np
import pandas as pd

from . import config
from . import events as events_mod
from . import scheduling
from . import signal as signal_mod
from . import tsmom


def weekly_return(daily_returns: pd.Series) -> pd.Series:
    """Compound daily returns within each ISO (Mon-Fri, W-FRI) week."""
    week_period = daily_returns.index.to_period(f"W-{config.REBALANCE_WEEKDAY}")
    return (1.0 + daily_returns.fillna(0.0)).groupby(week_period).prod() - 1.0


def weekly_sharpe(weekly_returns: pd.Series) -> float:
    r = weekly_returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(52)


def apply_overlay(panel, base_weights: pd.DataFrame, g_t_daily: pd.Series, apply_costs: bool = True):
    """Scale the whole base book by G_t (Friday-updated, applied to the
    following ISO week -- same cadence as the base book's own rebalance)."""
    g_applied = scheduling.friday_lag_apply(g_t_daily.to_frame("g")).iloc[:, 0]
    tilted_weights = base_weights.mul(g_applied, axis=0)
    returns = tsmom.portfolio_returns(panel, tilted_weights, apply_costs=apply_costs)
    return tilted_weights, returns, g_applied


def run_cell(panel, q: int, tau: int, kappa: float, base_weights: pd.DataFrame,
             base_returns: pd.Series, apply_costs: bool = True):
    """One (q, tau, kappa) grid cell: events -> R_hat/G_t -> tilted book."""
    returns = panel.simple_returns()
    events, x_t = events_mod.build(returns, q=q)
    r_hat, g_t, lam_t = signal_mod.build(x_t, q=q, tau=tau, kappa=kappa)
    tilted_weights, tilted_returns, g_applied = apply_overlay(panel, base_weights, g_t, apply_costs)
    return {
        "q": q, "tau": tau, "kappa": kappa,
        "x_t": x_t, "r_hat": r_hat, "g_t": g_t, "lam_t": lam_t,
        "g_applied": g_applied,
        "tilted_weights": tilted_weights, "tilted_returns": tilted_returns,
    }


def oracle_g(g_weekly: pd.Series, base_week_return: pd.Series) -> pd.Series:
    """Rearrangement-inequality oracle: reassigns the SAME multiset of G
    values (identical empirical distribution -- "matchad fordelning") to
    weeks in the order that maximizes sum(g * r), i.e. perfect foresight
    within the primary's own realized G distribution. Costs ignored (this
    is a theoretical ceiling, not a tradeable variant)."""
    common = pd.concat([g_weekly, base_week_return], axis=1, keys=["g", "r"]).dropna()
    if len(common) < 2:
        return pd.Series(dtype=float)
    g_sorted = np.sort(common["g"].to_numpy())
    order = common["r"].sort_values().index  # ascending return order
    oracle = pd.Series(g_sorted, index=order)
    return oracle.reindex(common.index)


def oracle_cap_test(base_returns: pd.Series, g_weekly: pd.Series) -> dict:
    base_week = weekly_return(base_returns)
    g_lagged_weekly = g_weekly.shift(1)  # the multiplier a week actually trades under
    oracle_multiplier = oracle_g(g_lagged_weekly, base_week)
    common = pd.concat([base_week, oracle_multiplier], axis=1, keys=["r", "g"]).dropna()
    oracle_returns = common["r"] * common["g"]
    base_sr = weekly_sharpe(common["r"])
    oracle_sr = weekly_sharpe(oracle_returns)
    increment = oracle_sr - base_sr
    return {
        "base_sr": base_sr, "oracle_sr": oracle_sr, "increment": increment,
        "passes": bool(np.isfinite(increment) and increment >= config.ORACLE_CAP_MIN_SR_INCREMENT),
    }
