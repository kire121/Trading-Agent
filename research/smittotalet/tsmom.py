"""Base engine: 12-month sign TSMOM, weekly rebalance, portfolio-level vol
targeting to 10% annualized, 200% gross cap.

"Repots dokumenterade TSMOM-proxy (Vindkastets levande komponent)": the
signal itself (sign(P_t/P_{t-252}-1), inverse-20d-vol sizing) is Vindkastet's
own documented academic proxy (research/vindkastet REPORT.md), used there
un-gated, daily-rebalanced, 100%-gross. Here it is promoted to a real base
book: weekly (Friday) rebalance, held one ISO week, portfolio-level vol
targeting, and a 200% gross cap -- "protokollregeln om portfoljniva-
volskalning ligger alltsaa i basen, fore overlagget".

The scaling constant k (raw signal -> target-vol weights) is solved
iteratively, not in one linear step, following Dammluckan's own documented
fix (research/dammluckan/backtest.py:_solve_k_for_target_vol): a one-shot
linear rescale is only exact when the gross cap never binds. Because the
200% cap binds on some weeks and not others, realized portfolio vol is a
concave, saturating function of k, not linear in it -- so k is calibrated by
fixed-point iteration against REALIZED IS vol, then frozen for OOS (the
"Dammluckan-fixade kalibreringsvagen" applied here).
"""
import numpy as np
import pandas as pd

from . import config
from . import costs
from . import scheduling


def raw_signal(panel) -> pd.DataFrame:
    """sign(P_t/P_{t-252}-1) / vol20_t, daily, no lag applied yet."""
    adj = panel.adjusted_close
    mom = np.sign(adj / adj.shift(config.TSMOM_LOOKBACK) - 1.0)
    daily_ret = adj.pct_change()
    vol20 = daily_ret.rolling(config.TSMOM_VOL_LOOKBACK).std()
    return mom / vol20


def weekly_rebalanced_position(panel) -> pd.DataFrame:
    """Sample raw_signal at each Friday (week-end) close, hold constant for
    the FOLLOWING ISO week (1-week lag -- no look-ahead)."""
    return scheduling.friday_lag_apply(raw_signal(panel))


def apply_gross_cap(daily_position: pd.DataFrame, k: float, gross_cap: float = config.GROSS_CAP) -> pd.DataFrame:
    scaled = daily_position * k
    gross = scaled.abs().sum(axis=1)
    scale_down = (gross_cap / gross).clip(upper=1.0).fillna(1.0)
    return scaled.mul(scale_down, axis=0)


def _turnover_cost_returns(panel, weights: pd.DataFrame) -> pd.Series:
    adv = panel.adv()
    delta = weights.diff().abs()
    cost_frac = costs.cost_fraction(adv)
    per_name_cost = delta * cost_frac
    return per_name_cost.sum(axis=1, skipna=True).reindex(weights.index).fillna(0.0)


def portfolio_returns(panel, weights: pd.DataFrame, apply_costs: bool = True) -> pd.Series:
    simple_ret = panel.simple_returns()
    gross_ret = (weights * simple_ret).sum(axis=1, skipna=True)
    if apply_costs:
        gross_ret = gross_ret - _turnover_cost_returns(panel, weights)
    return gross_ret


def solve_k_for_target_vol(panel, is_start, is_end, target_vol: float = config.PORTFOLIO_VOL_TARGET,
                            gross_cap: float = config.GROSS_CAP, k0: float | None = None,
                            max_iter: int = config.VOL_TARGET_SOLVE_MAX_ITER,
                            tol: float = config.VOL_TARGET_SOLVE_TOL) -> float:
    raw = weekly_rebalanced_position(panel)
    if k0 is None:
        unscaled = raw.mul(1.0)
        r0 = (unscaled * panel.simple_returns()).sum(axis=1, skipna=True).loc[is_start:is_end]
        vol0 = r0.std() * np.sqrt(config.TRADING_DAYS_YEAR)
        k0 = target_vol / vol0 if vol0 and vol0 > 0 else 1.0
    k = k0
    for _ in range(max_iter):
        weights = apply_gross_cap(raw, k, gross_cap)
        rets = portfolio_returns(panel, weights, apply_costs=False).loc[is_start:is_end]
        vol = rets.std() * np.sqrt(config.TRADING_DAYS_YEAR)
        if not vol or vol <= 0:
            break
        if abs(vol - target_vol) < tol:
            break
        k = k * (target_vol / vol)
    return k


def build_base_book(panel, k: float, gross_cap: float = config.GROSS_CAP, apply_costs: bool = True):
    """Returns (weights, daily_returns) for the frozen-k base TSMOM book."""
    raw = weekly_rebalanced_position(panel)
    weights = apply_gross_cap(raw, k, gross_cap)
    rets = portfolio_returns(panel, weights, apply_costs=apply_costs)
    return weights, rets
