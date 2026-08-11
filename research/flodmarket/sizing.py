"""Iterative-k portfolio sizing (spec SS4: "k lost iterativt mot 10%
arsvolmal med bruttotak sum|w| <= 200% -- exakt samma kalibreringsvag som
Smittotalets basbok").

Proveniens: research/smittotalet/tsmom.py::solve_k_for_target_vol +
apply_gross_cap, branch claude/smittotalet-portfolio-overlay-0bl1sh, commit
a67df1b (itself following "Dammluckans fixade kalibreringsvag":
research/dammluckan/backtest.py::_solve_k_for_target_vol -- a one-shot
linear rescale is only exact when the gross cap never binds; since the cap
DOES bind on some weeks, realized portfolio vol is a concave, saturating
function of k, so k is solved by fixed-point iteration against realized IS
vol, then frozen).

ADAPTED (not verbatim): the original operated on a strategy-specific
`Panel` object with `.simple_returns()`/`.adv()` methods; here it takes
plain wide DataFrames (date-index x ticker-columns) directly, since
Flodmarket's raw signal (raw_i = g_i / sigma_hat_i, spec SS4 -- NOT
Smittotalet's own TSMOM raw_signal) is computed upstream in signal.py. The
iterative-k ENGINE ITSELF (the fixed-point loop, the gross-cap-before-any-
rescale discipline) is unchanged.
"""
import numpy as np
import pandas as pd

from . import config
from . import costs


def apply_gross_cap(raw_position: pd.DataFrame, k: float, gross_cap: float = config.GROSS_CAP) -> pd.DataFrame:
    """w = k*raw, then scaled down (never up) so that sum|w| <= gross_cap on
    every date -- the cap is applied AFTER scaling by k, never re-rescaled
    afterward (Dammluckans bugglarning: no one-shot rescale on top of an
    already-binding cap)."""
    scaled = raw_position * k
    gross = scaled.abs().sum(axis=1)
    scale_down = (gross_cap / gross).clip(upper=1.0).fillna(1.0)
    return scaled.mul(scale_down, axis=0)


def turnover_cost_returns(weights: pd.DataFrame, adv_dollars: pd.DataFrame) -> pd.Series:
    """Round-trip ADV-bucket cost drag, charged on every rebalance (weekly
    here, spec SS4)."""
    delta = weights.diff().abs()
    cost_frac = costs.cost_fraction(adv_dollars)
    per_name_cost = delta * cost_frac
    return per_name_cost.sum(axis=1, skipna=True).reindex(weights.index).fillna(0.0)


def portfolio_returns(weights: pd.DataFrame, simple_returns: pd.DataFrame,
                       adv_dollars: pd.DataFrame = None, apply_costs: bool = True) -> pd.Series:
    gross_ret = (weights * simple_returns).sum(axis=1, skipna=True)
    if apply_costs and adv_dollars is not None:
        gross_ret = gross_ret - turnover_cost_returns(weights, adv_dollars)
    return gross_ret


def solve_k_for_target_vol(raw_position: pd.DataFrame, simple_returns: pd.DataFrame,
                            is_start: str, is_end: str,
                            target_vol: float = config.PORTFOLIO_VOL_TARGET,
                            gross_cap: float = config.GROSS_CAP,
                            periods_per_year: int = config.TRADING_DAYS_YEAR,
                            k0: float = None,
                            max_iter: int = config.VOL_TARGET_SOLVE_MAX_ITER,
                            tol: float = config.VOL_TARGET_SOLVE_TOL) -> float:
    """Fixed-point solve for the scalar k such that the gross-capped,
    k-scaled book realizes `target_vol` annualized vol over [is_start,
    is_end], WITHOUT transaction costs in the vol calc itself (costs affect
    realized returns, not the vol-targeting calibration, matching
    Smittotalet's own convention)."""
    if k0 is None:
        r0 = (raw_position * simple_returns).sum(axis=1, skipna=True).loc[is_start:is_end]
        vol0 = r0.std() * np.sqrt(periods_per_year)
        k0 = target_vol / vol0 if vol0 and vol0 > 0 else 1.0
    k = k0
    for _ in range(max_iter):
        weights = apply_gross_cap(raw_position, k, gross_cap)
        rets = portfolio_returns(weights, simple_returns, apply_costs=False).loc[is_start:is_end]
        vol = rets.std() * np.sqrt(periods_per_year)
        if not vol or vol <= 0 or np.isnan(vol):
            break
        if abs(vol - target_vol) < tol:
            break
        k = k * (target_vol / vol)
    return k


def build_book(raw_position: pd.DataFrame, simple_returns: pd.DataFrame, k: float,
                adv_dollars: pd.DataFrame = None, gross_cap: float = config.GROSS_CAP,
                apply_costs: bool = True):
    """Returns (weights, daily_returns) for the frozen-k book."""
    weights = apply_gross_cap(raw_position, k, gross_cap)
    rets = portfolio_returns(weights, simple_returns, adv_dollars, apply_costs=apply_costs)
    return weights, rets
