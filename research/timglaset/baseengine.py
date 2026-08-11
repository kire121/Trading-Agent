"""Portfolio construction: T0's verbatim base book, and the shared clock-book
builder used for the primary signal + T1/T2/T3/grid cells.

Generic mechanics (apply_gross_cap, portfolio_returns, _turnover_cost_returns,
solve_k_for_target_vol) ported from research/smittotalet/tsmom.py, branch
claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b -- generalized
here to accept an arbitrary already-computed `raw_weights` input instead of
hardcoding TSMOM's own raw_signal(), so the identical iterative
fixed-point-k mechanism (Dammluckan's calibration-path pattern, per that
module's own docstring) can drive BOTH T0's TSMOM book and Timglaset's own
op-clock book, per spec §5's "exakt samma mekanism som Smittotalets basbok".
This mirrors exactly how lib/twins.py::twin_is_alive was itself generalized
from a hardcoded-column-name original -- not a silent divergent copy.

T0's raw_signal (12-month sign momentum, rolling-20d, NOT EWMA, std) is kept
byte-faithful to smittotalet/tsmom.py because T0 must reproduce that
branch's own published number (IS 2004-2017 net Sharpe 0.533) exactly (spec
§7 T0 liveness assertion) -- any generalization there would risk silently
drifting from the number T0 exists to reproduce.
"""
import numpy as np
import pandas as pd

from . import config
from . import costs
from . import opclock
from . import scheduling


# ---------------------------------------------------------------------------
# Generic portfolio mechanics (shared by T0 and the clock-book builder)
# ---------------------------------------------------------------------------
def apply_gross_cap(daily_position: pd.DataFrame, k: float, gross_cap: float = None) -> pd.DataFrame:
    gross_cap = config.GROSS_CAP if gross_cap is None else gross_cap
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


def solve_k_for_target_vol(panel, raw_weights: pd.DataFrame, is_start, is_end,
                            target_vol: float = None, gross_cap: float = None, k0: float = None,
                            max_iter: int = None, tol: float = None) -> float:
    """Fixed-point solve for the scaling constant k such that the gross-capped
    book realizes `target_vol` annualized IS volatility. A one-shot linear
    rescale is only exact when the gross cap never binds; because the cap
    binds on some weeks and not others, realized vol is a concave,
    saturating function of k, so k is calibrated iteratively against
    REALIZED IS vol, then frozen (Dammluckan's fixed-calibration-path
    pattern, reused verbatim from research/smittotalet/tsmom.py)."""
    target_vol = config.PORTFOLIO_VOL_TARGET if target_vol is None else target_vol
    gross_cap = config.GROSS_CAP if gross_cap is None else gross_cap
    max_iter = config.VOL_TARGET_SOLVE_MAX_ITER if max_iter is None else max_iter
    tol = config.VOL_TARGET_SOLVE_TOL if tol is None else tol

    if k0 is None:
        r0 = (raw_weights * panel.simple_returns()).sum(axis=1, skipna=True).loc[is_start:is_end]
        vol0 = r0.std() * np.sqrt(config.TRADING_DAYS_YEAR)
        k0 = target_vol / vol0 if vol0 and vol0 > 0 else 1.0
    k = k0
    for _ in range(max_iter):
        weights = apply_gross_cap(raw_weights, k, gross_cap)
        rets = portfolio_returns(panel, weights, apply_costs=False).loc[is_start:is_end]
        vol = rets.std() * np.sqrt(config.TRADING_DAYS_YEAR)
        if not vol or vol <= 0:
            break
        if abs(vol - target_vol) < tol:
            break
        k = k * (target_vol / vol)
    return k


# ---------------------------------------------------------------------------
# T0: Smittotalet's verbatim 12m-sign / inverse-20d-(rolling, not EWMA)-vol
# base book -- pipeline-sanity twin only, never combined with the op-clock
# signal. Provenance: research/smittotalet/tsmom.py, same branch/commit.
# ---------------------------------------------------------------------------
def t0_raw_signal(panel) -> pd.DataFrame:
    """sign(P_t/P_{t-252}-1) / vol20_t, daily, no lag applied yet. vol20 is
    PLAIN rolling std (not the EWMA used for Timglaset's own sizing) --
    byte-faithful to smittotalet/tsmom.py::raw_signal."""
    adj = panel.adjusted_close
    mom = np.sign(adj / adj.shift(config.BASE_TSMOM_LOOKBACK) - 1.0)
    daily_ret = adj.pct_change()
    vol20 = daily_ret.rolling(config.BASE_TSMOM_VOL_LOOKBACK).std()
    return mom / vol20


def t0_weekly_rebalanced_position(panel) -> pd.DataFrame:
    return scheduling.friday_lag_apply(t0_raw_signal(panel))


def build_t0_book(panel, is_start, is_end, apply_costs: bool = True):
    """Returns (weights, daily_returns, k) for the frozen-k T0 base book."""
    raw = t0_weekly_rebalanced_position(panel)
    k = solve_k_for_target_vol(panel, raw, is_start, is_end)
    weights = apply_gross_cap(raw, k)
    rets = portfolio_returns(panel, weights, apply_costs=apply_costs)
    return weights, rets, k


# ---------------------------------------------------------------------------
# Shared clock-book builder: primary signal, T1 (calendar), T2 (shuffle),
# T3 (variance), and all 27 grid cells all go through this one function,
# differing only in which (M, V, T) triple and halflife/transform they pass.
# ---------------------------------------------------------------------------
def clock_signal_position(M: pd.DataFrame, V: pd.DataFrame, T: pd.DataFrame,
                           halflife_op: float, transform: str) -> pd.DataFrame:
    """f(z), sampled at Friday close, held through the following ISO week
    (t+1 lock, spec §5) -- not yet vol-scaled or gross-capped."""
    f_z = opclock.compute_signal(M, V, T, halflife_op, transform)
    return scheduling.friday_lag_apply(f_z)


def inverse_vol_weight(position: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """w_i = f(z_i) / sigma_hat_{i,20d} (inverse 20-day EWMA vol, spec §5;
    see config.SIGNAL_VOL_EWMA_SPAN for the DECLARED parameterization)."""
    vol = returns.ewm(span=config.SIGNAL_VOL_EWMA_SPAN,
                       min_periods=config.SIGNAL_VOL_EWMA_MIN_PERIODS).std()
    return position / vol


def build_clock_book(panel, M: pd.DataFrame, V: pd.DataFrame, T: pd.DataFrame,
                      halflife_op: float, transform: str, is_start, is_end,
                      apply_costs: bool = True):
    """End-to-end: signal -> Friday-lag -> inverse-vol -> iterative-k ->
    gross-cap -> net returns. Returns (weights, daily_returns, k)."""
    position = clock_signal_position(M, V, T, halflife_op, transform)
    raw_weights = inverse_vol_weight(position, panel.simple_returns())
    k = solve_k_for_target_vol(panel, raw_weights, is_start, is_end)
    weights = apply_gross_cap(raw_weights, k)
    rets = portfolio_returns(panel, weights, apply_costs=apply_costs)
    return weights, rets, k
