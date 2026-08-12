"""Frozen base book: 12-month sign TSMOM, weekly rebalance, portfolio-level
vol targeting to 10% annualized, 200% gross cap.

Provenance (rule 2 -- reuse before build): ported near-verbatim from
research/smittotalet/tsmom.py (branch claude/smittotalet-portfolio-overlay-
0bl1sh, commit a67df1b) -- the spec's own Sec.5 names this exact construction
("husets dokumenterade TSMOM-proxy") and requires grep-verification against
research/smittotalet/ before use, with the module's *tested* execution
convention authoritative over any restated assumption elsewhere in the spec
(logged as AVVIKELSER.md sec.3 where they differ). Only change from the
original: `sign_momentum` is factored out as its own function (still used
internally by `raw_signal`, formula byte-identical) so Metusalem's own
episode/age tracking (survival_trend.extract_episodes/age_panel) consumes
EXACTLY the same s_{i,t} = sign(12m return) series the base book itself
uses for direction -- not a second, divergent copy of the same quantity.

The k-solve constant is iterative (Dammluckan's fixed-point pattern, per the
original's own docstring) because the 200% gross cap binds on some weeks and
not others, making realized vol a concave, saturating function of k.
"""
import numpy as np
import pandas as pd

from . import config
from . import costs
from . import scheduling


def sign_momentum(panel) -> pd.DataFrame:
    """s_{i,t} = sign(P_t/P_{t-252} - 1), daily, no lag applied yet."""
    adj = panel.adjusted_close
    return np.sign(adj / adj.shift(config.TSMOM_LOOKBACK) - 1.0)


def raw_signal(panel) -> pd.DataFrame:
    """sign(P_t/P_{t-252}-1) / vol20_t, daily, no lag applied yet."""
    mom = sign_momentum(panel)
    daily_ret = panel.adjusted_close.pct_change()
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


def portfolio_returns(panel, weights: pd.DataFrame, apply_costs: bool = True,
                       one_way_bps: float = config.COST_BPS_IS_PRIMARY) -> pd.Series:
    simple_ret = panel.simple_returns()
    gross_ret = (weights * simple_ret).sum(axis=1, skipna=True)
    if apply_costs:
        gross_ret = costs.apply_costs(gross_ret, weights, one_way_bps)
    return gross_ret


def solve_k_for_target_vol(panel, is_start, is_end, target_vol: float = config.PORTFOLIO_VOL_TARGET,
                            gross_cap: float = config.GROSS_CAP, k0: float | None = None,
                            max_iter: int = config.VOL_TARGET_SOLVE_MAX_ITER,
                            tol: float = config.VOL_TARGET_SOLVE_TOL,
                            raw: pd.DataFrame | None = None) -> float:
    """Iterative k-solve (Dammluckan pattern) -- shared calibration path for
    the base book AND every tilt variant/twin (spec Sec.5/Sec.9: "delar
    kalibreringsvag" / "alla genom samma volmalslosare"). `raw` lets the
    caller pass an already-tilted raw signal (e.g. m*w_bas_raw) instead of
    re-deriving the base book's own raw_signal -- same solver, different
    input, per Sec.5 "w_tilt = m (*) w_bas, darefter genom exakt samma
    volmalslosare"."""
    if raw is None:
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


def build_base_book(panel, k: float, gross_cap: float = config.GROSS_CAP, apply_costs: bool = True,
                     one_way_bps: float = config.COST_BPS_IS_PRIMARY):
    """Returns (weights, daily_returns) for the frozen-k base TSMOM book."""
    raw = weekly_rebalanced_position(panel)
    weights = apply_gross_cap(raw, k, gross_cap)
    rets = portfolio_returns(panel, weights, apply_costs=apply_costs, one_way_bps=one_way_bps)
    return weights, rets


def weekly_return(daily_returns: pd.Series) -> pd.Series:
    """Compounds daily returns within each ISO (Mon-Fri, W-FRI) week.
    Provenance: research/smittotalet/backtest.py::weekly_return (branch
    claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b), verbatim.
    ALL Sharpe/DSR/uplift gate computations in gates.py operate on this
    weekly-compounded series, never on the raw daily returns basbok.py
    produces -- Sharpe with the default periods_per_year=52 (lib.metrics'
    convention) is only correct on weekly-frequency returns; using it
    directly on daily returns would silently misannualize every ΔSR_net
    and DSR figure in the study."""
    week_period = daily_returns.index.to_period(f"W-{config.REBALANCE_WEEKDAY}")
    return (1.0 + daily_returns.fillna(0.0)).groupby(week_period).prod() - 1.0
