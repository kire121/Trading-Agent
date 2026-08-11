"""Shared forward-return construction for the fast-exit ladder (Steg 1-4):
decision Friday close -> execution next trading day (Monday) -> hold
exactly one ISO week -> next execution (spec SS4). Vol-scaled by the same
sigma_hat used for position sizing (spec SS9 Steg1: "icke-overlappande
framatavkastning exekvering->exekvering (vol-skalad)")."""
import numpy as np
import pandas as pd

from . import signal


def execution_dates_after(decision_dates: pd.DatetimeIndex, trading_calendar: pd.DatetimeIndex) -> pd.Series:
    """Maps each decision date to the next trading day present in
    `trading_calendar` (spec: "nasta handelsdags close (mandag)")."""
    cal = pd.DatetimeIndex(sorted(trading_calendar))
    out = {}
    for d in decision_dates:
        later = cal[cal > d]
        out[d] = later[0] if len(later) else pd.NaT
    return pd.Series(out)


def weekly_execution_returns(adjC_wide: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Non-overlapping execution->execution simple returns, indexed by
    DECISION date t (the return realized by the position sized off week
    t's signal): r_i(t) = adjC_i[exec(t+1)]/adjC_i[exec(t)] - 1."""
    trading_calendar = adjC_wide.index
    exec_dates = execution_dates_after(decision_dates, trading_calendar)
    exec_dates = exec_dates.dropna()
    valid_decisions = exec_dates.index

    exec_prices = adjC_wide.reindex(exec_dates.to_numpy())
    exec_prices.index = valid_decisions  # align by decision date
    fwd = exec_prices.shift(-1) / exec_prices - 1.0
    return fwd


def vol_scaled_returns(fwd_returns: pd.DataFrame, sigma_hat_wide: pd.DataFrame) -> pd.DataFrame:
    sigma_at_decision = sigma_hat_wide.reindex(fwd_returns.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = fwd_returns / sigma_at_decision
    return scaled.replace([np.inf, -np.inf], np.nan)
