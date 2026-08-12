"""Backtest engine: weekly weight decisions -> next-trading-day execution ->
daily close-to-close P&L, with transaction costs and turnover accounting.

Execution timing convention: a weight decided at Friday close t_f becomes
*active* starting the next trading day (nominally Monday's open) and is held
constant until the following week's effective date. We do not model the
Friday-close -> Monday-open overnight gap separately (that would need a
second, unadjusted intraday-open price series); the new weight is treated as
earning the full close-to-close return starting the first day it is active.
This is a standard, documented simplification for a daily-close-only
backtest -- it does not introduce lookahead (the weight is fixed using only
information available at t_f, and only starts earning P&L on t_f+1 or later).
"""

import numpy as np
import pandas as pd

from . import config, signal


def daily_weights_from_weekly(weekly_w_df, daily_index):
    """Reindex weekly target weights onto the daily calendar: each week's
    weight becomes active on the first trading day strictly after its
    anchor (Friday) date, and holds until the next effective date.
    """
    eff_dates = []
    for anchor in weekly_w_df.index:
        eff_dates.append(signal.next_trading_day(daily_index, anchor))
    tmp = weekly_w_df.copy()
    tmp["__eff__"] = eff_dates
    tmp = tmp.dropna(subset=["__eff__"])
    tmp = tmp.set_index("__eff__")
    tmp = tmp[~tmp.index.duplicated(keep="last")].sort_index()
    daily_w = tmp.reindex(daily_index).ffill().fillna(0.0)
    return daily_w


def one_way_cost_bp(columns):
    return pd.Series(
        {c: config.COMMISSION_BP_PER_SIDE + config.HALF_SPREAD_BP.get(c, 2.0) for c in columns}
    )


def run_backtest(px_df, returns_df, weekly_weights_df, cost_bp=None):
    """Returns a dict with: daily_weights, portfolio_return (net of cost),
    gross_return (before cost), cost, turnover (sum |delta w| per day),
    equity curve, and per-instrument daily weight/return contribution.
    """
    daily_index = returns_df.index
    daily_w = daily_weights_from_weekly(weekly_weights_df, daily_index)
    daily_w = daily_w.reindex(columns=returns_df.columns).fillna(0.0)

    if cost_bp is None:
        cost_bp = one_way_cost_bp(returns_df.columns)

    delta_w = daily_w.diff()
    delta_w.iloc[0] = daily_w.iloc[0]
    turnover = delta_w.abs().sum(axis=1)
    cost = (delta_w.abs() * cost_bp).sum(axis=1) / 10000.0

    gross_return = (daily_w * returns_df).sum(axis=1)
    portfolio_return = gross_return - cost

    equity = (1.0 + portfolio_return).cumprod()

    return {
        "daily_weights": daily_w,
        "gross_return": gross_return,
        "cost": cost,
        "turnover": turnover,
        "portfolio_return": portfolio_return,
        "equity": equity,
    }


# --- Performance statistics ---------------------------------------------

def sharpe_ratio(returns, annualization=config.ANNUALIZATION, rf=0.0):
    returns = returns.dropna()
    if returns.std(ddof=0) == 0 or len(returns) < 2:
        return np.nan
    excess = returns - rf / annualization
    return float(excess.mean() / excess.std(ddof=0) * np.sqrt(annualization))


def sortino_ratio(returns, annualization=config.ANNUALIZATION, rf=0.0):
    returns = returns.dropna()
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std(ddof=0) == 0:
        return np.nan
    excess = returns - rf / annualization
    return float(excess.mean() / downside.std(ddof=0) * np.sqrt(annualization))


def max_drawdown(equity):
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def cagr(equity, annualization=config.ANNUALIZATION):
    n = len(equity)
    if n < 2 or equity.iloc[-1] <= 0:
        return np.nan
    years = n / annualization
    return float(equity.iloc[-1] ** (1 / years) - 1)


def calmar_ratio(equity, annualization=config.ANNUALIZATION):
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr(equity, annualization) / abs(mdd))


def performance_summary(portfolio_return, equity, turnover=None):
    out = {
        "sharpe": sharpe_ratio(portfolio_return),
        "sortino": sortino_ratio(portfolio_return),
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "vol_annual": float(portfolio_return.std(ddof=0) * np.sqrt(config.ANNUALIZATION)),
        "n_days": int(portfolio_return.notna().sum()),
    }
    if turnover is not None:
        out["avg_weekly_turnover"] = float(turnover[turnover > 0].mean()) if (turnover > 0).any() else 0.0
        out["annual_turnover"] = float(turnover.sum() / (len(turnover) / config.ANNUALIZATION))
    return out
