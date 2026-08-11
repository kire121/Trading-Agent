"""Shared weekly rebalance-timing helper: sample a signal at each week-end
(Friday) close, hold it constant through the FOLLOWING ISO week.

Ported verbatim from research/smittotalet/scheduling.py, branch
claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b. Matches spec
§5's "Signal fredag close -> fill måndag close (t+1-spärr)" exactly: both
the base book (T0) and the op-clock signal (primary + T1/T2/T3) share this
one function, so both are lagged by exactly one week with no look-ahead.
"""
import pandas as pd

from . import config


def friday_lag_apply(obj):
    """obj: pd.Series or pd.DataFrame, daily-indexed. Returns the same shape,
    where every day within ISO week W carries the value observed at week
    (W-1)'s Friday close (or the last trading day of that week, if Friday
    is a market holiday)."""
    week_period = obj.index.to_period(f"W-{config.REBALANCE_WEEKDAY}")
    week_end_date = pd.Series(obj.index, index=week_period).groupby(level=0).last()
    at_week_end = obj.loc[week_end_date.values].copy()
    at_week_end.index = week_end_date.index
    applied = at_week_end.shift(1)
    daily = applied.loc[week_period]
    daily.index = obj.index
    return daily


def week_end_values(obj):
    """The value sampled at each week's Friday close (no lag), indexed by
    weekly PeriodIndex."""
    week_period = obj.index.to_period(f"W-{config.REBALANCE_WEEKDAY}")
    week_end_date = pd.Series(obj.index, index=week_period).groupby(level=0).last()
    at_week_end = obj.loc[week_end_date.values].copy()
    at_week_end.index = week_end_date.index
    return at_week_end
