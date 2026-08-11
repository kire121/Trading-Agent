"""Shared weekly rebalance-timing helper: sample a signal at each week-end
(Friday) close, hold it constant through the FOLLOWING ISO week. Used by
both the TSMOM base book (its own signal) and the G_t overlay (a different
signal, same cadence) so both are lagged by exactly one week, no look-ahead.
"""
import pandas as pd

from . import config


def friday_lag_apply(obj):
    """obj: pd.Series or pd.DataFrame, daily-indexed. Returns the same shape,
    where every day within ISO week W carries the value observed at week
    (W-1)'s Friday close."""
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
    weekly PeriodIndex -- e.g. for reading off the primary G_t series used
    to quantile-map the null twins onto."""
    week_period = obj.index.to_period(f"W-{config.REBALANCE_WEEKDAY}")
    week_end_date = pd.Series(obj.index, index=week_period).groupby(level=0).last()
    at_week_end = obj.loc[week_end_date.values].copy()
    at_week_end.index = week_end_date.index
    return at_week_end
