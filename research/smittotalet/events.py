"""Volatility-exceedance event detection.

E_{i,t} = 1{|r_{i,t}| > rolling 252d percentile q, PIT t-1}; X_t = sum_i E_{i,t}.

PIT discipline (house convention, same as omori/data.py): the threshold at t
is computed on abs(returns).shift(1).rolling(lookback), i.e. strictly the
window t-lookback..t-1, never touching r_t itself.
"""
import numpy as np
import pandas as pd

from . import config


def event_threshold(returns: pd.DataFrame, q: int, lookback: int = config.EVENT_LOOKBACK) -> pd.DataFrame:
    """Rolling q-th percentile of |returns|, evaluated strictly on t-lookback..t-1."""
    abs_r = returns.abs()
    return abs_r.shift(1).rolling(lookback, min_periods=lookback).quantile(q / 100.0)


def event_matrix(returns: pd.DataFrame, q: int, lookback: int = config.EVENT_LOOKBACK) -> pd.DataFrame:
    """Boolean-as-int E_{i,t} matrix. NaN wherever the PIT threshold isn't yet defined."""
    abs_r = returns.abs()
    thresh = event_threshold(returns, q, lookback)
    events = (abs_r > thresh).astype(float)
    events[thresh.isna()] = np.nan
    return events


def daily_count(events: pd.DataFrame) -> pd.Series:
    """X_t = sum_i E_{i,t}, counting only assets with a defined threshold that day."""
    return events.sum(axis=1, skipna=True)


def build(returns: pd.DataFrame, q: int, lookback: int = config.EVENT_LOOKBACK):
    events = event_matrix(returns, q, lookback)
    x_t = daily_count(events)
    return events, x_t
