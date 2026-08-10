"""Load the committed OHLCV panels and derive the causal statistics the
event/signal machinery needs (dollar-volume robust z-score, return vol,
pairwise trailing correlation). Every rolling statistic here is computed on
the window (t-N .. t-1) and then evaluated at t, i.e. it never uses day t's
own value to judge day t -- see config.py's note on this declared
resolution of "endast data <= t0"."""
import os

import numpy as np
import pandas as pd

from research.omori import config

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

FIELDS = ["open", "high", "low", "close", "adjusted_close", "volume"]


class Panel:
    """Bag of aligned OHLCV DataFrames (columns=tickers, index=date) for one
    universe ("primary" or "secondary")."""

    def __init__(self, label):
        self.label = label
        self.raw = {}
        for field in FIELDS:
            path = os.path.join(DATA_DIR, f"{label}_{field}.csv")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            self.raw[field] = df
        self.tickers = list(self.raw["close"].columns)
        self.index = self.raw["close"].index

    @property
    def close(self):
        return self.raw["close"]

    @property
    def adj_close(self):
        return self.raw["adjusted_close"]

    @property
    def volume(self):
        return self.raw["volume"]

    def dollar_volume(self):
        return self.close * self.volume

    def log_returns(self):
        """Adjusted-close log returns (dividend/split adjusted)."""
        adj = self.adj_close
        return np.log(adj / adj.shift(1))

    def simple_returns(self):
        adj = self.adj_close
        return adj / adj.shift(1) - 1.0


def rolling_median_mad_z(series, lookback):
    """Causal robust z-score of `series` against its own trailing (t-N..t-1)
    median/MAD, evaluated at t. NaN wherever fewer than `lookback` prior
    observations exist."""
    prior = series.shift(1)
    med = prior.rolling(lookback, min_periods=lookback).median()
    mad = prior.rolling(lookback, min_periods=lookback).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True
    )
    denom = config.MAD_SCALE * mad
    z = (series - med) / denom.replace(0.0, np.nan)
    return z


def rolling_return_sigma(returns, lookback):
    """Causal trailing (t-N..t-1) realized-vol estimate of `returns`,
    evaluated at t (excludes day t's own return, matching the volume-z
    baseline convention)."""
    prior = returns.shift(1)
    return prior.rolling(lookback, min_periods=lookback).std(ddof=1)


def rolling_adv(dollar_volume, lookback):
    """Trailing (t-N..t-1) mean dollar ADV, causal, used for the cost model's
    liquidity bucket."""
    prior = dollar_volume.shift(1)
    return prior.rolling(lookback, min_periods=max(5, lookback // 4)).mean()


def trailing_pairwise_corr(returns, lookback, as_of_idx):
    """Pairwise trailing (t-N..t-1) correlation matrix among `returns`
    columns, evaluated as of `as_of_idx` (a positional row index into
    `returns`). Returns a DataFrame (tickers x tickers). NaN-filled columns
    (insufficient history) get NaN correlation with everything."""
    start = max(0, as_of_idx - lookback)
    window = returns.iloc[start:as_of_idx]
    if len(window) < max(20, lookback // 3):
        cols = returns.columns
        return pd.DataFrame(np.nan, index=cols, columns=cols)
    return window.corr(min_periods=max(20, lookback // 3))


def load(label):
    return Panel(label)
