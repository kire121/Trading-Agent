"""
Daily-bar derived features shared by universe construction, the Step-1
redundancy screen, the baseline twins, and portfolio P&L accounting.

All rolling stats are computed with .shift(1) causality where they feed
entry decisions -- day d's tradeable universe/controls must only use
information available through day d's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SPEC


def enrich(eod: pd.DataFrame) -> pd.DataFrame:
    """eod: ['date','open','high','low','close','adjusted_close','volume'] for one symbol."""
    df = eod.sort_values("date").reset_index(drop=True).copy()
    df["ret"] = df["adjusted_close"].pct_change()
    adj_factor = df["adjusted_close"] / df["close"]
    df["adj_open"] = df["open"] * adj_factor
    df["dollar_volume"] = df["close"] * df["volume"]
    df["adv"] = df["dollar_volume"].rolling(SPEC.universe.adv_window_days, min_periods=SPEC.universe.adv_window_days // 2).mean()
    df["realized_vol_20"] = df["ret"].rolling(20, min_periods=10).std() * np.sqrt(SPEC.sizing.trading_days_per_year)
    df["amihud"] = (df["ret"].abs() / df["dollar_volume"].replace(0, np.nan)).rolling(20, min_periods=10).mean() * 1e6
    df["turnover_rel"] = df["dollar_volume"] / df["adv"].replace(0, np.nan)
    df["ret_5d"] = df["adjusted_close"].pct_change(5)
    df["abs_ret_autocorr_20"] = df["ret"].abs().rolling(21, min_periods=15).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    df["volume_ar1_20"] = df["volume"].rolling(21, min_periods=15).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    return df


def forward_return(df: pd.DataFrame, horizon_days: int, price_col: str = "adjusted_close") -> pd.Series:
    """Return from tomorrow's open-equivalent (next close-to-close proxy) to
    horizon_days later -- used as the dependent variable in the Step-1 IC
    screen and (via portfolio.py) in the actual backtest."""
    return df[price_col].shift(-horizon_days) / df[price_col] - 1.0


def realized_vol_of_returns(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window, min_periods=max(5, window // 3)).std()
    if annualize:
        vol = vol * np.sqrt(SPEC.sizing.trading_days_per_year)
    return vol
