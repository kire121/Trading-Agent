"""Shared data-loading + eligibility helpers, reused by the backtest,
robustness sweep and null-hypothesis suite so price data is fetched once."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data_fetch
from .signal import build_d_l_panel


@dataclass
class Universe:
    name: str
    tickers: list
    returns: pd.DataFrame
    price: pd.DataFrame
    adv: pd.DataFrame
    is_fx: bool = False


def load_universe(name, tickers, is_fx=False, refresh=False):
    data = data_fetch.fetch_universe(tickers, refresh=refresh)
    price = data_fetch.build_price_panel(data)
    if is_fx:
        adv = pd.DataFrame(np.inf, index=price.index, columns=price.columns)
    else:
        vol = data_fetch.build_volume_panel(data)
        adv = data_fetch.dollar_adv(price, vol)
    returns = price.pct_change()
    return Universe(name=name, tickers=list(price.columns), returns=returns, price=price, adv=adv, is_fx=is_fx)


def eligibility_panel(universe: Universe, d_l_panel: pd.DataFrame, adv_threshold=20e6):
    """PIT eligibility: has a defined D_L (>= curr+prev days of history) AND
    (for non-FX) trailing dollar ADV above threshold as of that date."""
    adv_at_dates = universe.adv.reindex(d_l_panel.index)
    if universe.is_fx:
        adv_ok = pd.DataFrame(True, index=d_l_panel.index, columns=d_l_panel.columns)
    else:
        adv_ok = adv_at_dates > adv_threshold
    hist_ok = d_l_panel.notna()
    return adv_ok.reindex(columns=d_l_panel.columns).fillna(False) & hist_ok


def month_end_dates(index):
    s = pd.Series(index, index=index)
    return list(s.groupby([index.year, index.month]).apply(lambda x: x.iloc[-1]).values)


def build_signal_and_eligibility(universe: Universe, curr_window=126, prev_window=126, adv_threshold=20e6,
                                  eval_dates=None):
    if eval_dates is None:
        eval_dates = month_end_dates(universe.returns.index)
    d_l = build_d_l_panel(universe.returns.fillna(0.0), eval_dates, curr_window, prev_window)
    elig = eligibility_panel(universe, d_l, adv_threshold)
    return d_l, elig
