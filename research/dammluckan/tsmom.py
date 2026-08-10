"""
Dammluckan -- TSMOM (time-series momentum) proxy.

Adapted from research/formdriften/tsmom.py: this is "the TSMOM-proxyn
(dokumenterad i repot)" the brief refers to -- a mechanical, dependency-free
Moskowitz-Ooi-Pedersen-style construction with no fitted parameters, used
both as (a) a redundancy-screen regressor and (b) the ρ(TSMOM) > 0.4
diversification kill criterion, computed on the same panel/universe the
candidate strategy trades.
"""
import numpy as np
import pandas as pd

from . import config

LOOKBACK_DAYS = 252
VOL_WINDOW = 63
MAX_WEIGHT = 0.15


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def tsmom_returns(panel, lookback=LOOKBACK_DAYS, vol_window=VOL_WINDOW, max_weight=MAX_WEIGHT,
                   rebalance_dates=None) -> pd.Series:
    """
    Monthly-rebalanced TSMOM proxy: sign(trailing 12m adjusted-close return),
    inverse-63d-vol sized, 15% per-name cap, gross renormalized to 1.0 after
    clipping. Daily weights held flat between rebalance dates (signal decided
    strictly through the rebalance date's close, applied from the next day).
    """
    returns = panel.log_returns
    idx = returns.index
    if rebalance_dates is None:
        rebalance_dates = month_end_dates(idx)

    vol = returns.rolling(vol_window, min_periods=vol_window // 2).std(ddof=1)
    daily_w = pd.DataFrame(0.0, index=idx, columns=returns.columns)

    px = panel.adj_close
    for k, dt in enumerate(rebalance_dates):
        pos = idx.searchsorted(dt)
        if pos < lookback:
            continue
        trail_start = idx[pos - lookback]
        p0 = px.loc[trail_start]
        p1 = px.loc[dt]
        trail_ret = p1 / p0 - 1.0
        sig = np.sign(trail_ret).reindex(returns.columns)

        v = vol.loc[dt].reindex(returns.columns)
        raw = (sig / v).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross = raw.abs().sum()
        w = raw / gross if gross > 0 else raw * 0.0
        w = w.clip(lower=-max_weight, upper=max_weight)
        gross2 = w.abs().sum()
        if gross2 > 0:
            w = w / gross2

        end_dt = rebalance_dates[k + 1] if k + 1 < len(rebalance_dates) else idx[-1]
        mask = (idx > dt) & (idx <= end_dt)
        daily_w.loc[mask, :] = w.values

    # daily_w[d] already holds the weight decided at the most recent
    # rebalance date strictly before d (mask excludes the rebalance day
    # itself), so no additional lag is applied here -- matching Formdriften's
    # tsmom.py execution-timing convention exactly.
    port_ret = (daily_w.fillna(0.0) * returns.fillna(0.0)).sum(axis=1)
    return port_ret
