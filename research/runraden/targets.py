"""Target construction: vol-standardised next-week return z_{t+1}.

z_{i,t+1} = R_{i,t+1} / (sigma_i,daily(window) * sqrt(5))

sigma_i,daily(window) is the trailing realised daily-return volatility
computed from data up to and including the *signal* date (Friday close of
week t) -- i.e. it is point-in-time and identical to the sigma used later
for position sizing, so the same vol estimate standardises both the
regression target and x_{i,t+1} = ghat(w_t) / sigma_i(60d).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from words import daily_returns

WEEKLY_SCALE = np.sqrt(5.0)


def rolling_daily_vol(adj_close: pd.Series, window: int = 60, min_periods: int | None = None) -> pd.Series:
    """Trailing realised daily-return std, indexed by trading date (PIT)."""
    rets = daily_returns(adj_close)
    min_periods = min_periods or window
    return rets.rolling(window, min_periods=min_periods).std()


def attach_targets(panel: pd.DataFrame, prices: dict[str, pd.Series], vol_window: int = 60) -> pd.DataFrame:
    """Attach sigma_60d (as of t_signal) and z_next (vol-standardised target)."""
    panel = panel.copy()
    vol_lookup: dict[str, pd.Series] = {
        asset: rolling_daily_vol(series, window=vol_window) for asset, series in prices.items()
    }

    sigmas = np.full(len(panel), np.nan)
    for i, (asset, t_signal) in enumerate(zip(panel["asset"], panel["t_signal"])):
        vol_series = vol_lookup.get(asset)
        if vol_series is None or t_signal not in vol_series.index:
            continue
        sigmas[i] = vol_series.loc[t_signal]
    panel["sigma_60d"] = sigmas

    weekly_vol = panel["sigma_60d"] * WEEKLY_SCALE
    with np.errstate(invalid="ignore", divide="ignore"):
        panel["z_next"] = np.where(
            (weekly_vol > 0) & np.isfinite(weekly_vol),
            panel["next_week_return"] / weekly_vol,
            np.nan,
        )
    return panel
