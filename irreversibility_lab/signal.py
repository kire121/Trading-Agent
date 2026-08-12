"""Z-scored irreversibility signal, regime classification (with hysteresis),
and direction rules.

I_t is computed at *daily* resolution (cheap: ~0.15ms/window with ts2vg, so a
full 2000-2026 x 12-ticker x 3-estimator panel takes well under a minute) so
that the 750-day rolling z-score history is literally daily, as specified.
Regime/direction decisions are then sampled at weekly (Friday) anchors, since
that is the only cadence the strategy actually trades on.
"""

import numpy as np
import pandas as pd

from . import config
from .estimators import ESTIMATORS, rolling_estimator


def compute_irreversibility_panel(returns_df, W, estimator="hvg"):
    """Daily I_t panel (date x ticker): trailing-W-day estimator ending at
    each date. NaN until a ticker has W days of contiguous finite history.
    """
    func = ESTIMATORS[estimator]
    n = len(returns_df)
    out = {}
    for col in returns_df.columns:
        vals = rolling_estimator(returns_df[col].values, W, func)
        arr = np.full(n, np.nan)
        arr[W - 1:] = vals
        out[col] = arr
    return pd.DataFrame(out, index=returns_df.index)


def rolling_zscore(panel_df, history=config.Z_HISTORY, min_periods=None):
    """Causal (no-lookahead) rolling z-score of a daily I_t panel against its
    own trailing `history`-day distribution."""
    if min_periods is None:
        min_periods = max(60, history // 4)
    roll = panel_df.rolling(window=history, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std(ddof=0)
    z = (panel_df - mean) / std.replace(0, np.nan)
    return z


def weekly_anchor_dates(index):
    """Last trading day of each calendar week present in `index` (usually
    Friday, or the last available day before a Friday holiday)."""
    index = pd.DatetimeIndex(index)
    iso = index.isocalendar()
    key = pd.MultiIndex.from_arrays([iso["year"].values, iso["week"].values])
    s = pd.Series(index, index=key)
    anchors = s.groupby(level=[0, 1]).last()
    return pd.DatetimeIndex(sorted(anchors.values))


def next_trading_day(index, anchor_date):
    """First trading date strictly after `anchor_date` in `index`; None if
    `anchor_date` is the last available date."""
    pos = index.searchsorted(anchor_date, side="right")
    if pos >= len(index):
        return None
    return index[pos]


def classify_regime_series(z_series, upper=config.Z_THRESHOLD_DEFAULT, lower=None):
    """Hysteresis regime classification over an already-weekly z-series for
    ONE ticker. Starts NEUTRAL; holds the previous regime while
    lower < z < upper; NaN z (insufficient history) also holds previous.
    """
    if lower is None:
        lower = -upper
    regime = pd.Series(index=z_series.index, dtype=object)
    current = "NEUTRAL"
    for dt, z in z_series.items():
        if pd.notna(z):
            if z >= upper:
                current = "TREND"
            elif z <= lower:
                current = "MEANREV"
        regime.loc[dt] = current
    return regime


def classify_regime_panel(z_weekly_df, upper=config.Z_THRESHOLD_DEFAULT, lower=None):
    return pd.DataFrame(
        {col: classify_regime_series(z_weekly_df[col], upper=upper, lower=lower)
         for col in z_weekly_df.columns},
        index=z_weekly_df.index,
    )


def trend_direction(px_df, lookback=config.TREND_LOOKBACK):
    ret = px_df / px_df.shift(lookback) - 1.0
    return np.sign(ret)


def meanrev_direction(px_df, lookback=config.MEANREV_LOOKBACK):
    ret = px_df / px_df.shift(lookback) - 1.0
    return -np.sign(ret)


def combine_direction(regime_df, trend_dir_df, meanrev_dir_df):
    direction = pd.DataFrame(0.0, index=regime_df.index, columns=regime_df.columns)
    is_trend = regime_df == "TREND"
    is_meanrev = regime_df == "MEANREV"
    direction = direction.where(~is_trend, trend_dir_df)
    direction = direction.where(~is_meanrev, meanrev_dir_df)
    return direction.fillna(0.0)


def build_weekly_signals(px_df, returns_df, W=config.W_DEFAULT,
                          threshold=config.Z_THRESHOLD_DEFAULT, estimator="hvg"):
    """End-to-end: daily I_t -> daily Z -> weekly (Friday) regime with
    hysteresis -> weekly direction. Returns a dict of weekly-indexed
    DataFrames plus the underlying daily I_t/Z panels for diagnostics.
    """
    i_panel = compute_irreversibility_panel(returns_df, W=W, estimator=estimator)
    z_panel = rolling_zscore(i_panel, history=config.Z_HISTORY)

    anchors = weekly_anchor_dates(returns_df.index)
    z_weekly = z_panel.loc[anchors]

    regime_weekly = classify_regime_panel(z_weekly, upper=threshold)

    trend_dir = trend_direction(px_df).loc[anchors]
    meanrev_dir = meanrev_direction(px_df).loc[anchors]
    direction_weekly = combine_direction(regime_weekly, trend_dir, meanrev_dir)

    return {
        "i_panel_daily": i_panel,
        "z_panel_daily": z_panel,
        "z_weekly": z_weekly,
        "regime_weekly": regime_weekly,
        "direction_weekly": direction_weekly,
        "anchors": anchors,
    }
