"""Alternative weekly-weight construction rules used as baselines and as the
building blocks for the null/robustness machinery in validation.py:

- irreversibility-regime weights (the actual strategy; wraps signal.py)
- static 50/50 trend+meanrev mix, no regime switching   (baseline b)
- vol-z-score-driven regime switching instead of I_t     (baseline c)
- pure always-on time-series momentum (TSMOM) sleeve     (correlation check)
"""

import numpy as np
import pandas as pd

from . import config, signal, strategy


def irreversibility_weekly_weights(px_df, returns_df, W=config.W_DEFAULT,
                                    threshold=config.Z_THRESHOLD_DEFAULT,
                                    estimator="hvg", z_panel_daily=None):
    """The actual strategy. If z_panel_daily is precomputed (expensive HVG
    step reused across the threshold grid), pass it to skip recomputation.
    """
    if z_panel_daily is None:
        i_panel = signal.compute_irreversibility_panel(returns_df, W=W, estimator=estimator)
        z_panel_daily = signal.rolling_zscore(i_panel, history=config.Z_HISTORY)

    anchors = signal.weekly_anchor_dates(returns_df.index)
    z_weekly = z_panel_daily.loc[anchors]
    regime_weekly = signal.classify_regime_panel(z_weekly, upper=threshold)
    trend_dir = signal.trend_direction(px_df).loc[anchors]
    meanrev_dir = signal.meanrev_direction(px_df).loc[anchors]
    direction_weekly = signal.combine_direction(regime_weekly, trend_dir, meanrev_dir)

    weekly_w = strategy.build_weekly_weights(direction_weekly, returns_df, anchors)
    return {
        "weekly_weights": weekly_w,
        "regime_weekly": regime_weekly,
        "direction_weekly": direction_weekly,
        "z_weekly": z_weekly,
        "anchors": anchors,
    }


def static_mix_weekly_weights(px_df, returns_df):
    """Baseline (b): fixed 50/50 blend of the trend sleeve and the
    mean-reversion sleeve, no regime switching at all."""
    anchors = signal.weekly_anchor_dates(returns_df.index)
    trend_dir = signal.trend_direction(px_df).loc[anchors]
    meanrev_dir = signal.meanrev_direction(px_df).loc[anchors]

    trend_w = strategy.build_weekly_weights(trend_dir, returns_df, anchors)
    meanrev_w = strategy.build_weekly_weights(meanrev_dir, returns_df, anchors)
    mix_w = 0.5 * trend_w + 0.5 * meanrev_w
    return {"weekly_weights": mix_w, "anchors": anchors}


def vol_z_weekly_weights(px_df, returns_df, threshold=config.Z_THRESHOLD_DEFAULT,
                          vol_lookback=config.VOL_LOOKBACK):
    """Baseline (c): same regime/hysteresis/direction machinery, but the
    regime driver is a rolling z-score of realized volatility instead of the
    HVG/ordinal/psi irreversibility measure. Tests whether irreversibility
    adds anything beyond a plain vol-timing regime switch.
    """
    vol_daily = strategy.instrument_vol(returns_df, lookback=vol_lookback)
    z_panel_daily = signal.rolling_zscore(vol_daily, history=config.Z_HISTORY)

    anchors = signal.weekly_anchor_dates(returns_df.index)
    z_weekly = z_panel_daily.loc[anchors]
    regime_weekly = signal.classify_regime_panel(z_weekly, upper=threshold)
    trend_dir = signal.trend_direction(px_df).loc[anchors]
    meanrev_dir = signal.meanrev_direction(px_df).loc[anchors]
    direction_weekly = signal.combine_direction(regime_weekly, trend_dir, meanrev_dir)

    weekly_w = strategy.build_weekly_weights(direction_weekly, returns_df, anchors)
    return {
        "weekly_weights": weekly_w,
        "regime_weekly": regime_weekly,
        "z_weekly": z_weekly,
        "anchors": anchors,
    }


def pure_tsmom_weekly_weights(px_df, returns_df):
    """Always-on time-series momentum sleeve (no mean-reversion, no regime
    switching) used only for the TSMOM correlation hard-limit check."""
    anchors = signal.weekly_anchor_dates(returns_df.index)
    trend_dir = signal.trend_direction(px_df).loc[anchors]
    weekly_w = strategy.build_weekly_weights(trend_dir, returns_df, anchors)
    return {"weekly_weights": weekly_w, "anchors": anchors}
