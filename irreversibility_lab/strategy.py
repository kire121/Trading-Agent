"""Vol-target sizing and weekly weight construction."""

import numpy as np
import pandas as pd

from . import config


def instrument_vol(returns_df, lookback=config.VOL_LOOKBACK, annualization=config.ANNUALIZATION):
    """Trailing realized annualized vol per instrument (causal: uses data up
    to and including the row's own date)."""
    return returns_df.rolling(lookback, min_periods=lookback).std(ddof=0) * np.sqrt(annualization)


def position_size(vol_df, target_annual=config.VOL_TARGET_ANNUAL,
                   n=config.N_INSTRUMENTS, cap=config.PER_INSTRUMENT_CAP):
    """v_i = min(cap, (target_annual / n) / sigma_i)."""
    raw = (target_annual / n) / vol_df.replace(0, np.nan)
    return raw.clip(upper=cap)


def apply_gross_cap(weights_df, gross_cap=config.GROSS_CAP):
    gross = weights_df.abs().sum(axis=1)
    scale = (gross_cap / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    return weights_df.mul(scale, axis=0)


def build_weekly_weights(direction_weekly, returns_df, anchors,
                          target_annual=config.VOL_TARGET_ANNUAL,
                          n=config.N_INSTRUMENTS, per_inst_cap=config.PER_INSTRUMENT_CAP,
                          gross_cap=config.GROSS_CAP, vol_lookback=config.VOL_LOOKBACK):
    """Target weight per instrument at each weekly anchor date: sizing uses
    only vol data available as of that anchor (causal), direction comes from
    the (already weekly, already causal) regime/direction signal.
    """
    vol_daily = instrument_vol(returns_df, lookback=vol_lookback)
    vol_weekly = vol_daily.reindex(anchors)
    size_weekly = position_size(vol_weekly, target_annual=target_annual, n=n, cap=per_inst_cap)
    raw_weights = direction_weekly * size_weekly
    raw_weights = raw_weights.fillna(0.0)
    return apply_gross_cap(raw_weights, gross_cap=gross_cap)
