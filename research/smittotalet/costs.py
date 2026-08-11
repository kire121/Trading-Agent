"""ADV-bucket transaction cost model.

Ported verbatim (bucket thresholds, bps schedule, and the documented spread
caveat) from research/dammluckan/costs.py, itself ported from
research/formdriften/costs.py -- the only ADV-bucketed cost model anywhere
in this repo's research-branch series. This is what the brief means by
"repots ... ADV-bucket-kostnader (ej Runradens platta 2 bp)".
"""
import numpy as np
import pandas as pd

from . import config

_ADV_BUCKETS = config.ADV_BUCKETS  # (min ADV $, half-spread bps), most liquid first


def half_spread_bps(adv_dollars):
    """Vectorized monotone ADV-bucket lookup. Accepts scalar, ndarray, or Series."""
    arr = np.asarray(adv_dollars, dtype=float)
    out = np.full(arr.shape, _ADV_BUCKETS[-1][1], dtype=float)
    for min_adv, bps in sorted(_ADV_BUCKETS, key=lambda x: x[0]):
        out = np.where(arr >= min_adv, bps, out)
    if np.isscalar(adv_dollars):
        return float(out)
    if isinstance(adv_dollars, pd.Series):
        return pd.Series(out, index=adv_dollars.index)
    if isinstance(adv_dollars, pd.DataFrame):
        return pd.DataFrame(out, index=adv_dollars.index, columns=adv_dollars.columns)
    return out


def round_trip_cost_bps(adv_dollars):
    return config.COMMISSION_BPS + half_spread_bps(adv_dollars)


def cost_fraction(adv_dollars):
    """Round-trip cost as a fraction of notional (not bps). NaN ADV -> illiquid default."""
    return round_trip_cost_bps(adv_dollars) / 10_000.0
