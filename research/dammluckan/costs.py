"""
Dammluckan -- ADV-bucket transaction cost model.

Ported verbatim (bucket thresholds, bps schedule, and the documented spread
caveat) from research/formdriften/costs.py -- the only ADV-bucketed cost
model anywhere in this repo's research-branch series. This is what the brief
means by "Kostnader ur repots ADV-bucketmodell (med dess dokumenterade
spread-caveat)".

DOCUMENTED SPREAD CAVEAT (verbatim convention, restated here): EODHD does not
provide quoted bid/ask spread history, so the half-spread leg below is
approximated from trailing dollar-ADV via a monotone liquidity bucket --
a documented approximation, not a measured spread. Only the commission leg
(3bp/side) is a clean, un-approximated assumption.
"""
import numpy as np
import pandas as pd

from . import config

_ADV_BUCKETS = config.ADV_BUCKETS  # (min ADV $, half-spread bps), most liquid first


def half_spread_bps(adv_dollars):
    """Vectorized monotone ADV-bucket lookup. Accepts scalar, ndarray, or Series.

    Buckets are applied most-liquid-first: the first (highest) threshold an
    ADV value clears wins, so NaN ADV (not-yet-eligible-for-ADV history)
    falls through to the illiquid default bucket rather than raising.
    """
    arr = np.asarray(adv_dollars, dtype=float)
    out = np.full(arr.shape, _ADV_BUCKETS[-1][1], dtype=float)
    # Apply ascending by threshold so higher (more-liquid) thresholds are
    # applied last and correctly overwrite the looser bucket.
    for min_adv, bps in sorted(_ADV_BUCKETS, key=lambda x: x[0]):
        out = np.where(arr >= min_adv, bps, out)
    if np.isscalar(adv_dollars):
        return float(out)
    if isinstance(adv_dollars, pd.Series):
        return pd.Series(out, index=adv_dollars.index)
    return out


def round_trip_cost_bps(adv_dollars):
    return config.COMMISSION_BPS + half_spread_bps(adv_dollars)


def cost_fraction(adv_dollars):
    """Round-trip cost as a fraction of notional (not bps)."""
    return round_trip_cost_bps(adv_dollars) / 10_000.0
