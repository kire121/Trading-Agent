"""
Transaction cost model: 3bp/side commission-equivalent + half-spread bucketed
by ADV. We do not have EODHD quoted bid/ask spread data, so the half-spread
is approximated from trailing dollar-ADV via a monotone liquidity bucket --
documented approximation, not a measured spread.
"""
import numpy as np

COMMISSION_BPS = 3.0

# (min ADV $, half-spread bps) buckets, most liquid first.
_ADV_BUCKETS = [
    (500e6, 0.5),
    (200e6, 1.0),
    (100e6, 1.5),
    (50e6, 2.5),
    (20e6, 4.0),
    (0.0, 7.0),
]


def half_spread_bps(adv_dollars):
    """Vectorized ADV -> approximate half-spread (bps) bucket lookup."""
    adv = np.asarray(adv_dollars, dtype=float)
    out = np.full(adv.shape, _ADV_BUCKETS[-1][1])
    for min_adv, spread in sorted(_ADV_BUCKETS, key=lambda x: x[0]):
        out = np.where(adv >= min_adv, spread, out)
    return out


def round_trip_cost_bps(adv_dollars):
    """Cost (bps of traded notional) charged per unit of |weight change|."""
    return COMMISSION_BPS + half_spread_bps(adv_dollars)
