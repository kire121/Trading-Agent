import numpy as np
import pandas as pd

from .. import config
from .. import costs


def test_half_spread_monotone_in_liquidity():
    illiquid = costs.half_spread_bps(1e6)
    liquid = costs.half_spread_bps(1e9)
    assert illiquid > liquid


def test_round_trip_includes_commission():
    adv = 1e9
    assert costs.round_trip_cost_bps(adv) == config.COMMISSION_BPS + costs.half_spread_bps(adv)


def test_cost_fraction_scales_bps_by_1e4():
    adv = pd.Series([1e6, 1e8, 1e10])
    frac = costs.cost_fraction(adv)
    bps = costs.round_trip_cost_bps(adv)
    assert np.allclose(frac.values, bps.values / 10_000.0)


def test_nan_adv_falls_through_to_illiquid_bucket():
    val = costs.half_spread_bps(np.array([np.nan]))[0]
    assert val == config.ADV_BUCKETS[-1][1]
