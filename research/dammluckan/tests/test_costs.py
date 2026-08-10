import numpy as np
from research.dammluckan import costs


def test_half_spread_bucket_boundaries():
    # Exact bucket boundaries from config.ADV_BUCKETS, hand-checked.
    assert costs.half_spread_bps(500e6) == 0.5
    assert costs.half_spread_bps(499_999_999) == 1.0
    assert costs.half_spread_bps(200e6) == 1.0
    assert costs.half_spread_bps(100e6) == 1.5
    assert costs.half_spread_bps(50e6) == 2.5
    assert costs.half_spread_bps(20e6) == 4.0
    assert costs.half_spread_bps(19_999_999) == 7.0
    assert costs.half_spread_bps(0.0) == 7.0


def test_round_trip_adds_commission():
    assert costs.round_trip_cost_bps(500e6) == 3.0 + 0.5
    assert costs.round_trip_cost_bps(0.0) == 3.0 + 7.0


def test_cost_fraction_hand_computed():
    # Hand-computed, independent of the production formula: a $600M-ADV name
    # costs 3.5bp round trip = 0.00035 of notional.
    frac = costs.cost_fraction(600e6)
    assert abs(frac - 0.00035) < 1e-12


def test_vectorized_matches_scalar():
    advs = np.array([1e9, 300e6, 75e6, 25e6, 5e6])
    vec = costs.half_spread_bps(advs)
    scalar = np.array([costs.half_spread_bps(float(a)) for a in advs])
    assert np.allclose(vec, scalar)
