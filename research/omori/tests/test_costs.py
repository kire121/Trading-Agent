import numpy as np

from research.omori import config, costs


class TestHalfSpreadBps:
    def test_most_liquid_bucket(self):
        assert costs.half_spread_bps(600e6) == 0.5

    def test_least_liquid_bucket(self):
        assert costs.half_spread_bps(1e6) == 7.0

    def test_bucket_boundaries_exact(self):
        assert costs.half_spread_bps(500e6) == 0.5
        assert costs.half_spread_bps(499.99e6) == 1.0

    def test_nan_adv_gets_worst_bucket(self):
        out = costs.half_spread_bps(np.array([np.nan]))
        assert out[0] == 7.0

    def test_vectorized_matches_scalar(self):
        advs = np.array([600e6, 300e6, 150e6, 60e6, 25e6, 1e6])
        expected = [0.5, 1.0, 1.5, 2.5, 4.0, 7.0]
        out = costs.half_spread_bps(advs)
        assert list(out) == expected


class TestRoundTripCost:
    def test_includes_commission(self):
        bps = costs.round_trip_cost_bps(600e6)
        assert bps == config.COMMISSION_BPS + 0.5

    def test_trade_cost_return_is_bps_over_1e4(self):
        r = costs.trade_cost_return(600e6)
        assert r == (config.COMMISSION_BPS + 0.5) / 1e4
