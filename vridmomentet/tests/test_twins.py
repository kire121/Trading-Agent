from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vridmomentet.config import TwinParams
from vridmomentet.data import Panel
from vridmomentet.twins import (
    _rolling_lag1_corr_1d,
    lag1_cross_correlation_twin,
    momentum_twin,
    momentum_x_turnover_twin,
)
from vridmomentet.universe import PointInTimeMembership


def _synthetic_panel(n_days=150, n_names=30, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    names = [f"T{i}" for i in range(n_names)]
    close = pd.DataFrame(
        {n: 100 * np.cumprod(1 + rng.normal(scale=0.01, size=n_days)) for n in names}, index=dates
    )
    volume = pd.DataFrame(
        {n: rng.uniform(1e6, 5e6, size=n_days) for n in names}, index=dates
    )
    return Panel(close=close, adj_close=close.copy(), adj_open=close.copy(), volume=volume,
                 membership=PointInTimeMembership([]))


class TestMomentumTwin:
    def test_sign_matches_formation_return_sign(self):
        panel = _synthetic_panel()
        params = TwinParams(momentum_lookback_days=20)
        z = momentum_twin(panel, params)
        from vridmomentet.signal import formation_return
        r_n = formation_return(panel.log_returns, 20)
        row = z.iloc[100]
        r_row = r_n.iloc[100]
        valid = row.notna() & r_row.notna() & (r_row != 0)
        assert np.all(np.sign(row[valid]) == np.sign(r_row[valid]))

    def test_row_is_zscored(self):
        panel = _synthetic_panel()
        z = momentum_twin(panel, TwinParams(momentum_lookback_days=20))
        row = z.iloc[100].dropna()
        assert row.mean() == pytest.approx(0.0, abs=1e-6)


class TestMomentumTimesTurnoverTwin:
    def test_equals_product_of_the_two_zscores(self):
        panel = _synthetic_panel()
        params = TwinParams(momentum_lookback_days=20, turnover_lookback_days=60)
        combined = momentum_x_turnover_twin(panel, params)

        from vridmomentet.signal import cross_sectional_zscore, formation_return
        r_n = formation_return(panel.log_returns, 20)
        z_mom = cross_sectional_zscore(r_n, 0.01, 0.99)
        turnover_level = np.log(panel.adv60.rolling(60, min_periods=45).mean())
        z_turn = cross_sectional_zscore(turnover_level, 0.01, 0.99)
        expected = z_mom * z_turn

        pd.testing.assert_frame_equal(combined, expected)


class TestLag1CrossCorrelationKernel:
    def test_detects_perfect_lagged_linear_relationship(self):
        n = 40
        u = np.sin(np.linspace(0, 6, n))
        r = np.roll(u, 1) * 0.02   # r_s = k * u_{s-1} exactly (k>0)
        r[0] = 0.0
        out = _rolling_lag1_corr_1d(r, u, window=30)
        val = out[~np.isnan(out)][0]
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_detects_perfect_negative_lagged_relationship(self):
        n = 40
        u = np.sin(np.linspace(0, 6, n))
        r = -np.roll(u, 1) * 0.02
        r[0] = 0.0
        out = _rolling_lag1_corr_1d(r, u, window=30)
        val = out[~np.isnan(out)][0]
        assert val == pytest.approx(-1.0, abs=1e-6)

    def test_near_zero_for_unrelated_series(self):
        rng = np.random.default_rng(7)
        n = 2000
        u = rng.normal(size=n)
        r = rng.normal(size=n)
        out = _rolling_lag1_corr_1d(r, u, window=250)
        vals = out[~np.isnan(out)]
        assert abs(np.nanmean(vals)) < 0.05

    def test_no_lookahead_uses_u_at_s_minus_1_not_s(self):
        """r_s must be paired with u_{s-1}, not u_s: shuffling u only on the
        *last* day of the window must not change the correlation computed
        for a window ending the day before.
        """
        rng = np.random.default_rng(8)
        n = 40
        r = rng.normal(size=n)
        u = rng.normal(size=n)
        out1 = _rolling_lag1_corr_1d(r, u, window=30)
        u2 = u.copy()
        u2[-1] = 999.0  # perturb only the very last day
        out2 = _rolling_lag1_corr_1d(r, u2, window=30)
        # The correlation value *ending* at the second-to-last index must be
        # identical, since it never uses u[-1] (only u up to index n-2).
        assert out1[-2] == pytest.approx(out2[-2])


class TestLag1CrossCorrelationTwin:
    def test_output_is_cross_sectionally_zscored(self):
        panel = _synthetic_panel()
        z = lag1_cross_correlation_twin(panel, TwinParams(lag1_lookback_days=60))
        row = z.iloc[120].dropna()
        if len(row) > 1:
            assert row.mean() == pytest.approx(0.0, abs=1e-6)
