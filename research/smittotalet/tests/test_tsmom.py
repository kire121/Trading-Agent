from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from .. import config
from .. import tsmom


def _synthetic_panel(n_days=1000, n_assets=6, seed=0):
    idx = pd.date_range("2010-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    cols = [f"A{i}" for i in range(n_assets)]
    close = pd.DataFrame(prices, index=idx, columns=cols)
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, size=(n_days, n_assets)),
                           index=idx, columns=cols).astype(float)

    @dataclass
    class P:
        open: pd.DataFrame
        high: pd.DataFrame
        low: pd.DataFrame
        close: pd.DataFrame
        adjusted_close: pd.DataFrame
        volume: pd.DataFrame

        def simple_returns(self):
            return self.adjusted_close.pct_change()

        def dollar_volume(self):
            return self.close * self.volume

        def adv(self, lookback=config.ADV_COST_LOOKBACK):
            return self.dollar_volume().shift(1).rolling(lookback).mean()

    return P(open=close, high=close, low=close, close=close, adjusted_close=close, volume=volume)


def test_weekly_rebalanced_position_has_no_lookahead():
    panel = _synthetic_panel()
    pos = tsmom.weekly_rebalanced_position(panel)
    raw = tsmom.raw_signal(panel)
    # every row of `pos` must equal SOME earlier row of `raw` (same week's
    # Friday-close signal, shifted one week) -- not today's raw signal
    mismatches = 0
    for date in pos.index[-30:]:
        if not pos.loc[date].isna().all() and not raw.loc[date].isna().all():
            if (pos.loc[date] == raw.loc[date]).all():
                mismatches += 1
    assert mismatches == 0


def test_apply_gross_cap_never_exceeds_cap():
    panel = _synthetic_panel()
    raw = tsmom.weekly_rebalanced_position(panel)
    weights = tsmom.apply_gross_cap(raw, k=1000.0, gross_cap=2.0)  # deliberately huge k
    gross = weights.abs().sum(axis=1).dropna()
    assert (gross <= 2.0 + 1e-9).all()


def test_solve_k_hits_target_vol_when_cap_never_binds():
    panel = _synthetic_panel(n_days=1200, n_assets=8)
    is_start, is_end = panel.close.index[300], panel.close.index[900]
    k = tsmom.solve_k_for_target_vol(panel, is_start, is_end, target_vol=0.10, gross_cap=200.0)
    weights = tsmom.apply_gross_cap(tsmom.weekly_rebalanced_position(panel), k, gross_cap=200.0)
    rets = tsmom.portfolio_returns(panel, weights, apply_costs=False).loc[is_start:is_end]
    achieved_vol = rets.std() * np.sqrt(config.TRADING_DAYS_YEAR)
    assert achieved_vol == pytest.approx(0.10, rel=0.15)


def test_solve_k_respects_binding_gross_cap():
    panel = _synthetic_panel(n_days=1200, n_assets=8)
    is_start, is_end = panel.close.index[300], panel.close.index[900]
    # a very tight cap forces binding on most weeks -> achieved vol should be
    # BELOW an unreasonably high target, not blow up
    k = tsmom.solve_k_for_target_vol(panel, is_start, is_end, target_vol=5.0, gross_cap=0.5)
    weights = tsmom.apply_gross_cap(tsmom.weekly_rebalanced_position(panel), k, gross_cap=0.5)
    gross = weights.abs().sum(axis=1).dropna()
    assert (gross <= 0.5 + 1e-6).all()
