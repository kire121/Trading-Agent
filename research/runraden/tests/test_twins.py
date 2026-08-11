import numpy as np
import pandas as pd

from synth import make_iid_panel
from targets import attach_targets
from twins import build_market_basket, fit_market_timing_signal, reversal_signal, tsmom_signal
from words import build_asset_week_panel


def test_reversal_and_tsmom_are_opposite_sign():
    panel = pd.DataFrame({"week_return": [0.01, -0.02, 0.0]})
    rev = reversal_signal(panel)
    mom = tsmom_signal(panel)
    assert np.allclose(rev.to_numpy(), -mom.to_numpy())
    assert np.allclose(mom.to_numpy(), panel["week_return"].to_numpy())


def test_build_market_basket_is_equal_weighted_average():
    dates = pd.bdate_range("2021-01-04", periods=10)
    a = pd.Series(100 * (1.01 ** np.arange(10)), index=dates)
    b = pd.Series(100 * (0.99 ** np.arange(10)), index=dates)
    basket = build_market_basket({"A": a, "B": b})
    a_ret = a.pct_change().dropna()
    b_ret = b.pct_change().dropna()
    expected = (a_ret + b_ret) / 2.0  # equal-weighted average return series (index date[1:])

    # basket.iloc[0] anchors expected.iloc[0]; each later pct_change reproduces the rest.
    assert np.isclose(basket.iloc[0] - 1.0, expected.iloc[0], atol=1e-10)
    basket_ret_tail = basket.pct_change().dropna()
    assert np.allclose(basket_ret_tail.to_numpy(), expected.iloc[1:].to_numpy(), atol=1e-10)


def test_fit_market_timing_signal_runs_and_is_date_indexed():
    prices = make_iid_panel(n_assets=5, n_years=4, vol=0.01, seed=7)
    ghat_market = fit_market_timing_signal(prices, kappa=300.0, vol_window=60, burn_in_years=1)
    assert isinstance(ghat_market.index, pd.DatetimeIndex) or len(ghat_market) == 0
    if len(ghat_market) > 0:
        assert ghat_market.index.is_monotonic_increasing
