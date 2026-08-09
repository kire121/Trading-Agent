import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fasflocken.portfolio import (
    eligible_sectors,
    select_legs,
    HysteresisState,
    equal_dollar_direction,
    vol_target_scale,
    trailing_cov_matrix,
    compute_turnover,
    build_target_weights,
)
from fasflocken.config import GICS_SECTORS, SECTOR_TO_ETF


def test_eligible_sectors_gates_on_inception():
    before_xlre = eligible_sectors(dt.date(2010, 1, 1))
    assert "Real Estate" not in before_xlre
    assert "Communication Services" not in before_xlre
    assert "Technology" in before_xlre

    after_all = eligible_sectors(dt.date(2019, 1, 1))
    assert "Real Estate" in after_all
    assert "Communication Services" in after_all


def _z(sectors, values):
    return pd.Series(values, index=sectors)


def test_select_legs_basic_ranking():
    sectors = list(GICS_SECTORS)[:9]
    z = _z(sectors, [-2, -1.8, -1.5, -0.5, 0, 0.5, 1.2, 1.6, 2.0])
    state, longs, shorts = select_legs(z, HysteresisState(), n_legs=3, hysteresis_band=4)
    assert longs == sectors[:3]
    assert set(shorts) == set(sectors[-3:])


def test_select_legs_hysteresis_retention_and_exit():
    sectors = list(GICS_SECTORS)[:9]
    z0 = _z(sectors, [-2, -1.8, -1.5, -0.5, 0, 0.5, 1.2, 1.6, 2.0])
    state, longs, shorts = select_legs(z0, HysteresisState(), n_legs=3, hysteresis_band=4)
    incumbent = sectors[2]  # currently rank-3 long

    z1 = z0.copy()
    z1[incumbent] = z0[sectors[3]] + 0.01  # slip to rank 4, still inside the band
    state2, longs2, _ = select_legs(z1, state, n_legs=3, hysteresis_band=4)
    assert incumbent in longs2
    assert sectors[3] not in longs2

    z2 = z0.copy()
    z2[incumbent] = 5.0  # pushed fully out of band
    state3, longs3, _ = select_legs(z2, state, n_legs=3, hysteresis_band=4)
    assert incumbent not in longs3
    assert sectors[3] in longs3


def test_select_legs_drops_nan_and_ineligible():
    sectors = list(GICS_SECTORS)
    values = list(range(11))
    z = _z(sectors, values)
    z["Real Estate"] = np.nan
    elig = eligible_sectors(dt.date(2010, 1, 1))
    state, longs, shorts = select_legs(z, HysteresisState(), n_legs=3, hysteresis_band=4, eligible=elig)
    assert "Real Estate" not in longs and "Real Estate" not in shorts
    assert "Communication Services" not in longs and "Communication Services" not in shorts


def test_select_legs_raises_when_not_enough_eligible():
    z = _z(["Technology", "Financials"], [-1, 1])
    with pytest.raises(ValueError):
        select_legs(z, HysteresisState(), n_legs=3, hysteresis_band=4)


def test_equal_dollar_direction_is_dollar_neutral_and_gross_2():
    w = equal_dollar_direction(["Technology", "Financials", "Energy"], ["Utilities", "Materials", "Consumer Staples"])
    assert np.isclose(w.sum(), 0.0)
    assert np.isclose(w.abs().sum(), 2.0)
    assert np.isclose(w[SECTOR_TO_ETF["Technology"]], 1 / 3)
    assert np.isclose(w[SECTOR_TO_ETF["Utilities"]], -1 / 3)


def test_vol_target_scale_hits_target():
    tickers = ["XLK", "XLF", "XLB", "XLU"]
    w = pd.Series([1 / 2, 1 / 2, -1 / 2, -1 / 2], index=tickers)
    n = len(tickers)
    daily_vol, corr = 0.01, 0.2
    sigma = np.full((n, n), corr * daily_vol**2)
    np.fill_diagonal(sigma, daily_vol**2)
    cov = pd.DataFrame(sigma, index=tickers, columns=tickers)

    k = vol_target_scale(w, cov, target_vol_ann=0.08, max_gross=2.0)
    realized = np.sqrt((k * w.to_numpy()) @ sigma @ (k * w.to_numpy()) * 252)
    assert np.isclose(realized, 0.08, atol=1e-6)


def test_vol_target_scale_capped_by_max_gross():
    tickers = ["XLK", "XLF"]
    w = pd.Series([1.0, -1.0], index=tickers)  # gross = 2.0 already
    sigma = pd.DataFrame(np.eye(2) * (0.0001) ** 2, index=tickers, columns=tickers)  # tiny vol
    k = vol_target_scale(w, sigma, target_vol_ann=0.08, max_gross=2.0)
    assert k <= 1.0 + 1e-9  # base_gross == max_gross -> can only delever


def test_vol_target_scale_missing_ticker_returns_zero():
    w = pd.Series([1.0, -1.0], index=["XLK", "XLF"])
    cov = pd.DataFrame(np.eye(1), index=["XLK"], columns=["XLK"])
    assert vol_target_scale(w, cov, 0.08, 2.0) == 0.0


def test_compute_turnover_full_entry_and_flip():
    prev = pd.Series(dtype=float)
    new = pd.Series({"XLK": 1 / 3, "XLF": 1 / 3, "XLU": -1 / 3})
    assert np.isclose(compute_turnover(prev, new), 1.0)

    flipped = pd.Series({"XLK": -1 / 3, "XLF": 1 / 3, "XLU": -1 / 3})
    assert np.isclose(compute_turnover(new, flipped), 2 / 3)


def test_trailing_cov_matrix_is_point_in_time():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(size=(100, 2)), index=idx, columns=["XLK", "XLF"])
    as_of = idx[50]
    cov = trailing_cov_matrix(returns, as_of, window_days=60)
    assert cov.shape == (2, 2)
    manual = returns.loc[returns.index <= as_of].tail(60).cov()
    pd.testing.assert_frame_equal(cov, manual)


def test_build_target_weights_end_to_end():
    sectors = list(GICS_SECTORS)[:9]
    z = _z(sectors, [-2, -1.8, -1.5, -0.5, 0, 0.5, 1.2, 1.6, 2.0])
    etfs = [SECTOR_TO_ETF[s] for s in sectors]
    n = len(etfs)
    sigma = np.eye(n) * (0.01) ** 2
    cov = pd.DataFrame(sigma, index=etfs, columns=etfs)
    state, weights, info = build_target_weights(
        z, HysteresisState(), n_legs=3, hysteresis_band=4, cov_matrix=cov, target_vol_ann=0.08, max_gross=2.0
    )
    assert set(info["long_sectors"]) == set(sectors[:3])
    assert set(info["short_sectors"]) == set(sectors[-3:])
    assert np.isclose(weights.sum(), 0.0)
