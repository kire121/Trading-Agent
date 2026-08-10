import numpy as np
import pandas as pd
import pytest

from research.dammluckan import signal, data, config


def _toy_panel(prices, dates=None, ticker="TST"):
    n = len(prices)
    if dates is None:
        dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({ticker: prices}, index=dates)
    vol = pd.DataFrame({ticker: [1_000_000.0] * n}, index=dates)
    return data.Panel(tickers=[ticker], raw_close=close, raw_open=close, high=close, low=close,
                       adj_close=close, adj_open=close, volume=vol)


def test_m_high_excludes_today():
    # A strictly increasing series: M_t at position t must be the max of the
    # PRECEDING n values only, so a new all-time high always satisfies E+.
    prices = np.arange(1.0, 51.0)
    panel = _toy_panel(prices)
    m = signal.rolling_max_excl_today(panel.raw_close, n=5)["TST"]
    # At position 10 (0-indexed), preceding 5 values are prices[5:10] = 6..10
    assert m.iloc[10] == prices[9]  # max of prices[5:10] (0-indexed) == prices[9]
    # Today's own price must never be included: mutate prices[10] and confirm
    # m.iloc[10] (which depends only on prices[5:10]) is unaffected.
    prices2 = prices.copy()
    prices2[10] = 999.0
    panel2 = _toy_panel(prices2)
    m2 = signal.rolling_max_excl_today(panel2.raw_close, n=5)["TST"]
    assert m2.iloc[10] == m.iloc[10]


def test_no_lookahead_prefix_invariance():
    """Mutating price strictly AFTER date t must not change M_t/E+_t/O+_t
    computed at t -- the central causality guarantee of the whole signal."""
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    panel_a = _toy_panel(prices)

    prices_b = prices.copy()
    prices_b[200:] = prices_b[200:] * 1.5 + rng.normal(0, 5, 100)  # scramble everything from t=200 on
    panel_b = _toy_panel(prices_b)

    n, c = 60, 1.0
    band_vol_a = panel_a.band_vol.shift(1)
    band_vol_b = panel_b.band_vol.shift(1)

    m_a = signal.rolling_max_excl_today(panel_a.raw_close, n)["TST"]
    m_b = signal.rolling_max_excl_today(panel_b.raw_close, n)["TST"]
    assert np.allclose(m_a.iloc[:200], m_b.iloc[:200], equal_nan=True)

    o_a = signal.occupation_high(panel_a.raw_close, band_vol_a, n, c)["TST"]
    o_b = signal.occupation_high(panel_b.raw_close, band_vol_b, n, c)["TST"]
    # band_vol at t=199 depends on returns through t=199 only (shift(1) means
    # it uses data through t-1=198), unaffected by the t>=200 mutation.
    assert np.allclose(o_a.iloc[:199], o_b.iloc[:199], equal_nan=True)


def test_occupation_bounds():
    """O+ is always a fraction in [0, 1] wherever it's defined, and a
    perfectly flat price series has O+ == 1 (every day sits at the ceiling)."""
    prices = np.full(200, 100.0)
    panel = _toy_panel(prices)
    band_vol = panel.band_vol.shift(1).fillna(0.0)
    o = signal.occupation_high(panel.raw_close, band_vol, n=50, c=1.0)["TST"].dropna()
    assert (o >= 0).all() and (o <= 1).all()
    assert np.allclose(o, 1.0)


def test_occupation_monotone_in_c():
    """Widening the band (larger c) can only ever increase or hold O+ fixed,
    never decrease it -- more days qualify as the band widens."""
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    panel = _toy_panel(prices)
    band_vol = panel.band_vol.shift(1)
    o_narrow = signal.occupation_high(panel.raw_close, band_vol, n=60, c=0.5)["TST"]
    o_wide = signal.occupation_high(panel.raw_close, band_vol, n=60, c=3.0)["TST"]
    valid = o_narrow.notna() & o_wide.notna()
    assert (o_wide[valid] >= o_narrow[valid] - 1e-12).all()


def test_record_event_matches_direct_comparison():
    rng = np.random.default_rng(2)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 250)))
    panel = _toy_panel(prices)
    n = 40
    m = signal.rolling_max_excl_today(panel.raw_close, n)
    e = signal.record_events_high(panel.raw_close, m)["TST"]
    px = panel.raw_close["TST"]
    for t in range(n + 1, len(prices)):
        expected = float(px.iloc[t] > m["TST"].iloc[t])
        assert e.iloc[t] == expected
