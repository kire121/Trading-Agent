"""
Live integration tests against the real EODHD API.

Skipped entirely unless EODHD_API_KEY (or EODHD_API_TOKEN) is set --
these make real network calls and are not part of the offline/synthetic
test suite. They exist to validate universe.EODHDProvider's actual
parsing/mapping logic against real API responses, not to re-verify the
signal/backtest math (that's covered by the synthetic-data suite).

Uses a module-scoped, disposable on-disk cache so repeated assertions in
this file share fetches instead of re-hitting the API per test.
"""

import datetime as dt
import os

import pytest

from fasflocken.universe import EODHDProvider

pytestmark = pytest.mark.skipif(
    not (os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_API_TOKEN")),
    reason="requires a real EODHD_API_KEY/EODHD_API_TOKEN",
)


@pytest.fixture(scope="module")
def live_provider(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("eodhd_live_cache")
    return EODHDProvider(cache_dir=str(cache_dir), request_delay=0.02)


def test_live_membership_is_point_in_time(live_provider):
    # Medco Health Solutions: S&P 500 constituent up to its 2012-04-04
    # removal, not after. (Not Lehman Brothers -- EODHD's fundamentals
    # record for that one returns GicSector="NA" even historically, so it's
    # excluded by the documented sector-coverage gap this test isn't
    # meant to exercise; see EODHDProvider's docstring / sector_coverage().)
    before = live_provider.constituents("Health Care", dt.date(2012, 1, 1))
    after = live_provider.constituents("Health Care", dt.date(2013, 1, 1))
    assert "MHS" in before
    assert "MHS" not in after


def test_live_gics_sector_mapping(live_provider):
    # Cross-checks _GICSECTOR_TO_OURS and the raw GicSector passthrough.
    tech = live_provider.constituents("Technology", dt.date(2024, 1, 1))
    financials = live_provider.constituents("Financials", dt.date(2024, 1, 1))
    energy = live_provider.constituents("Energy", dt.date(2024, 1, 1))
    assert "AAPL" in tech
    assert "JPM" in financials
    assert "XOM" in energy
    # sanity: no ticker double-counted into an unrelated sector at the same date
    assert "AAPL" not in financials


def test_live_eod_prices_are_sane(live_provider):
    prices = live_provider.prices(["AAPL"], dt.date(2023, 1, 1), dt.date(2023, 1, 31))
    assert "AAPL" in prices.columns
    assert not prices.columns.duplicated().any()  # regression: adjusted_close/close rename collision
    series = prices["AAPL"]
    assert series.ndim == 1  # regression: duplicate "close" label makes single-bracket selection a DataFrame
    series = series.dropna()
    assert len(series) >= 15  # ~20 trading days in January, allow some slack
    assert (series > 0).all()
    assert series.max() / series.min() < 2.0  # no absurd jump in one month


def test_live_sector_etf_prices(live_provider):
    prices = live_provider.sector_etf_prices(dt.date(2023, 1, 1), dt.date(2023, 3, 31), etfs=["XLK", "XLF"])
    assert set(prices.columns) == {"XLK", "XLF"}
    assert (prices.dropna() > 0).all().all()


def test_live_trading_calendar_excludes_weekends_and_holidays(live_provider):
    cal = live_provider.trading_calendar(dt.date(2023, 1, 1), dt.date(2023, 1, 10))
    assert (cal.weekday < 5).all()
    # 2023-01-02 was the observed New Year's Day holiday (market closed)
    assert dt.datetime(2023, 1, 2) not in cal


def test_live_sector_coverage_diagnostics(live_provider):
    coverage = live_provider.sector_coverage()
    assert coverage["covered"] > 400  # most of the ~800 historical S&P names resolve
    assert coverage["covered"] > coverage["dropped"]
