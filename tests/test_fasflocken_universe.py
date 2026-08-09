import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fasflocken.universe import (
    SyntheticUniverseProvider,
    PointInTimeMembership,
    MembershipInterval,
    UniverseProvider,
    constant_coherence,
    step_coherence,
    NorgateProvider,
    SharadarProvider,
    EODHDProvider,
    DataProviderNotConfigured,
)


def test_point_in_time_membership_reclassification():
    intervals = [
        MembershipInterval("AAA", "Telecom", dt.date(2010, 1, 1), dt.date(2018, 9, 24)),
        MembershipInterval("AAA", "Communication Services", dt.date(2018, 9, 24), dt.date(9999, 1, 1)),
    ]
    m = PointInTimeMembership(intervals)
    assert m.sector_of("AAA", dt.date(2015, 1, 1)) == "Telecom"
    assert m.sector_of("AAA", dt.date(2018, 9, 24)) == "Communication Services"
    assert m.sector_of("AAA", dt.date(2018, 9, 23)) == "Telecom"
    assert "AAA" in m.constituents_as_of("Communication Services", dt.date(2020, 1, 1))
    assert "AAA" not in m.constituents_as_of("Telecom", dt.date(2020, 1, 1))


def test_synthetic_provider_trading_calendar_and_prices():
    start, end = dt.date(2020, 1, 1), dt.date(2020, 6, 30)
    provider = SyntheticUniverseProvider(start=start, end=end, n_per_sector=5, seed=1, sectors=("Technology", "Energy"))
    dates = provider.trading_calendar(start, end)
    assert (dates.weekday < 5).all()

    tickers = provider.constituents("Technology", end)
    assert len(tickers) == 5
    prices = provider.prices(tickers, start, end)
    assert prices.shape == (len(dates), 5)
    assert (prices > 0).all().all()

    etf_prices = provider.sector_etf_prices(start, end)
    assert "XLK" in etf_prices.columns
    assert "XLE" in etf_prices.columns


def test_synthetic_provider_deterministic_with_seed():
    start, end = dt.date(2020, 1, 1), dt.date(2020, 3, 31)
    p1 = SyntheticUniverseProvider(start=start, end=end, n_per_sector=4, seed=42, sectors=("Technology",))
    p2 = SyntheticUniverseProvider(start=start, end=end, n_per_sector=4, seed=42, sectors=("Technology",))
    px1 = p1.sector_etf_prices(start, end)
    px2 = p2.sector_etf_prices(start, end)
    pd.testing.assert_frame_equal(px1, px2)


def test_synthetic_provider_coherence_schedule_raises_kuramoto_R():
    from fasflocken.signals import causal_bandpass, rolling_analytic_phase, kuramoto_order_parameter

    start, end = dt.date(2018, 1, 1), dt.date(2020, 12, 31)
    sector = "Technology"
    provider = SyntheticUniverseProvider(
        start=start, end=end, n_per_sector=20, seed=9, sectors=(sector,),
        coherence_schedules={sector: step_coherence(0.1, 0.98, switch_frac=0.5)},
    )
    tickers = provider.constituents(sector, end)
    prices = provider.prices(tickers, start, end)
    returns = np.log(prices).diff()
    bp = causal_bandpass(returns, 5, 20)
    phase = rolling_analytic_phase(bp, window=90)
    R = kuramoto_order_parameter(phase, min_constituents=5)
    n = len(R)
    early = np.nanmean(R[: n // 4])
    late = np.nanmean(R[-n // 4 :])
    assert late > early  # coherence step-up should visibly raise R


def test_constant_coherence_schedule_shape():
    dates = pd.bdate_range("2020-01-01", periods=10)
    sched = constant_coherence(0.5)
    out = sched(dates)
    assert len(out) == 10
    assert np.all(out == 0.5)


@pytest.mark.parametrize("cls", [NorgateProvider, SharadarProvider])
def test_stub_providers_fail_clearly_without_credentials(cls):
    with pytest.raises(DataProviderNotConfigured):
        cls()


def test_eodhd_fails_clearly_without_credentials(monkeypatch):
    # Hermetic regardless of the ambient environment: this repo's own dev
    # session has a real EODHD_API_KEY set, so explicitly clear it here
    # rather than relying on it being absent.
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    with pytest.raises(DataProviderNotConfigured):
        EODHDProvider()


def test_eodhd_constructs_with_explicit_token(tmp_path):
    provider = EODHDProvider(api_token="dummy-token-not-a-real-key", cache_dir=str(tmp_path))
    assert provider.api_token == "dummy-token-not-a-real-key"
    assert provider.session is not None


def test_eodhd_constructs_from_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("EODHD_API_KEY", "dummy-token-from-env")
    provider = EODHDProvider(cache_dir=str(tmp_path))
    assert provider.api_token == "dummy-token-from-env"


class _DynamicMembershipProvider(UniverseProvider):
    """Minimal provider whose sector membership changes mid-sample, to test
    that pipeline.build_membership_mask correctly excludes a name before it
    joins / after it leaves, independent of price availability.
    """

    def __init__(self, dates, tickers, join_dates, leave_dates):
        self.dates = dates
        self.tickers = tickers
        self.join_dates = join_dates
        self.leave_dates = leave_dates
        rng = np.random.default_rng(0)
        self._prices = pd.DataFrame(
            {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))) for t in tickers}, index=dates
        )

    def constituents(self, sector, as_of):
        out = []
        for t in self.tickers:
            if self.join_dates[t] <= as_of < self.leave_dates[t]:
                out.append(t)
        return out

    def prices(self, tickers, start, end):
        return self._prices.loc[str(start):str(end), list(tickers)]

    def sector_etf_prices(self, start, end, etfs=None):
        raise NotImplementedError

    def trading_calendar(self, start, end):
        return self.dates[(self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))]


def test_membership_mask_excludes_before_join_and_after_leave():
    from fasflocken.pipeline import build_membership_mask

    dates = pd.bdate_range("2020-01-01", periods=100)
    tickers = ["A", "B"]
    provider = _DynamicMembershipProvider(
        dates, tickers,
        join_dates={"A": dt.date(2020, 1, 1), "B": dt.date(2020, 3, 1)},
        leave_dates={"A": dt.date(2020, 4, 1), "B": dt.date(9999, 1, 1)},
    )
    mask = build_membership_mask(provider, "sector", tickers, dates, membership_refresh_days=1)
    assert mask.loc[dates[0], "A"] == True  # noqa: E712
    assert mask.loc[dates[0], "B"] == False  # noqa: E712
    assert mask.loc[dates[-1], "A"] == False  # noqa: E712
    assert mask.loc[dates[-1], "B"] == True  # noqa: E712
