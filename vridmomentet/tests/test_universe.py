from __future__ import annotations

import datetime as dt
import os

import pytest

from vridmomentet.universe import (
    DataProviderNotConfigured,
    EODHDProvider,
    MembershipInterval,
    NorgateProvider,
    PointInTimeMembership,
    SharadarProvider,
)


class TestMembershipInterval:
    def test_covers_is_start_inclusive_end_exclusive(self):
        iv = MembershipInterval(ticker="A", source="sp500", start=dt.date(2020, 1, 1), end=dt.date(2021, 1, 1), point_in_time=True)
        assert iv.covers(dt.date(2020, 1, 1)) is True
        assert iv.covers(dt.date(2020, 6, 1)) is True
        assert iv.covers(dt.date(2020, 12, 31)) is True
        assert iv.covers(dt.date(2021, 1, 1)) is False
        assert iv.covers(dt.date(2019, 12, 31)) is False


class TestPointInTimeMembership:
    def _sample(self) -> PointInTimeMembership:
        return PointInTimeMembership([
            MembershipInterval("A", "sp500", dt.date(2000, 1, 1), dt.date(2020, 1, 1), True),
            MembershipInterval("B", "sp500", dt.date(2010, 1, 1), dt.date(9999, 1, 1), True),
            MembershipInterval("C", "sp400", dt.date(2000, 1, 1), dt.date(9999, 1, 1), False),
            # A rejoins later under a second, non-adjoining interval.
            MembershipInterval("A", "sp500", dt.date(2022, 1, 1), dt.date(9999, 1, 1), True),
        ])

    def test_constituents_as_of_reflects_point_in_time_membership(self):
        mem = self._sample()
        assert set(mem.constituents_as_of(dt.date(2005, 1, 1))) == {"A", "C"}  # A: [2000,2020), B not yet in
        assert set(mem.constituents_as_of(dt.date(2015, 1, 1))) == {"A", "B", "C"}  # both A and B now in
        assert set(mem.constituents_as_of(dt.date(2023, 1, 1))) == {"A", "B", "C"}  # A back via 2nd spell

    def test_delisted_name_drops_out_after_its_end_date(self):
        mem = self._sample()
        assert "A" not in mem.constituents_as_of(dt.date(2021, 1, 1))

    def test_readded_name_reappears_via_second_interval(self):
        mem = self._sample()
        assert "A" in mem.constituents_as_of(dt.date(2023, 6, 1))
        assert "A" not in mem.constituents_as_of(dt.date(2021, 1, 1))

    def test_all_tickers_is_deduplicated(self):
        mem = self._sample()
        assert mem.all_tickers() == ["A", "B", "C"]

    def test_tickers_from_source_splits_sp500_and_sp400(self):
        mem = self._sample()
        assert mem.tickers_from_source("sp500") == {"A", "B"}
        assert mem.tickers_from_source("sp400") == {"C"}

    def test_intervals_for_returns_both_spells_of_a_readded_ticker(self):
        mem = self._sample()
        assert len(mem.intervals_for("A")) == 2

    def test_len_counts_intervals_not_unique_tickers(self):
        mem = self._sample()
        assert len(mem) == 4


class TestProviderStubsFailFast:
    def test_norgate_stub_raises_with_actionable_message(self):
        with pytest.raises(DataProviderNotConfigured, match="Norgate"):
            NorgateProvider()

    def test_sharadar_stub_raises_with_actionable_message(self):
        with pytest.raises(DataProviderNotConfigured, match="Sharadar"):
            SharadarProvider()


class TestEODHDProviderConfiguration:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("EODHD_API_KEY", raising=False)
        monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
        with pytest.raises(DataProviderNotConfigured, match="EODHD_API_KEY"):
            EODHDProvider(cache_dir="/tmp/vridmomentet_test_cache_unused")

    def test_accepts_explicit_api_token_without_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EODHD_API_KEY", raising=False)
        monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
        provider = EODHDProvider(api_token="dummy-token-for-test", cache_dir=str(tmp_path))
        assert provider.api_token == "dummy-token-for-test"

    def test_reads_api_key_from_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EODHD_API_KEY", "env-token")
        provider = EODHDProvider(cache_dir=str(tmp_path))
        assert provider.api_token == "env-token"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("EODHD_API_KEY"), reason="requires a real EODHD_API_KEY")
class TestEODHDProviderLive:
    def test_membership_returns_point_in_time_sp500_and_current_sp400(self, tmp_path):
        provider = EODHDProvider(cache_dir=str(tmp_path))
        mem = provider.membership()
        assert len(mem.tickers_from_source("sp500")) > 400
        assert len(mem.tickers_from_source("sp400")) > 300
        # A name known to have left the S&P 500 must still appear historically.
        assert "AET" in mem.constituents_as_of(dt.date(2015, 1, 1))
        assert "AET" not in mem.constituents_as_of(dt.date(2020, 1, 1))

    def test_prices_returns_ohlcv_with_volume(self, tmp_path):
        provider = EODHDProvider(cache_dir=str(tmp_path))
        px = provider.prices(["AAPL"], dt.date(2023, 1, 1), dt.date(2023, 1, 31))
        assert "AAPL" in px
        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            assert col in px["AAPL"].columns
        assert (px["AAPL"]["volume"] > 0).all()
