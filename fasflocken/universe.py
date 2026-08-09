"""
Fasflocken (PH-1) -- point-in-time universe & data-provider abstraction.

The strategy trades 11 liquid sector ETFs, but the *signal* is built on
sector constituents (Kuramoto order parameter across N names per sector),
so we need point-in-time S&P 500 membership + GICS sector history. Getting
this wrong -- e.g. using today's constituent list for a 2006 backtest, or
ignoring the Sept-2018 GICS reclassification -- is the classic survivorship
/ look-ahead trap flagged in the spec.

This module defines:
  * `UniverseProvider`      -- the abstract interface every data source
                                implements (constituents / prices / ETF
                                prices / trading calendar).
  * `NorgateProvider`, `SharadarProvider`, `EODHDProvider`
                              -- thin real-vendor stubs. None of these
                                 vendors are reachable from this sandbox
                                 (no credentials, no vendor packages
                                 installed), so each raises
                                 `DataProviderNotConfigured` with the exact
                                 package + call a user needs to finish
                                 wiring, rather than silently returning
                                 fabricated data.
  * `SyntheticUniverseProvider`
                              -- a fully deterministic, seeded generator
                                 used by the test suite and by run.py's
                                 demo mode. Exposes a per-sector
                                 "coherence schedule" so tests can force a
                                 sector into a known phase-locked or
                                 phase-random regime and check the signal
                                 pipeline reacts correctly.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from fasflocken.config import GICS_SECTORS


class DataProviderNotConfigured(RuntimeError):
    """Raised by real-vendor providers when credentials/packages are missing."""


# ---------------------------------------------------------------------------
# Point-in-time membership bookkeeping shared by every provider.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MembershipInterval:
    ticker: str
    sector: str
    start: _dt.date
    end: _dt.date  # exclusive; use date.max sentinel-free by convention: still-current -> far future

    def covers(self, as_of: _dt.date) -> bool:
        return self.start <= as_of < self.end


class PointInTimeMembership:
    """A flat list of (ticker, sector, [start, end)) intervals.

    Reclassifications are just two intervals for the same ticker with
    different sectors and adjoining dates -- e.g. a name that moved from
    Telecom to Communication Services on 2018-09-24 has one interval
    ending then and a new one starting then.
    """

    def __init__(self, intervals: Sequence[MembershipInterval]):
        self._intervals = list(intervals)

    def constituents_as_of(self, sector: str, as_of: _dt.date) -> list[str]:
        return sorted(
            {iv.ticker for iv in self._intervals if iv.sector == sector and iv.covers(as_of)}
        )

    def sector_of(self, ticker: str, as_of: _dt.date) -> str | None:
        for iv in self._intervals:
            if iv.ticker == ticker and iv.covers(as_of):
                return iv.sector
        return None

    def all_tickers(self) -> list[str]:
        return sorted({iv.ticker for iv in self._intervals})


# ---------------------------------------------------------------------------
# Abstract provider interface.
# ---------------------------------------------------------------------------

class UniverseProvider(ABC):
    """Everything the signal pipeline needs, point-in-time."""

    @abstractmethod
    def constituents(self, sector: str, as_of: _dt.date) -> list[str]:
        """S&P 500 tickers classified under `sector` as of `as_of` (point-in-time)."""

    @abstractmethod
    def prices(self, tickers: Sequence[str], start: _dt.date, end: _dt.date) -> pd.DataFrame:
        """Wide DataFrame of daily close prices, index=trading dates, columns=tickers.

        Must NOT forward-fill across gaps that represent "not yet listed" /
        "delisted" -- those should stay NaN so downstream code can exclude
        the name rather than fabricate a flat price.
        """

    @abstractmethod
    def sector_etf_prices(self, start: _dt.date, end: _dt.date, etfs: Sequence[str] | None = None) -> pd.DataFrame:
        """Wide DataFrame of daily close prices for the tradable sector ETFs."""

    @abstractmethod
    def trading_calendar(self, start: _dt.date, end: _dt.date) -> pd.DatetimeIndex:
        """Trading dates in [start, end]."""


# ---------------------------------------------------------------------------
# Real-vendor stubs. Each documents exactly what's needed to finish it.
# ---------------------------------------------------------------------------

class NorgateProvider(UniverseProvider):
    """Norgate Data via the `norgatedata` package (requires local NDU + license).

    Wiring sketch (not executable here -- no NDU install/license in this
    sandbox):
        import norgatedata
        norgatedata.watchlist_symbols('S&P 500 Current & Past')
        norgatedata.classification_at_date(ticker, 'GICS-Sector', 'Name', as_of)
        norgatedata.price_timeseries(ticker, start_date=..., end_date=..., ...)
    """

    def __init__(self, *_, **__):
        raise DataProviderNotConfigured(
            "Norgate requires the local Norgate Data Updater (NDU) running plus a "
            "licensed `norgatedata` package. Neither is available in this "
            "environment. Install `norgatedata`, run NDU, then implement the "
            "methods below using norgatedata.classification_at_date / "
            "norgatedata.price_timeseries / norgatedata.watchlist_symbols."
        )

    def constituents(self, sector, as_of):  # pragma: no cover - unreachable without vendor
        raise NotImplementedError

    def prices(self, tickers, start, end):  # pragma: no cover
        raise NotImplementedError

    def sector_etf_prices(self, start, end, etfs=None):  # pragma: no cover
        raise NotImplementedError

    def trading_calendar(self, start, end):  # pragma: no cover
        raise NotImplementedError


class SharadarProvider(UniverseProvider):
    """Sharadar Core US Equities Bundle via `nasdaqdatalink` (needs paid API key).

    Wiring sketch:
        import nasdaqdatalink as ndl
        ndl.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
        ndl.get_table('SHARADAR/TICKERS', table='SF1')       # sector metadata
        ndl.get_table('SHARADAR/SEP', ticker=tickers, ...)   # daily prices
        ndl.get_table('SHARADAR/SF3', ...)                   # index membership history
    """

    def __init__(self, *_, **__):
        raise DataProviderNotConfigured(
            "Sharadar requires a paid Nasdaq Data Link subscription and the "
            "`nasdaqdatalink` package with NASDAQ_DATA_LINK_API_KEY set. Neither "
            "is configured in this environment."
        )

    def constituents(self, sector, as_of):  # pragma: no cover
        raise NotImplementedError

    def prices(self, tickers, start, end):  # pragma: no cover
        raise NotImplementedError

    def sector_etf_prices(self, start, end, etfs=None):  # pragma: no cover
        raise NotImplementedError

    def trading_calendar(self, start, end):  # pragma: no cover
        raise NotImplementedError


class EODHDProvider(UniverseProvider):
    """EODHD (budget alternative) via plain REST calls.

    Wiring sketch:
        GET https://eodhd.com/api/fundamentals/{ticker}.US?api_token=...   # GICS sector
        GET https://eodhd.com/api/eod/{ticker}.US?api_token=...&fmt=json  # OHLCV
        GET https://eodhd.com/api/index_constituents/GSPC.INDX?...        # current constituents only;
            point-in-time history is NOT available on EODHD's standard tier and must be
            reconstructed from historical index-addition/removal press releases.
    """

    def __init__(self, api_token: str | None = None, *_, **__):
        raise DataProviderNotConfigured(
            "EODHD requires EODHD_API_TOKEN and does not provide point-in-time S&P "
            "500 membership on its standard tier (only current constituents) -- "
            "the point-in-time history would need to be reconstructed separately. "
            "Neither the token nor that reconstruction is available in this "
            "environment."
        )

    def constituents(self, sector, as_of):  # pragma: no cover
        raise NotImplementedError

    def prices(self, tickers, start, end):  # pragma: no cover
        raise NotImplementedError

    def sector_etf_prices(self, start, end, etfs=None):  # pragma: no cover
        raise NotImplementedError

    def trading_calendar(self, start, end):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Synthetic provider -- deterministic, seeded, used by tests and demo runs.
# ---------------------------------------------------------------------------

CoherenceSchedule = Callable[[pd.DatetimeIndex], np.ndarray]


def constant_coherence(level: float) -> CoherenceSchedule:
    """Coherence schedule that is constant at `level` in [0, 1]."""

    def _sched(dates: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(dates), level, dtype=float)

    return _sched


def step_coherence(level_before: float, level_after: float, switch_frac: float = 0.5) -> CoherenceSchedule:
    """Step schedule: `level_before` for the first `switch_frac` of the window, then `level_after`."""

    def _sched(dates: pd.DatetimeIndex) -> np.ndarray:
        n = len(dates)
        cut = int(n * switch_frac)
        out = np.full(n, level_before, dtype=float)
        out[cut:] = level_after
        return out

    return _sched


class SyntheticUniverseProvider(UniverseProvider):
    """Generates a small deterministic multi-sector universe.

    Daily log returns for constituent j in sector s are built as:

        r_{j,t} = drift + idio_noise_{j,t} + amp_j * sin(2*pi*t/period + offset_{j,t})

    where `amp_j` is a per-stock amplitude drawn once (heterogeneous by
    design -- this is what makes the "amplitude-free" claim testable: R_s
    is invariant to `amp_j` scaling as long as phases line up) and
    `offset_{j,t}` is a per-stock phase offset whose *dispersion* is driven
    by the sector's coherence schedule: coherence=1 collapses every
    stock's offset to 0 (perfectly phase-locked), coherence=0 gives each
    stock an independent random-walk offset (phase-incoherent).

    Sector ETF prices are the equal-weighted constituent index (in log
    terms) plus a small idiosyncratic tracking-noise term, not a literal
    replica of the constituent average.
    """

    def __init__(
        self,
        start: _dt.date,
        end: _dt.date,
        n_per_sector: int = 30,
        cycle_period_days: int = 10,
        seed: int = 1234,
        coherence_schedules: dict[str, CoherenceSchedule] | None = None,
        sectors: Sequence[str] | None = None,
    ):
        self.start = start
        self.end = end
        self.n_per_sector = n_per_sector
        self.cycle_period_days = cycle_period_days
        self.seed = seed
        self.sectors = tuple(sectors) if sectors else GICS_SECTORS
        self.coherence_schedules = coherence_schedules or {}

        self._dates = self.trading_calendar(start, end)
        self._rng = np.random.default_rng(seed)
        self._membership, self._prices, self._etf_prices = self._generate()

    # -- UniverseProvider interface -----------------------------------

    def trading_calendar(self, start: _dt.date, end: _dt.date) -> pd.DatetimeIndex:
        return pd.bdate_range(start, end)

    def constituents(self, sector: str, as_of: _dt.date) -> list[str]:
        return self._membership.constituents_as_of(sector, as_of)

    def prices(self, tickers: Sequence[str], start: _dt.date, end: _dt.date) -> pd.DataFrame:
        cols = [t for t in tickers if t in self._prices.columns]
        return self._prices.loc[str(start):str(end), cols]

    def sector_etf_prices(self, start: _dt.date, end: _dt.date, etfs: Sequence[str] | None = None) -> pd.DataFrame:
        cols = list(etfs) if etfs is not None else list(self._etf_prices.columns)
        cols = [c for c in cols if c in self._etf_prices.columns]
        return self._etf_prices.loc[str(start):str(end), cols]

    # -- generation ------------------------------------------------------

    def _generate(self):
        dates = self._dates
        n = len(dates)
        t = np.arange(n)
        omega = 2 * np.pi / self.cycle_period_days

        intervals: list[MembershipInterval] = []
        price_cols: dict[str, np.ndarray] = {}
        sector_log_index: dict[str, np.ndarray] = {}

        for sector in self.sectors:
            sched = self.coherence_schedules.get(sector, constant_coherence(0.3))
            coherence = np.clip(sched(dates), 0.0, 1.0)
            incoherence = 1.0 - coherence

            sector_returns = np.zeros((self.n_per_sector, n))
            for j in range(self.n_per_sector):
                ticker = f"{sector[:3].upper()}{j:03d}"
                intervals.append(
                    MembershipInterval(ticker, sector, self.start, _dt.date(9999, 1, 1))
                )

                amp = self._rng.uniform(0.3, 2.0)          # heterogeneous amplitude
                idio_vol = self._rng.uniform(0.006, 0.02)  # heterogeneous idiosyncratic vol
                drift = self._rng.normal(0.0, 0.0002)

                idio_noise = self._rng.normal(0.0, idio_vol, size=n)

                phase_walk = np.cumsum(self._rng.normal(0.0, 0.15, size=n))
                offset = incoherence * phase_walk

                cyclical = amp * 0.01 * np.sin(omega * t + offset)

                r = drift + idio_noise + cyclical
                sector_returns[j] = r

                log_price = np.cumsum(r) + np.log(100.0)
                price_cols[ticker] = np.exp(log_price)

            sector_log_index[sector] = sector_returns.mean(axis=0)

        prices = pd.DataFrame(price_cols, index=dates)

        etf_cols: dict[str, np.ndarray] = {}
        from fasflocken.config import SECTOR_TO_ETF

        for sector in self.sectors:
            etf = SECTOR_TO_ETF[sector]
            tracking_noise = self._rng.normal(0.0, 0.001, size=n)
            r_etf = sector_log_index[sector] + tracking_noise
            etf_cols[etf] = np.exp(np.cumsum(r_etf) + np.log(50.0))

        etf_prices = pd.DataFrame(etf_cols, index=dates)
        membership = PointInTimeMembership(intervals)
        return membership, prices, etf_prices
