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
  * `NorgateProvider`, `SharadarProvider`
                              -- thin real-vendor stubs. Neither vendor is
                                 reachable from this environment (no NDU
                                 license, no Nasdaq Data Link key), so each
                                 raises `DataProviderNotConfigured` with the
                                 exact package + call a user needs to
                                 finish wiring, rather than silently
                                 returning fabricated data.
  * `EODHDProvider`          -- a real, working implementation (not a
                                 stub): point-in-time S&P 500 membership
                                 from `fundamentals/{index}.INDX`'s
                                 `HistoricalTickerComponents`, per-ticker
                                 GICS sector, and adjusted EOD prices, all
                                 via plain REST calls against api_token
                                 read from EODHD_API_KEY / EODHD_API_TOKEN
                                 (or passed explicitly). Raises
                                 `DataProviderNotConfigured` if no key is
                                 available. See its class docstring for
                                 the (disclosed, non-silent) sector-coverage
                                 gap on some old delisted names.
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
import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from fasflocken.config import GICS_SECTORS, SECTOR_TO_ETF


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
    """EODHD, via plain REST calls, using a real EODHD_API_KEY when one is
    configured (checked at construction time).

    Contrary to this module's earlier assumption (and this package's
    original README), EODHD's `fundamentals/{index}.INDX` endpoint DOES
    expose point-in-time S&P 500 membership: its `HistoricalTickerComponents`
    field is a StartDate/EndDate interval per ticker that's ever been a
    constituent (not just the current 503), which is exactly what
    universe.PointInTimeMembership needs. Verified live against the API
    (e.g. Lehman Brothers shows EndDate=2008-09-16, IsDelisted=1).

    GICS sector comes from each ticker's own `fundamentals/{ticker}.US`
    `General.GicSector` field (the literal GICS taxonomy -- confirmed to
    match this package's sector names exactly except "Information
    Technology" -> "Technology", handled by `_GICSECTOR_TO_OURS` below).
    This is a per-ticker API call, so it's cached to disk; it also isn't
    reliable for older delisted names (some very old bankruptcies/M&A
    return GicSector="NA" -- EODHD's fundamentals record for them is too
    sparse), which are then excluded from that sector's constituent pool.
    That's a real, disclosed gap, not a silent one: see `sector_coverage()`.

    Prices are EOD *adjusted* close (splits/dividends already applied) via
    `eod/{ticker}.US`, which EODHD retains for delisted tickers too (no
    survivorship bias in the price series itself, unlike the sector
    coverage caveat above).

    All network responses are cached to `cache_dir` (default: a directory
    under the system temp dir, i.e. *outside* any git working tree --
    EODHD's data is licensed, not something this package's tests or repo
    should ever bundle or commit).
    """

    BASE_URL = "https://eodhd.com/api"

    # EODHD's GicSector uses the literal GICS sector name, which matches
    # config.GICS_SECTORS verbatim except for Information Technology.
    _GICSECTOR_TO_OURS: dict[str, str] = {"Information Technology": "Technology"}

    def __init__(
        self,
        api_token: str | None = None,
        index_ticker: str = "GSPC.INDX",
        cache_dir: str | None = None,
        session=None,
        request_delay: float = 0.05,
        timeout: float = 30.0,
        max_workers: int = 6,
    ):
        self.api_token = api_token or os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_API_TOKEN")
        if not self.api_token:
            raise DataProviderNotConfigured(
                "EODHD requires an API key: pass api_token= explicitly, or set the "
                "EODHD_API_KEY (or EODHD_API_TOKEN) environment variable."
            )

        self.index_ticker = index_ticker
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "fasflocken_eodhd_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.request_delay = request_delay
        self.timeout = timeout
        # Kept conservative by default: a local dev/CI network path (e.g.
        # routed through a policy-enforcing egress proxy) may not sustain
        # high fan-out as reliably as a direct connection to EODHD would.
        # Bump this once you've confirmed your own network path handles it.
        self.max_workers = max_workers

        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise DataProviderNotConfigured(
                    "EODHDProvider requires the `requests` package (pip install requests)."
                ) from exc
            session = requests.Session()
            pool_size = max(10, max_workers * 2)
            adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        self.session = session

        self._membership: PointInTimeMembership | None = None
        self._sector_lookup_failures: list[str] = []
        self._price_cache: dict[str, pd.DataFrame] = {}

    # -- low-level HTTP -----------------------------------------------

    def _get(self, path: str, max_retries: int = 6, **params) -> dict:
        """A transient failure here (timeout, connection reset, 5xx, 429)
        must not be silently indistinguishable from "this ticker genuinely
        has no data" -- callers like _lookup_gics_sector treat any
        exception as the latter and permanently drop the ticker from the
        membership pool, so a flaky proxy hop or rate-limit blip under
        concurrent load could otherwise quietly corrupt the constituent
        set. Retried with backoff (honoring a Retry-After header on 429,
        confirmed live: EODHD returns `x-ratelimit-limit` on every
        response, i.e. a real short-window throttle on top of the daily
        cap). A non-429 4xx (bad ticker, bad auth) fails fast instead,
        since retrying that would just waste calls on the same bad request.
        """
        import requests

        params = {**params, "api_token": self.api_token, "fmt": "json"}
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.session.get(f"{self.BASE_URL}/{path}", params=params, timeout=self.timeout)
                if self.request_delay:
                    time.sleep(self.request_delay)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else 1.0 * (2**attempt)
                    last_exc = exc
                elif 400 <= resp.status_code < 500:
                    raise  # bad ticker / auth problem -- retrying won't help
                else:
                    last_exc = exc
                    wait = 0.25 * (2**attempt)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                wait = 0.25 * (2**attempt)
            if attempt < max_retries - 1:
                time.sleep(min(wait, 20.0))
        raise last_exc

    def _cache_path(self, kind: str, key: str) -> str:
        safe_key = key.replace("/", "_")
        return os.path.join(self.cache_dir, f"{kind}_{safe_key}.json")

    def _get_cached(self, kind: str, key: str, path: str, **params) -> dict:
        cache_file = self._cache_path(kind, key)
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)
        data = self._get(path, **params)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data

    # -- point-in-time membership + GICS sector ------------------------

    def _load_membership(self, max_workers: int | None = None) -> None:
        if self._membership is not None:
            return
        max_workers = max_workers or self.max_workers
        data = self._get_cached("index", self.index_ticker, f"fundamentals/{self.index_ticker}")
        hist = data.get("HistoricalTickerComponents", {}) or {}

        entries = [e for e in hist.values() if e.get("Code")]
        codes = [e["Code"] for e in entries]

        # Sector lookup is one HTTP round-trip per ticker (~800 of them for
        # the full S&P 500 history) and purely I/O-bound, so it's fanned out
        # across a thread pool rather than done serially -- otherwise a cold
        # cache means minutes of sequential proxy round-trips before any
        # backtest can even start.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            sectors = list(pool.map(self._lookup_gics_sector, codes))

        intervals: list[MembershipInterval] = []
        for entry, sector in zip(entries, sectors):
            code = entry["Code"]
            start = (
                _dt.datetime.strptime(entry["StartDate"], "%Y-%m-%d").date()
                if entry.get("StartDate")
                else _dt.date(1970, 1, 1)
            )
            end = (
                _dt.datetime.strptime(entry["EndDate"], "%Y-%m-%d").date()
                if entry.get("EndDate")
                else _dt.date(9999, 1, 1)
            )
            if sector is None:
                self._sector_lookup_failures.append(code)
                continue
            intervals.append(MembershipInterval(code, sector, start, end))

        self._membership = PointInTimeMembership(intervals)

    def _lookup_gics_sector(self, ticker: str) -> str | None:
        # A single (comma-free) `filter=` returns the bare value directly
        # (e.g. the JSON string "Information Technology"), not a
        # {"General::GicSector": ...} envelope -- only multi-field filters
        # get wrapped that way.
        try:
            raw = self._get_cached(
                "fundamentals", ticker, f"fundamentals/{ticker}.US", filter="General::GicSector"
            )
        except Exception:
            return None
        if not isinstance(raw, str) or not raw or raw == "NA":
            return None
        return self._GICSECTOR_TO_OURS.get(raw, raw)

    def sector_coverage(self) -> dict:
        """Diagnostics: how many historical S&P 500 tickers got a usable
        GICS sector vs. how many were dropped (see class docstring).
        """
        self._load_membership()
        covered = len(self._membership.all_tickers()) if self._membership else 0
        return {
            "covered": covered,
            "dropped": len(self._sector_lookup_failures),
            "dropped_tickers": list(self._sector_lookup_failures),
        }

    # -- prices ----------------------------------------------------------

    def _fetch_eod(self, ticker: str) -> pd.DataFrame:
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        try:
            rows = self._get_cached("eod", ticker, f"eod/{ticker}.US", period="d", order="a")
        except Exception:
            # Unresolvable ticker (delisted beyond EODHD's coverage, a
            # symbol collision, transient error, ...): treated as "no price
            # data", not a fatal error for the whole prices() call -- the
            # ticker just won't be usable for anything downstream.
            rows = None
        if not rows or not isinstance(rows, list):
            df = pd.DataFrame(columns=["close"])
            df.index = pd.DatetimeIndex([])
        else:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            # Raw EOD rows carry both "close" (unadjusted) and
            # "adjusted_close"; select adjusted_close FIRST, then rename --
            # renaming in place would leave two columns both labeled
            # "close" (the pre-existing raw one plus the renamed one),
            # and single-bracket selection on a duplicate label returns a
            # DataFrame instead of a Series, breaking every caller that
            # expects prices()/sector_etf_prices() to hand back one
            # column per ticker.
            df = df[["adjusted_close"]].rename(columns={"adjusted_close": "close"})
        self._price_cache[ticker] = df
        return df

    # -- UniverseProvider interface -----------------------------------

    def constituents(self, sector: str, as_of: _dt.date) -> list[str]:
        self._load_membership()
        return self._membership.constituents_as_of(sector, as_of)

    def prices(self, tickers, start: _dt.date, end: _dt.date, max_workers: int | None = None) -> pd.DataFrame:
        tickers = list(tickers)
        max_workers = max_workers or self.max_workers
        uncached = [t for t in tickers if t not in self._price_cache]
        if uncached:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(self._fetch_eod, uncached))  # populates self._price_cache as a side effect

        cols = {t: self._fetch_eod(t)["close"].loc[str(start) : str(end)] for t in tickers}
        return pd.DataFrame(cols)

    def sector_etf_prices(self, start: _dt.date, end: _dt.date, etfs=None) -> pd.DataFrame:
        etfs = list(etfs) if etfs is not None else list(SECTOR_TO_ETF.values())
        return self.prices(etfs, start, end)

    def trading_calendar(self, start: _dt.date, end: _dt.date) -> pd.DatetimeIndex:
        ref = self._fetch_eod("SPY")["close"]
        return ref.loc[str(start) : str(end)].index


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
