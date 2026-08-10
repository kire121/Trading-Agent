"""Point-in-time universe membership and the data-provider seam.

The brief calls for "Punkt-i-tid S&P 500 + 400 (inkl. delistade med
delisting-avkastning) ... Data: dagliga OHLCV + historiska konstituenter
fran Norgate Data eller Sharadar/Nasdaq Data Link" and execution "via
IBKR". None of Norgate, Sharadar/Nasdaq Data Link, or an IBKR connection
are available in this environment. Following the precedent already set by
the sibling strategy branches in this repository (Oglegrinden substitutes
Yahoo Finance for Tiingo/Norgate/EODHD; Fasflocken substitutes a real,
working EODHD subscription for the same set), we substitute EODHD here too
-- an `EODHD_API_KEY` is present in this environment's process
environment, and EODHD's fundamentals endpoint for `GSPC.INDX` (S&P 500)
exposes genuine point-in-time membership (`HistoricalTickerComponents`,
with per-ticker StartDate/EndDate), which is materially closer to the
brief's own request than a free/no-key substitute would have been.

Two DECLARED gaps remain, both surfaced explicitly rather than papered
over (see README "Declared deviations"):

1. EODHD's `MID.INDX` (S&P MidCap 400) exposes only *current* constituents,
   not a HistoricalTickerComponents feed. The 400-leg of the universe is
   therefore NOT point-in-time -- it is today's mid-cap membership,
   projected backward, which introduces survivorship bias for that slice.
   This is measured and disclosed, not hidden: every `MembershipInterval`
   carries a `point_in_time: bool` flag so downstream code (and the
   README/REPORT) can report the S&P 500 and S&P 400 legs separately.
2. Delisting return is approximated as the return to the last EOD print
   available under a ticker's *historical* symbol. EODHD does keep pricing
   a name that has gone to OTC/bankruptcy under a changed ticker (e.g.
   `SIVB` -> `SIVBQ` after Silicon Valley Bank's 2023 collapse), but we do
   not chain ticker-symbol changes -- a declared simplification that will
   understate losses for names that wind down slowly under a new symbol
   rather than stopping abruptly under their original one.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from vridmomentet import config

_FAR_FUTURE = _dt.date(9999, 1, 1)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class DataProviderNotConfigured(RuntimeError):
    """Raised when a declared-but-unavailable paid data vendor is selected."""


# --------------------------------------------------------------------------
# Point-in-time membership
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MembershipInterval:
    ticker: str
    source: str              # "sp500" or "sp400"
    start: _dt.date
    end: _dt.date             # exclusive; _FAR_FUTURE if still current
    point_in_time: bool       # False for the sp400 current-constituents-only proxy

    def covers(self, as_of: _dt.date) -> bool:
        return self.start <= as_of < self.end


class PointInTimeMembership:
    """A flat list of membership spells, queried by date."""

    def __init__(self, intervals: Sequence[MembershipInterval]):
        self._intervals = list(intervals)

    def constituents_as_of(self, as_of: _dt.date) -> list[str]:
        return sorted({iv.ticker for iv in self._intervals if iv.covers(as_of)})

    def all_tickers(self) -> list[str]:
        return sorted({iv.ticker for iv in self._intervals})

    def intervals_for(self, ticker: str) -> list[MembershipInterval]:
        return [iv for iv in self._intervals if iv.ticker == ticker]

    def tickers_from_source(self, source: str) -> set[str]:
        return {iv.ticker for iv in self._intervals if iv.source == source}

    def __len__(self) -> int:
        return len(self._intervals)


# --------------------------------------------------------------------------
# Provider abstraction
# --------------------------------------------------------------------------

class UniverseProvider(ABC):
    @abstractmethod
    def membership(self) -> PointInTimeMembership: ...

    @abstractmethod
    def prices(self, tickers: Sequence[str], start: _dt.date, end: _dt.date) -> dict[str, pd.DataFrame]:
        """ticker -> DataFrame[open, high, low, close, adjusted_close, volume], date-indexed."""
        ...


# --------------------------------------------------------------------------
# Stubs for the brief's named vendors -- neither is reachable in this
# environment; both fail fast with the exact remediation needed, mirroring
# the fasflocken/universe.py precedent.
# --------------------------------------------------------------------------

class NorgateProvider(UniverseProvider):
    """Stub: raises in __init__ with the exact remediation needed. The
    membership()/prices() overrides below exist only so Python's ABC
    machinery (which checks for unimplemented @abstractmethods *before*
    __init__ runs) doesn't mask that helpful error behind a generic
    "can't instantiate abstract class" TypeError instead.
    """

    def __init__(self, *_args, **_kwargs):
        raise DataProviderNotConfigured(
            "Norgate requires the local Norgate Data Updater (NDU) running plus a "
            "licensed `norgatedata` package. Neither is available in this "
            "environment. Install `norgatedata`, run NDU, then implement "
            "membership()/prices() using norgatedata.watchlist_symbols / "
            "norgatedata.index_constituent_timeseries / norgatedata.price_timeseries."
        )

    def membership(self) -> PointInTimeMembership: ...  # pragma: no cover -- unreachable, __init__ always raises

    def prices(self, tickers: Sequence[str], start: _dt.date, end: _dt.date) -> dict[str, pd.DataFrame]: ...  # pragma: no cover


class SharadarProvider(UniverseProvider):
    """Stub: same rationale as NorgateProvider above."""

    def __init__(self, *_args, **_kwargs):
        raise DataProviderNotConfigured(
            "Sharadar (Nasdaq Data Link) requires a paid SHARADAR/SEP + SHARADAR/TICKERS "
            "subscription and an NASDAQ_DATA_LINK_API_KEY. Neither is available in this "
            "environment. Set the key, `pip install nasdaq-data-link`, then implement "
            "membership()/prices() using the SHARADAR/TICKERS (historical index flags) "
            "and SHARADAR/SEP (+ SHARADAR/SFP for delisted) tables."
        )

    def membership(self) -> PointInTimeMembership: ...  # pragma: no cover -- unreachable, __init__ always raises

    def prices(self, tickers: Sequence[str], start: _dt.date, end: _dt.date) -> dict[str, pd.DataFrame]: ...  # pragma: no cover


# --------------------------------------------------------------------------
# EODHD -- the real, working substitute in this environment.
# --------------------------------------------------------------------------

class EODHDProvider(UniverseProvider):
    BASE_URL = "https://eodhd.com/api"

    def __init__(
        self,
        api_token: str | None = None,
        sp500_index: str = config.SP500_INDEX_TICKER,
        sp400_index: str = config.SP400_INDEX_TICKER,
        cache_dir: str | None = None,
        max_workers: int = 8,
        request_timeout: float = 30.0,
    ):
        self.api_token = api_token or os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_API_TOKEN")
        if not self.api_token:
            raise DataProviderNotConfigured(
                "EODHD requires an API key: pass api_token= explicitly, or set the "
                "EODHD_API_KEY (or EODHD_API_TOKEN) environment variable."
            )
        self.sp500_index = sp500_index
        self.sp400_index = sp400_index
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_workers = max_workers
        self.request_timeout = request_timeout

    # -- HTTP plumbing --------------------------------------------------

    def _cache_path(self, kind: str, key: str) -> str:
        safe = key.replace("/", "_")
        return os.path.join(self.cache_dir, f"{kind}_{safe}.json")

    def _get(self, path: str, retries: int = 5, **params) -> dict | list:
        import requests

        params = {**params, "api_token": self.api_token, "fmt": "json"}
        url = f"{self.BASE_URL}/{path}"
        last_err: Exception | str | None = None
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=_HEADERS, params=params, timeout=self.request_timeout)
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(2.0 * (2 ** attempt))
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2.0 * (2 ** attempt)))
                time.sleep(wait)
                last_err = f"HTTP 429 for {path}"
                continue
            if 500 <= resp.status_code < 600:
                last_err = f"HTTP {resp.status_code} for {path}"
                time.sleep(2.0 * (2 ** attempt))
                continue
            # Other 4xx (bad ticker, auth, etc.) -- do not retry, wastes calls.
            raise RuntimeError(f"EODHD request failed, HTTP {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError(f"EODHD request failed after {retries} attempts: {last_err}")

    def _get_cached(self, kind: str, key: str, path: str, **params) -> dict | list:
        cache_file = self._cache_path(kind, key)
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)
        data = self._get(path, **params)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data

    # -- Membership -------------------------------------------------------

    def _sp500_membership(self) -> list[MembershipInterval]:
        data = self._get_cached("fundamentals", self.sp500_index, f"fundamentals/{self.sp500_index}")
        htc = data.get("HistoricalTickerComponents") or {}
        out = []
        for _key, row in htc.items():
            ticker = row.get("Code")
            if not ticker:
                continue
            start = _parse_date(row.get("StartDate")) or HISTORY_FLOOR
            end_raw = row.get("EndDate")
            end = _parse_date(end_raw) if end_raw else _FAR_FUTURE
            out.append(MembershipInterval(ticker=ticker, source="sp500", start=start, end=end, point_in_time=True))
        return out

    def _sp400_membership(self) -> list[MembershipInterval]:
        # DECLARED (see module docstring): EODHD's MID.INDX has no
        # HistoricalTickerComponents feed. We can only see today's 400
        # constituents; every interval is stamped point_in_time=False so
        # this leg's survivorship bias is visible to every downstream
        # consumer rather than silently blended into the S&P 500 leg.
        data = self._get_cached("fundamentals", self.sp400_index, f"fundamentals/{self.sp400_index}")
        comp = data.get("Components") or {}
        out = []
        for _key, row in comp.items():
            ticker = row.get("Code")
            if not ticker:
                continue
            out.append(
                MembershipInterval(
                    ticker=ticker, source="sp400", start=HISTORY_FLOOR, end=_FAR_FUTURE, point_in_time=False
                )
            )
        return out

    def membership(self) -> PointInTimeMembership:
        return PointInTimeMembership(self._sp500_membership() + self._sp400_membership())

    def sector_coverage_note(self) -> str:
        return (
            "S&P 500 leg: point-in-time (EODHD HistoricalTickerComponents). "
            "S&P 400 leg: current constituents only, projected backward (not point-in-time)."
        )

    # -- Prices -------------------------------------------------------------

    @staticmethod
    def _eod_symbol(ticker: str) -> str:
        # EODHD's US common-stock suffix convention.
        return f"{ticker}.US"

    def _fetch_one_price_series(self, ticker: str, start: _dt.date, end: _dt.date) -> pd.DataFrame | None:
        cache_file = self._cache_path("eod", ticker)
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                rows = json.load(f)
        else:
            try:
                rows = self._get(
                    f"eod/{self._eod_symbol(ticker)}",
                    **{"from": HISTORY_FLOOR.isoformat(), "to": _dt.date.today().isoformat(), "period": "d"},
                )
            except RuntimeError:
                return None
            if not isinstance(rows, list):
                return None
            with open(cache_file, "w") as f:
                json.dump(rows, f)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        keep = [c for c in ["open", "high", "low", "close", "adjusted_close", "volume"] if c in df.columns]
        df = df[keep]
        df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
        return df if not df.empty else None

    def prices(self, tickers: Sequence[str], start: _dt.date, end: _dt.date) -> dict[str, pd.DataFrame]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._fetch_one_price_series, t, start, end): t for t in tickers}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    df = fut.result()
                except Exception as exc:  # noqa: BLE001 -- one bad ticker must not kill the whole fetch
                    print(f"[universe] WARNING: could not fetch {ticker}: {exc}")
                    continue
                if df is not None:
                    out[ticker] = df
        return out


HISTORY_FLOOR = _dt.date(2000, 1, 1)


def _parse_date(s: str | None) -> _dt.date | None:
    if not s:
        return None
    try:
        return _dt.datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
