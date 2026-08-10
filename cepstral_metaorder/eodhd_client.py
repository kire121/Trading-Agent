"""
Thin REST client for EODHD (https://eodhd.com/financial-apis/).

Used for two distinct purposes in this module:
  1. Daily EOD OHLCV -> PIT universe construction (price/ADV filters), daily
     returns for the validation battery, and diversification proxies (SPY,
     TSMOM basket).
  2. Intraday 1-minute OHLCV -> the cepstral slicing signal itself.

EODHD caps 1-minute intraday requests at 120 calendar days per call (measured
empirically: 121 days returns HTTP 422 "Max period length is 120 days"). This
client chunks automatically. There is no such cap on the EOD daily endpoint.

No secrets are hardcoded: the API key is read from the EODHD_API_KEY
environment variable, which is already provisioned in this workspace.
"""

from __future__ import annotations

import os
import time
import datetime as dt
from typing import List, Optional

import pandas as pd
import requests

BASE_URL = "https://eodhd.com/api"
INTRADAY_MAX_DAYS = 120
_SESSION = requests.Session()


class EODHDError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("EODHD_API_KEY")
    if not key:
        raise EODHDError(
            "EODHD_API_KEY is not set in the environment. This client will not "
            "fabricate data -- set the key or use synthetic.py for a dry run."
        )
    return key


def _get(path: str, params: dict, retries: int = 4, timeout: int = 30) -> requests.Response:
    params = dict(params)
    params["api_token"] = _api_key()
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(f"{BASE_URL}/{path}", params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            time.sleep(2 ** attempt * 2)
            continue
        if resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        # 4xx other than 429 (e.g. 422 bad range, 404 unknown symbol) -> don't retry
        raise EODHDError(f"EODHD {path} failed: HTTP {resp.status_code}: {resp.text[:300]}")
    raise EODHDError(f"EODHD {path} failed after {retries} retries: {last_exc}")


def get_us_common_stock_symbols() -> pd.DataFrame:
    """Full US exchange symbol list, so the PIT universe builder can be pointed
    at the real market rather than a curated candidate list when budget allows."""
    resp = _get("exchange-symbol-list/US", {"fmt": "json"})
    df = pd.DataFrame(resp.json())
    return df[df["Type"] == "Common Stock"].reset_index(drop=True)


def get_eod(symbol: str, frm: Optional[str] = None, to: Optional[str] = None) -> pd.DataFrame:
    """Daily OHLCV, adjusted and unadjusted close. frm/to are 'YYYY-MM-DD' or None."""
    params = {"fmt": "json", "period": "d", "order": "a"}
    if frm:
        params["from"] = frm
    if to:
        params["to"] = to
    resp = _get(f"eod/{symbol}.US", params)
    data = resp.json()
    if not data:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adjusted_close", "volume"])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _chunk_ranges(frm: dt.date, to: dt.date, max_days: int) -> List[tuple]:
    chunks = []
    cur = frm
    one_day = dt.timedelta(days=1)
    while cur <= to:
        end = min(cur + dt.timedelta(days=max_days - 1), to)
        chunks.append((cur, end))
        cur = end + one_day
    return chunks


def get_intraday_1m(symbol: str, frm: dt.date, to: dt.date, pause_s: float = 0.15) -> pd.DataFrame:
    """1-minute OHLCV across an arbitrary date range, auto-chunked at the
    120-day API limit. Returns UTC timestamps (EODHD's 'datetime' field is
    already UTC / gmtoffset=0); RTH filtering happens downstream in
    signal.py where the exchange calendar/timezone is applied."""
    frames = []
    for start, end in _chunk_ranges(frm, to, INTRADAY_MAX_DAYS):
        params = {
            "fmt": "json",
            "interval": "1m",
            "from": int(dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc).timestamp()),
            "to": int(dt.datetime.combine(end + dt.timedelta(days=1), dt.time.min, tzinfo=dt.timezone.utc).timestamp()),
        }
        resp = _get(f"intraday/{symbol}.US", params)
        data = resp.json()
        if data:
            frames.append(pd.DataFrame(data))
        time.sleep(pause_s)  # be a polite API citizen; budget is generous but not infinite
    if not frames:
        return pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return df


def get_market_cap(symbol: str) -> Optional[float]:
    """Current market cap from fundamentals. Used as a static size control in
    the Step-1 redundancy screen -- NOT point-in-time historical, which is a
    documented simplification for the pilot (see README)."""
    try:
        resp = _get(f"fundamentals/{symbol}.US", {"fmt": "json", "filter": "Highlights::MarketCapitalization"})
    except EODHDError:
        return None
    try:
        val = resp.json()
        return float(val) if val not in (None, "", "NA") else None
    except (ValueError, TypeError):
        return None
