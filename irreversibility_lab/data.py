"""Fetch and cache daily adjusted-close data for the ETF universe.

Uses the Yahoo Finance chart API directly via `requests` rather than the
`yfinance` package: `yfinance`'s default transport (curl_cffi, used to
impersonate a browser TLS fingerprint) does not respect this environment's
HTTPS_PROXY and fails with connection resets. Plain `requests` (which does
respect HTTPS_PROXY) reaches the same endpoint successfully.
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import config

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _date_to_epoch(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_symbol(symbol, start=config.START_DATE, end=None, retries=4):
    """Fetch daily adjusted close for one symbol from Yahoo's chart API."""
    period1 = _date_to_epoch(start)
    period2 = int(time.time()) if end is None else _date_to_epoch(end) + 86400
    params = {"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"}
    url = _CHART_URL.format(symbol=symbol)

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, params=params, timeout=30)
            if r.status_code == 200:
                break
            last_err = RuntimeError(f"HTTP {r.status_code} for {symbol}")
        except requests.RequestException as e:
            last_err = e
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Failed to fetch {symbol} after {retries} tries: {last_err}")

    payload = r.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        err = payload.get("chart", {}).get("error")
        raise RuntimeError(f"No data returned for {symbol}: {err}")
    res = result[0]
    ts = res["timestamp"]
    adjclose = res["indicators"]["adjclose"][0]["adjclose"]
    close = res["indicators"]["quote"][0]["close"]

    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
    px = pd.Series(adjclose, index=idx, dtype="float64")
    px = px.where(px.notna(), pd.Series(close, index=idx))
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px.name = symbol
    return px.dropna()


def _cache_path(symbol):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return os.path.join(config.DATA_DIR, f"{symbol}.csv")


def load_symbol(symbol, start=config.START_DATE, end=None, refresh=False):
    path = _cache_path(symbol)
    if not refresh and os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True)[symbol]
        # Refresh if the cache is stale (doesn't reach close to "today").
        if end is None and (pd.Timestamp.utcnow().tz_localize(None) - cached.index.max()).days <= 5:
            return cached
        if end is not None and cached.index.max() >= pd.Timestamp(end):
            return cached.loc[start:end]
    px = fetch_symbol(symbol, start=start, end=end)
    px.to_frame().to_csv(path)
    return px


def load_universe(symbols=config.UNIVERSE, start=config.START_DATE, end=None, refresh=False):
    series = {}
    for sym in symbols:
        series[sym] = load_symbol(sym, start=start, end=end, refresh=refresh)
    px = pd.DataFrame(series)
    # Outer-join calendars. Several ETFs in the universe post-date 2000
    # (DBC inception 2006-02, UUP/HYG 2007-04), so we do NOT drop to the
    # intersection of all tickers' history -- that would erase 2000-2007
    # entirely. Instead each column simply starts NaN until its own
    # inception; downstream code treats an instrument as "not yet tradeable"
    # until it has a full warm-up (W + Z_HISTORY days) of its own history.
    # Only forward-fill short fund-specific holiday gaps, on days where the
    # majority of the universe already has quotes (i.e. after inception).
    px = px.sort_index()
    has_any = px.notna().sum(axis=1) >= max(2, len(symbols) // 2)
    px.loc[has_any] = px.loc[has_any].ffill(limit=3)
    return px


def log_returns(px):
    return np.log(px / px.shift(1)).dropna(how="all")


if __name__ == "__main__":
    px = load_universe(refresh=False)
    print(px.shape)
    print(px.index.min(), px.index.max())
    print(px.tail())
