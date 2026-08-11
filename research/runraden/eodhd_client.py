"""Thin EODHD EOD data client with on-disk parquet caching.

Only fetches adjusted daily OHLCV. No point-in-time survivorship handling is
attempted beyond what EODHD serves for the ticker list supplied by the
caller -- see README "Data & universe caveats".
"""
from __future__ import annotations

import os
import time
from typing import Iterable

import pandas as pd
import requests

from config import CACHE_DIR, EODHD_API_KEY_ENV

EODHD_BASE = "https://eodhd.com/api/eod"


class EODHDError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get(EODHD_API_KEY_ENV)
    if not key:
        raise EODHDError(f"{EODHD_API_KEY_ENV} not set in environment")
    return key


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}.parquet")


def fetch_eod(ticker: str, start: str = "1990-01-01", end: str | None = None,
              force_refresh: bool = False, max_retries: int = 4) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV for `ticker`, cached to disk as parquet.

    Returns a DataFrame indexed by date with columns
    [open, high, low, close, adjusted_close, volume].
    """
    cache_file = _cache_path(ticker)
    if not force_refresh and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    key = _api_key()
    params = {
        "api_token": key,
        "fmt": "json",
        "period": "d",
        "from": start,
    }
    if end:
        params["to"] = end

    url = f"{EODHD_BASE}/{ticker}"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                payload = resp.json()
                if not isinstance(payload, list) or len(payload) == 0:
                    raise EODHDError(f"Empty/invalid payload for {ticker}: {str(payload)[:200]}")
                df = pd.DataFrame(payload)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
                df = df[[c for c in cols if c in df.columns]]
                df.to_parquet(cache_file)
                return df
            last_err = EODHDError(f"HTTP {resp.status_code} for {ticker}: {resp.text[:200]}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2 ** attempt)
    raise EODHDError(f"Failed to fetch {ticker} after {max_retries} attempts: {last_err}")


def fetch_panel(tickers: Iterable[str], start: str = "1990-01-01", end: str | None = None,
                 force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    panel: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for ticker in tickers:
        try:
            panel[ticker] = fetch_eod(ticker, start=start, end=end, force_refresh=force_refresh)
        except EODHDError as exc:
            failures[ticker] = str(exc)
    if failures:
        print(f"[eodhd_client] WARNING: {len(failures)} tickers failed to fetch:")
        for t, msg in failures.items():
            print(f"    {t}: {msg}")
    return panel
