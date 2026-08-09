"""Data fetcher for Oglegrinden.

Data source note (declared deviation from the strategy brief):
The brief specifies Tiingo / Norgate / EODHD as data sources. All three
require a paid API key that is not available in this environment. We
substitute Yahoo Finance's public chart endpoint (no key required), which
provides split/dividend-adjusted daily close plus raw OHLCV back to each
instrument's actual first trade date. This is the same category of
substitution already used elsewhere in this repository (`data.py` uses
stooq via pandas-datareader for the same reason). Results should be
treated as indicative; a production deployment should re-run against a
licensed vendor feed to confirm adjustment-methodology consistency.

All fetched series are cached to CSV under `data_cache/` so repeated runs
don't re-hit the network.
"""

import os
import time
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _fetch_chart_json(ticker: str, retries: int = 5, backoff: float = 2.0) -> dict:
    # NOTE: requesting range="max" causes Yahoo's chart endpoint to silently
    # downsample very long histories to ~monthly bars regardless of the
    # requested interval. Passing explicit period1/period2 (unix seconds)
    # avoids this and returns genuine daily bars back to inception.
    params = {
        "period1": 0,
        "period2": int(time.time()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                _CHART_URL.format(ticker=ticker),
                params=params,
                headers=_HEADERS,
                timeout=20,
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch {ticker} after {retries} attempts: {last_err}")


def _parse_chart_json(payload: dict, ticker: str) -> pd.DataFrame:
    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise ValueError(f"No data for {ticker}: {error}")
    r = result[0]
    timestamps = r.get("timestamp")
    if not timestamps:
        raise ValueError(f"No timestamps for {ticker}")
    quote = r["indicators"]["quote"][0]
    adjclose = r["indicators"].get("adjclose", [{}])[0].get("adjclose")

    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "adjclose": adjclose if adjclose is not None else quote.get("close"),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None),
    )
    df.index.name = "date"
    # Yahoo intraday timestamps can produce duplicate calendar dates near
    # DST transitions; keep the last observation for each date.
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close", "adjclose"], how="all")
    df.attrs["first_trade_date"] = r.get("meta", {}).get("firstTradeDate")
    return df


def get_history(ticker: str, cache_dir: str = CACHE_DIR, refresh: bool = False, polite_delay: float = 0.3) -> pd.DataFrame:
    """Fetch (or load cached) full daily history for `ticker`."""
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{ticker}.csv")

    if not refresh and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        return df

    payload = _fetch_chart_json(ticker)
    df = _parse_chart_json(payload, ticker)
    df.to_csv(csv_path)
    time.sleep(polite_delay)
    return df


def load_universe(tickers, cache_dir: str = CACHE_DIR, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch/load full daily history for every ticker in `tickers`."""
    histories = {}
    for t in tickers:
        try:
            histories[t] = get_history(t, cache_dir=cache_dir, refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[data] WARNING: could not fetch {t}: {exc}")
    return histories


class Panel:
    """Wide-format panel of adjusted close, raw close, volume, dollar volume,
    daily log returns, and per-ticker inception (first available trading)
    date, built from a dict of per-ticker OHLCV DataFrames.
    """

    def __init__(self, histories: Dict[str, pd.DataFrame]):
        self.tickers = sorted(histories.keys())
        self.adjclose = pd.DataFrame({t: histories[t]["adjclose"] for t in self.tickers}).sort_index()
        self.close = pd.DataFrame({t: histories[t]["close"] for t in self.tickers}).sort_index()
        self.open = pd.DataFrame({t: histories[t]["open"] for t in self.tickers}).sort_index()
        self.volume = pd.DataFrame({t: histories[t]["volume"] for t in self.tickers}).sort_index()
        self.dollar_volume = self.close * self.volume
        # Daily log returns; a ticker with no price on a given date has NaN
        # return there rather than a fabricated 0, so downstream code must
        # explicitly handle missing history (point-in-time universe).
        self.log_returns = np.log(self.adjclose / self.adjclose.shift(1))
        self.inception = {t: self.adjclose[t].first_valid_index() for t in self.tickers}

    def eligible_on(self, date, min_adv: float, adv_lookback: int = 63) -> list:
        """Tickers with >= `adv_lookback` trading days of history and a
        trailing average dollar volume above `min_adv`, evaluated using only
        data available up to and including `date` (no look-ahead).
        """
        eligible = []
        for t in self.tickers:
            series = self.close[t].loc[:date]
            if series.dropna().shape[0] < adv_lookback:
                continue
            dv = self.dollar_volume[t].loc[:date].tail(adv_lookback)
            if dv.isna().any():
                continue
            if dv.mean() >= min_adv:
                eligible.append(t)
        return eligible
