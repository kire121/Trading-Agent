"""
Data acquisition for the Formdriften (Wasserstein form-drift) strategy.

No EODHD access or pre-existing PIT panel is available in this repo, so this
module builds one from scratch against the Yahoo Finance chart API. yfinance's
own HTTP client (curl_cffi, browser TLS impersonation) does not work through
this environment's egress proxy, so we talk to the same endpoint directly with
`requests` (which does respect HTTPS_PROXY) and a browser User-Agent.

Adjusted close = dividend/split adjusted ("adjclose" field), which is what the
strategy needs for return computation. Requesting explicit period1/period2
unix timestamps is required: `range=max` silently degrades to monthly
candles on this endpoint even when interval=1d is requested.
"""
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# ~40 most liquid US-listed sector + single-country ETFs (candidate superset;
# the true ADV>$20M universe is determined empirically in build_pit_panel).
SECTOR_ETFS = [
    "XLF", "XLK", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC",
]
COUNTRY_ETFS = [
    "EWJ", "EWG", "EWU", "EWC", "EWA", "EWY", "EWT", "EWZ", "EWW", "EWH", "EWS",
    "EWQ", "EWL", "EWN", "EWI", "EWP", "EZA", "INDA", "FXI", "MCHI", "TUR",
    "EIDO", "EPHE", "THD", "EWM", "ARGT", "VNM", "KSA", "GREK",
]
ETF_UNIVERSE = SECTOR_ETFS + COUNTRY_ETFS

# G10 FX pairs vs USD (secondary, untouched replication surface). Yahoo quotes
# some as USD/XXX (JPY=X, CHF=X, CAD=X, SEK=X, NOK=X) and some as XXX/USD
# (EURUSD=X, GBPUSD=X, AUDUSD=X, NZDUSD=X) -- the sign of the quoting
# convention doesn't matter for a cross-sectionally z-scored, dollar-neutral
# signal, but is tracked here for anyone reading returns directionally.
FX_UNIVERSE = [
    "EURUSD=X", "JPY=X", "GBPUSD=X", "CHF=X", "CAD=X",
    "AUDUSD=X", "NZDUSD=X", "SEK=X", "NOK=X",
]

FETCH_START = dt.datetime(1999, 1, 1)


def _fetch_one(session, ticker, start=FETCH_START, end=None, retries=4):
    end = end or dt.datetime.utcnow()
    p1, p2 = int(start.timestamp()), int(end.timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"}
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            result = data.get("chart", {}).get("result")
            if not result:
                last_err = data.get("chart", {}).get("error")
                time.sleep(1 + attempt)
                continue
            res = result[0]
            ts = res["timestamp"]
            quote = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
            df = pd.DataFrame(
                {
                    "Open": quote["open"],
                    "High": quote["high"],
                    "Low": quote["low"],
                    "Close": adj,
                    "RawClose": quote["close"],
                    "Volume": quote["volume"],
                },
                index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None),
            )
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df = df.dropna(subset=["Close"])
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to fetch {ticker}: {last_err}")


def fetch_universe(tickers, refresh=False, sleep=0.15):
    """Fetch (or load cached) daily OHLCV+adjclose for each ticker."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    out = {}
    for t in tickers:
        cache_path = os.path.join(CACHE_DIR, f"{t.replace('=', '_')}.csv")
        if not refresh and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            out[t] = df
            continue
        df = _fetch_one(session, t)
        df.to_csv(cache_path)
        out[t] = df
        time.sleep(sleep)
    return out


def build_price_panel(data_dict):
    """Wide adjusted-close panel aligned on the union of trading dates."""
    closes = {t: df["Close"] for t, df in data_dict.items()}
    panel = pd.DataFrame(closes).sort_index()
    # All are US-listed instruments (ETFs trade NYSEArca hours; FX quoted
    # against the same US trading calendar by Yahoo), so a straight union
    # join is appropriate. Forward-fill isolated single-day gaps (e.g. a
    # local holiday feed hiccup) but do not paper over real absence of data.
    panel = panel.ffill(limit=2)
    return panel


def build_volume_panel(data_dict):
    vols = {t: df["Volume"] for t, df in data_dict.items() if "Volume" in df.columns}
    return pd.DataFrame(vols).sort_index()


def dollar_adv(price_panel, volume_panel, window=63):
    dollar_vol = price_panel * volume_panel
    return dollar_vol.rolling(window, min_periods=window // 2).mean()


if __name__ == "__main__":
    all_tickers = ETF_UNIVERSE + FX_UNIVERSE
    data = fetch_universe(all_tickers)
    panel = build_price_panel(data)
    print(panel.shape, panel.index.min(), panel.index.max())
    print(panel.tail())
