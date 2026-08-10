"""
Efterskalvsklockan -- data fetcher.

Pulls daily OHLCV (raw + adjusted close) from EODHD for:
  - the 40-ticker IS/design panel (config.IS_UNIVERSE) -- spent freely.
  - the locked OOS panel (config.OOS_UNIVERSE = 16 lands ETFs + 12 single-
    commodity ETF/ETC), read exactly once by run_oos.py.

Caches raw JSON responses to research/omori/data/raw/ and writes a combined,
NaN-preserving (no forward-fill) OHLCV panel per universe to
research/omori/data/{primary,secondary}_<field>.csv -- same layout as
research/dammluckan/fetch_data.py and research/vindkastet/fetch_data.py.
"""
import os
import json
import time
import urllib.request
import urllib.error
import ssl
import certifi

from research.omori import config

API_KEY = os.environ["EODHD_API_KEY"]
BASE = "https://eodhd.com/api/eod/{sym}.US"
FROM = config.HISTORY_FROM
TO = config.HISTORY_TO

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

FIELDS = ["open", "high", "low", "close", "adjusted_close", "volume"]

CTX = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE", certifi.where()))


def _fetch_one(sym, retries=4):
    cache_path = os.path.join(RAW_DIR, f"{sym}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    url = BASE.format(sym=sym) + f"?api_token={API_KEY}&period=d&fmt=json&from={FROM}&to={TO}"
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
    handlers.append(urllib.request.HTTPSHandler(context=CTX))
    opener = urllib.request.build_opener(*handlers)
    last_err = None
    for attempt in range(retries):
        try:
            with opener.open(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {sym}: {last_err}")


def fetch_all(symbols):
    out = {}
    for sym in symbols:
        data = _fetch_one(sym)
        if isinstance(data, dict) and "code" in data and "message" in data and len(data) <= 3:
            print(f"WARNING: {sym} returned error payload: {data}")
            continue
        if not data:
            print(f"WARNING: {sym} returned an empty series")
            continue
        out[sym] = data
        print(f"{sym}: {len(data)} rows, {data[0]['date']} -> {data[-1]['date']}")
    return out


def _write_panel(label, raw):
    import pandas as pd

    df = None
    for field in FIELDS:
        panel = {}
        for sym, rows in raw.items():
            s = pd.Series({r["date"]: r.get(field) for r in rows if r.get(field) is not None})
            s.index = pd.to_datetime(s.index)
            panel[sym] = s
        df = pd.DataFrame(panel).sort_index()
        out_path = os.path.join(HERE, "data", f"{label}_{field}.csv")
        df.to_csv(out_path)
    print(f"Saved {label}: shape={df.shape}, date range {df.index.min()} -> {df.index.max()}")
    print("Live-name count (last row):", df.iloc[-1].notna().sum())


if __name__ == "__main__":
    for label, syms in [("primary", config.IS_UNIVERSE), ("secondary", config.OOS_UNIVERSE)]:
        print(f"\n=== Fetching {label} universe ({len(syms)} tickers) ===")
        raw = fetch_all(syms)
        _write_panel(label, raw)
