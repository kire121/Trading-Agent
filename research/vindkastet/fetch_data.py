"""
Vindkastet -- data fetcher.

Pulls daily adjusted closes from EODHD for:
  - the 16-ETF primary universe specified in the hypothesis writeup
  - ACWI (diversification benchmark)
  - a secondary, untouched single-country-equity ETF universe used only for
    the OOS "sign replication" cross-check (since the primary program has
    already spent a 2018-2026 OOS window on US multi-asset/sector data).

Caches raw JSON responses to research/vindkastet/data/raw/ and writes a
combined, forward-filled-free (NaN preserved) adjusted-close panel to
research/vindkastet/data/prices_primary.csv and prices_secondary.csv plus
prices_bench.csv.
"""
import os
import json
import time
import urllib.request
import urllib.error
import ssl
import certifi

API_KEY = os.environ["EODHD_API_KEY"]
BASE = "https://eodhd.com/api/eod/{sym}.US"
FROM = "2005-01-01"
TO = "2026-08-10"

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

PRIMARY = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD",
           "SLV", "DBC", "USO", "UUP", "FXE", "FXY", "VNQ"]
BENCH = ["ACWI"]
# Untouched secondary surface: single-country equity ETFs, never used in the
# primary US multi-asset universe or in prior 2018-2026 OOS work.
SECONDARY = ["EWJ", "EWG", "EWU", "EWQ", "EWI", "EWP", "EWL", "EWA", "EWC",
             "EWY", "EWT", "EWZ", "EWW", "EWS", "EWH", "EWD"]

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
        out[sym] = data
        print(f"{sym}: {len(data)} rows, {data[0]['date'] if data else 'EMPTY'} -> {data[-1]['date'] if data else 'EMPTY'}")
    return out


if __name__ == "__main__":
    import pandas as pd

    for label, syms in [("primary", PRIMARY), ("bench", BENCH), ("secondary", SECONDARY)]:
        print(f"\n=== Fetching {label} universe ===")
        raw = fetch_all(syms)
        panel = {}
        for sym, rows in raw.items():
            s = pd.Series({r["date"]: r["adjusted_close"] for r in rows if r.get("adjusted_close") is not None})
            s.index = pd.to_datetime(s.index)
            panel[sym] = s
        df = pd.DataFrame(panel).sort_index()
        out_path = os.path.join(HERE, "data", f"prices_{label}.csv")
        df.to_csv(out_path)
        print(f"Saved {out_path}: shape={df.shape}, date range {df.index.min()} -> {df.index.max()}")
        print("Live-name count over time (last row):", df.iloc[-1].notna().sum())
        print("First valid index per column:\n", df.apply(lambda c: c.first_valid_index()))
