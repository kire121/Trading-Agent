"""Fetch the IS_UNIVERSE panel from EODHD and cache combined per-field CSVs.

Mirrors the sibling convention (omori/dammluckan): one wide CSV per OHLCV
field under data/, columns = tickers, index = date. Run once; downstream
code reads the CSVs, not EODHD directly.
"""
import os

import pandas as pd

from . import config
from . import eodhd_client


def build_panel_csvs(tickers=None, start=None, end=None, force_refresh=False):
    tickers = tickers or config.IS_UNIVERSE
    start = start or config.HISTORY_START
    raw = eodhd_client.fetch_panel(tickers, start=start, end=end, force_refresh=force_refresh)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    for field in config.FIELDS:
        cols = {}
        for ticker, df in raw.items():
            if field in df.columns:
                cols[ticker] = df[field]
        if not cols:
            continue
        wide = pd.DataFrame(cols).sort_index()
        wide.to_csv(os.path.join(config.DATA_DIR, f"primary_{field}.csv"))

    fetched = sorted(raw.keys())
    missing = sorted(set(tickers) - set(fetched))
    print(f"[fetch_data] fetched {len(fetched)}/{len(tickers)} tickers"
          + (f", missing: {missing}" if missing else ""))
    return raw


if __name__ == "__main__":
    build_panel_csvs()
