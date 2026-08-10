"""
Local parquet cache in front of eodhd_client, so iterating on the pipeline
doesn't re-spend API budget or wall-clock re-downloading the same bars.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

from . import eodhd_client as client

CACHE_DIR = Path(__file__).parent / "data_cache"


def _eod_path(symbol: str) -> Path:
    return CACHE_DIR / "eod" / f"{symbol}.parquet"


def _intraday_path(symbol: str, frm: dt.date, to: dt.date) -> Path:
    return CACHE_DIR / "intraday_1m" / f"{symbol}_{frm.isoformat()}_{to.isoformat()}.parquet"


def cached_eod(symbol: str, frm: Optional[str] = None, to: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """Always caches the FULL available history for `symbol` (one file per
    symbol, no range in the cache key -- unlike intraday there's no per-
    request range cap on the EOD endpoint, so there's no reason to key on
    range and every reason not to: a range-keyed cache silently returns a
    stale, differently-scoped result for any later call with a different
    frm/to on the same symbol). frm/to filter the cached full history."""
    path = _eod_path(symbol)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        df = client.get_eod(symbol, frm=None, to=None)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    if frm:
        df = df[df["date"] >= pd.Timestamp(frm)]
    if to:
        df = df[df["date"] <= pd.Timestamp(to)]
    return df.reset_index(drop=True)


def cached_intraday_1m(symbol: str, frm: dt.date, to: dt.date, refresh: bool = False) -> pd.DataFrame:
    path = _intraday_path(symbol, frm, to)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = client.get_intraday_1m(symbol, frm, to)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def cached_market_cap(symbol: str) -> Optional[float]:
    path = CACHE_DIR / "market_cap.parquet"
    if path.exists():
        table = pd.read_parquet(path)
    else:
        table = pd.DataFrame(columns=["symbol", "market_cap"]).set_index("symbol")
    if symbol in table.index:
        val = table.loc[symbol, "market_cap"]
        return None if pd.isna(val) else float(val)
    cap = client.get_market_cap(symbol)
    table.loc[symbol, "market_cap"] = cap
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)
    return cap
