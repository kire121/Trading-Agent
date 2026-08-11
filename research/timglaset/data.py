"""Panel loader: EODHD OHLCV fetch, split-adjusted volume, PIT-safe wide panel.

Panel/ADV/simple_returns structure ported from research/smittotalet/data.py
(branch claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b), which
itself just wraps cached-CSV loading -- adapted here to fetch fresh from the
canonical lib.eodhd_client (per docs/INSTRUKTION.md avsnitt 7: "det är den
kanoniska startpunkten för all FRAMTIDA strategikod som behöver EODHD-data")
rather than committing raw EODHD OHLCV to git the way the older sibling
branches did (see AVVIKELSER.md: this deliberately follows the newer,
explicitly-documented lib/eodhd_client.py policy of never committing
licensed vendor data, over the older per-branch convention).

Split-adjusted volume is new (see lib/eodhd_client.py::get_splits) --
Timglaset is the first strategy in this repo to read the volume column as a
signal input at all (spec §0/§4), so no prior branch had a reason to
split-adjust it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib import eodhd_client
from lib.hashutil import compute_config_hash

from . import config


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adjusted_close: pd.DataFrame
    volume: pd.DataFrame          # split-adjusted (opclock.compute_tau input)
    volume_raw: pd.DataFrame      # as reported, unadjusted (dollar-volume/ADV input)

    @property
    def tickers(self):
        return list(self.adjusted_close.columns)

    def simple_returns(self) -> pd.DataFrame:
        return self.adjusted_close.pct_change()

    def dollar_volume(self) -> pd.DataFrame:
        # Actual traded notional on the day: unadjusted close x unadjusted
        # volume are both "as-traded" that day, so their product is correct
        # without any split adjustment (which only matters when comparing
        # share COUNTS across a split event, not a single day's notional).
        return self.close * self.volume_raw

    def adv(self, lookback: int = None) -> pd.DataFrame:
        lookback = lookback or config.ADV_COST_LOOKBACK
        return self.dollar_volume().shift(1).rolling(lookback).mean()

    def pit_start(self) -> pd.Series:
        """First date with a defined adjusted_close, per ticker."""
        out = {}
        for col in self.adjusted_close.columns:
            valid = self.adjusted_close[col].dropna()
            out[col] = valid.index[0] if len(valid) else pd.NaT
        return pd.Series(out)

    def restrict(self, start=None, end=None) -> "Panel":
        def _cut(df):
            return df.loc[start:end]
        return Panel(
            open=_cut(self.open), high=_cut(self.high), low=_cut(self.low),
            close=_cut(self.close), adjusted_close=_cut(self.adjusted_close),
            volume=_cut(self.volume), volume_raw=_cut(self.volume_raw),
        )


def split_adjustment_factor(splits: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Cumulative factor to convert historical (pre-split) volume into
    post-split-equivalent share units: for each split, multiply every
    observation strictly BEFORE the split's ex-date by its ratio (shares
    outstanding increase by `ratio` from that date forward -- symmetric to
    how EODHD's adjusted_close divides historical PRICE by the same ratio;
    see lib/eodhd_client.py::get_splits for why price adjustment alone
    cannot be reused to derive this)."""
    factor = pd.Series(1.0, index=index)
    if splits is None or splits.empty:
        return factor
    for _, row in splits.iterrows():
        factor.loc[index < row["date"]] *= row["ratio"]
    return factor


def fetch_ticker_raw(ticker: str, exchange: str = "US", end: str = None,
                      cache_dir: str = None) -> dict:
    """Fetch one ticker's full OHLCV history + splits (raw, not yet
    combined into a panel). `end` bounds the fetch at data_end (IS_END) --
    the US panel is IS-only per spec §1.2.1 and its own is_end lock, so
    there is never a reason to fetch past it in this branch."""
    cache_dir = cache_dir or config.DATA_CACHE_DIR
    eod = eodhd_client.get_eod(ticker, start=config.FETCH_START, end=end,
                                exchange=exchange, cache_dir=cache_dir)
    splits = eodhd_client.get_splits(ticker, exchange=exchange, cache_dir=cache_dir)
    return {"eod": eod, "splits": splits}


def fetch_ticker_raw_suffixed(ticker_with_suffix: str, exchange=None, end: str = None,
                               cache_dir: str = None) -> dict:
    """Like fetch_ticker_raw, but for a ticker that already carries its own
    "TICKER.EXCHANGE" suffix (research/runraden/config.py:OOS_UNIVERSE's own
    convention -- e.g. "IWDA.LSE", "EUNK.XETRA" -- a mixed-exchange UCITS
    universe cannot use build_panel's single `exchange=` argument for every
    ticker the way the single-exchange US panel can). Splits on the LAST
    "." so a base ticker containing a literal dot is still handled. Accepts
    (and ignores) an `exchange` kwarg purely so it matches build_panel's
    uniform fetch_fn(ticker, exchange=..., end=..., cache_dir=...) call site."""
    base, _, parsed_exchange = ticker_with_suffix.rpartition(".")
    if not base:
        raise ValueError(f"expected 'TICKER.EXCHANGE', got: {ticker_with_suffix!r}")
    return fetch_ticker_raw(base, exchange=parsed_exchange, end=end, cache_dir=cache_dir)


def build_panel(tickers: list, exchange: str = "US", end: str = None,
                 cache_dir: str = None, fetch_fn=fetch_ticker_raw) -> tuple:
    """Fetch and assemble the wide panel. Returns (Panel, fetch_report) where
    fetch_report maps ticker -> {"n_rows": int, "first_date": ..., "error": ...}
    for tickers that fail to fetch (missing/renamed/delisted), so Steg 0a can
    account for them explicitly rather than silently vanishing from the panel."""
    raw = {}
    fetch_report = {}
    for t in tickers:
        try:
            d = fetch_fn(t, exchange=exchange, end=end, cache_dir=cache_dir)
            raw[t] = d
            fetch_report[t] = {"n_rows": int(len(d["eod"])), "error": None}
        except Exception as exc:  # noqa: BLE001 -- report and continue; Steg 0a decides pass/fail
            fetch_report[t] = {"n_rows": 0, "error": str(exc)}

    all_idx = None
    for d in raw.values():
        idx = d["eod"].index
        all_idx = idx if all_idx is None else all_idx.union(idx)
    all_idx = all_idx.sort_values() if all_idx is not None else pd.DatetimeIndex([])

    fields = {f: {} for f in config.FIELDS}
    volume_adj = {}
    for t in tickers:
        if t not in raw:
            for f in config.FIELDS:
                fields[f][t] = pd.Series(np.nan, index=all_idx)
            volume_adj[t] = pd.Series(np.nan, index=all_idx)
            continue
        eod = raw[t]["eod"].reindex(all_idx)
        for f in config.FIELDS:
            fields[f][t] = eod[f] if f in eod.columns else pd.Series(np.nan, index=all_idx)
        factor = split_adjustment_factor(raw[t]["splits"], all_idx)
        volume_adj[t] = fields["volume"][t] * factor

    panel = Panel(
        open=pd.DataFrame(fields["open"], index=all_idx),
        high=pd.DataFrame(fields["high"], index=all_idx),
        low=pd.DataFrame(fields["low"], index=all_idx),
        close=pd.DataFrame(fields["close"], index=all_idx),
        adjusted_close=pd.DataFrame(fields["adjusted_close"], index=all_idx),
        volume=pd.DataFrame(volume_adj, index=all_idx),
        volume_raw=pd.DataFrame(fields["volume"], index=all_idx),
    )
    return panel, fetch_report


def ticker_list_hash(tickers: list) -> str:
    """sha256 of the sorted ticker list (spec §1.2.4: registry keys on
    ticker lists, not panel names -- the "29/40-incidenten"). Reuses
    lib.hashutil.compute_config_hash verbatim (it hashes any JSON-safe
    object, not just a config dict)."""
    return compute_config_hash(sorted(tickers))
