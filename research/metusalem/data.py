"""Panel construction + OOS-gated EODHD data loading.

All real data access goes through lib.oos_loader.load_market_data() with a
custom fetch_fn built on lib.eodhd_client (rule 3: OOS-lock via the shared
loader, never bypassed; --unlock-oos is never passed during development).
The fetch_fn itself performs the AUTHORITATIVE enforce_oos_gate() call on
the exact config it receives, per lib/oos_loader.py's own contract.

Panel dataclass mirrors research/smittotalet/data.py::Panel (same shape,
same field names) -- not imported cross-package (each research/<strategy>
dir is self-contained, matching the repo's existing convention), but built
to the identical, already-tested contract so basbok.py (ported from
smittotalet/tsmom.py) works unmodified against it.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lib import eodhd_client
from lib.loader_guard import require_loader_active
from lib.oos_loader import enforce_oos_gate, load_market_data

from . import config


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adjusted_close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def tickers(self):
        return list(self.adjusted_close.columns)

    def simple_returns(self) -> pd.DataFrame:
        return self.adjusted_close.pct_change()

    def dollar_volume(self) -> pd.DataFrame:
        return self.close * self.volume

    def adv(self, lookback: int = 63) -> pd.DataFrame:
        return self.dollar_volume().shift(1).rolling(lookback).mean()


def _split_ticker(raw: str) -> tuple:
    """"IWDA.LSE" -> ("IWDA", "LSE"); "SPY" -> ("SPY", "US") (Sec 6:
    IS_UNIVERSE tickers are bare US ETFs; lib.eodhd_client's own convention
    is a bare ticker + separate exchange arg -- see its docstring's warning
    about the Runraden/Smittotalet suffix-convention mismatch)."""
    if "." in raw:
        ticker, exchange = raw.rsplit(".", 1)
        return ticker, exchange
    return raw, "US"


def _fetch_universe(universe: list, start: str, end: str, cache_dir: str) -> Panel:
    fields = {f: {} for f in config.FIELDS}
    for raw_ticker in universe:
        ticker, exchange = _split_ticker(raw_ticker)
        df = eodhd_client.get_eod(ticker, start=start, end=end, exchange=exchange,
                                   cache_dir=cache_dir, env_var=config.EODHD_API_KEY_ENV)
        for f in config.FIELDS:
            if f in df.columns:
                fields[f][raw_ticker] = df[f]

    frames = {f: pd.DataFrame(cols) for f, cols in fields.items()}
    common_idx = None
    for df in frames.values():
        common_idx = df.index if common_idx is None else common_idx.union(df.index)
    common_cols = sorted(set.intersection(*(set(df.columns) for df in frames.values())))
    aligned = {f: df.reindex(index=sorted(common_idx), columns=common_cols) for f, df in frames.items()}
    return Panel(open=aligned["open"], high=aligned["high"], low=aligned["low"],
                 close=aligned["close"], adjusted_close=aligned["adjusted_close"],
                 volume=aligned["volume"])


def _make_fetch_fn(universe: list, cache_dir: str):
    def _fetch(cfg: dict, twin: str, *, unlock_oos: bool, log_path):
        require_loader_active()
        enforce_oos_gate(cfg, unlock_oos, log_path)
        return _fetch_universe(universe, cfg["data_start"], cfg["data_end"], cache_dir)
    return _fetch


def load_is_panel(data_end: str = config.IS_DATA_END, unlock_oos: bool = False) -> Panel:
    """Loads the US 40-ETF IS panel through the OOS-locked loader.
    data_end<=is_end always for IS work -- unlock_oos MUST stay False here;
    the OOS panel has its own dedicated loader below."""
    cfg = {
        "strategy_name": config.STRATEGY_NAME,
        "seed": config.SEED,
        "data_start": config.HISTORY_START,
        "data_end": data_end,
        "is_end": config.IS_END,
    }
    fetch_fn = _make_fetch_fn(config.IS_UNIVERSE, config.CACHE_DIR)
    return load_market_data(cfg, twin="primary", unlock_oos=unlock_oos, fetch_fn=fetch_fn)


def load_oos_panel(unlock_oos: bool) -> Panel:
    """Loads the UCITS OOS panel. unlock_oos must be explicitly True (rule 3:
    only after the user has cleared it in this session, following the full
    IS-A..IS-B pass).

    lib.oos_loader's gate is DATE-based (data_end > is_end); Metusalem's OOS
    split is UNIVERSE-based (a disjoint UCITS panel, not a later date range
    on the same US-ETF panel -- spec Sec.11) with the SAME data_end as IS
    (both "..2026-06-30"), so the literal date comparison would never trip.
    To route this universe switch through the same shared, logged gate
    (rather than bypassing it -- rule 3 forbids that), the config fed to the
    loader uses a sentinel is_end (config.OOS_LOCK_SENTINEL_IS_END, well
    before the OOS panel's own inception dates) so that data_end > is_end
    is unconditionally true for ANY attempt to fetch this universe --
    stricter than, never weaker than, the mechanism's literal date-based
    design. This sentinel config is used ONLY to drive lib.oos_loader's gate
    and its logs/oos_unlocks.jsonl entry; it is NOT the study-level config
    used for results.json/config_frozen.yaml (see run_research.py), which
    records Metusalem's real is_end=2026-06-30 throughout."""
    cfg = {
        "strategy_name": config.STRATEGY_NAME,
        "seed": config.SEED,
        "data_start": config.HISTORY_START,
        "data_end": config.OOS_DATA_END,
        "is_end": config.OOS_LOCK_SENTINEL_IS_END,
    }
    fetch_fn = _make_fetch_fn(config.OOS_UNIVERSE, config.CACHE_DIR)
    return load_market_data(cfg, twin="primary", unlock_oos=unlock_oos, fetch_fn=fetch_fn)
