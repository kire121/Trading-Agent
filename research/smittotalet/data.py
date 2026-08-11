"""Panel loader -- reads the cached per-field CSVs built by fetch_data.py."""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

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

    def adv(self, lookback=config.ADV_COST_LOOKBACK) -> pd.DataFrame:
        return self.dollar_volume().shift(1).rolling(lookback).mean()


def _load_field(field: str) -> pd.DataFrame:
    path = os.path.join(config.DATA_DIR, f"primary_{field}.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_primary_panel() -> Panel:
    fields = {f: _load_field(f) for f in config.FIELDS}
    common_cols = set.intersection(*(set(df.columns) for df in fields.values()))
    common_idx = fields["close"].index
    for df in fields.values():
        common_idx = common_idx.union(df.index)
    aligned = {
        f: df.reindex(index=sorted(common_idx), columns=sorted(common_cols))
        for f, df in fields.items()
    }
    return Panel(
        open=aligned["open"], high=aligned["high"], low=aligned["low"],
        close=aligned["close"], adjusted_close=aligned["adjusted_close"],
        volume=aligned["volume"],
    )


def restrict(panel: Panel, start=None, end=None) -> Panel:
    def _cut(df):
        return df.loc[(slice(start, end)), :]
    return Panel(
        open=_cut(panel.open), high=_cut(panel.high), low=_cut(panel.low),
        close=_cut(panel.close), adjusted_close=_cut(panel.adjusted_close),
        volume=_cut(panel.volume),
    )
