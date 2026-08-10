"""
Dammluckan -- panel construction from the cached EODHD CSVs.

Design choice (DECLARED, and worth stating explicitly): record/occupation
detection (M_t, E+/E-, O+/O-) runs on the RAW (unadjusted) close, because the
hypothesis is about anchoring to actual historical traded price levels
(resting limit orders parked at literal past highs/lows) -- using a
dividend-adjusted series would silently shift those historical levels for
higher-yielding names (HYG, LQD, TLT, VNQ, EEM...) and distort exactly the
mechanism being tested. Everything downstream of the entry decision (returns,
volatility, sizing, Sharpe) uses the adjusted series, which is the standard,
correct convention for total-return performance measurement.
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from . import config

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

FIELDS = ["open", "high", "low", "close", "adjusted_close", "volume"]


def _load_field(universe_label, field):
    path = os.path.join(DATA_DIR, f"{universe_label}_{field}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


@dataclass
class Panel:
    tickers: list
    raw_close: pd.DataFrame
    raw_open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    adj_close: pd.DataFrame
    adj_open: pd.DataFrame
    volume: pd.DataFrame
    dollar_volume: pd.DataFrame = field(init=False)
    log_returns: pd.DataFrame = field(init=False)
    band_vol: pd.DataFrame = field(init=False)      # sigma_hat for O+/O- band width
    sizing_vol: pd.DataFrame = field(init=False)     # sigma_hat for position sizing
    adv: pd.DataFrame = field(init=False)            # trailing dollar ADV for costs

    def __post_init__(self):
        # dollar volume uses RAW close (matches Vridmomentet convention:
        # "close * volume (raw close, not adjusted)")
        self.dollar_volume = self.raw_close * self.volume
        self.log_returns = np.log(self.adj_close / self.adj_close.shift(1))
        self.band_vol = self.log_returns.rolling(
            config.VOL_LOOKBACK, min_periods=config.VOL_MIN_PERIODS
        ).std(ddof=1)
        self.sizing_vol = self.log_returns.rolling(
            config.SIZING_VOL_LOOKBACK, min_periods=config.SIZING_VOL_LOOKBACK
        ).std(ddof=1)
        self.adv = self.dollar_volume.rolling(
            config.ADV_COST_LOOKBACK, min_periods=config.ADV_COST_LOOKBACK // 2
        ).mean()

    @property
    def dates(self):
        return self.raw_close.index

    def eligible_on(self, as_of) -> pd.Index:
        """PIT-eligible: ticker has traded (non-NaN raw close) as of this date."""
        row = self.raw_close.loc[as_of]
        return row[row.notna()].index

    def slice(self, start=None, end=None) -> "Panel":
        def cut(df):
            return df.loc[start:end]
        return Panel(
            tickers=self.tickers,
            raw_close=cut(self.raw_close), raw_open=cut(self.raw_open),
            high=cut(self.high), low=cut(self.low),
            adj_close=cut(self.adj_close), adj_open=cut(self.adj_open),
            volume=cut(self.volume),
        )


def _adjust_open(raw_open, raw_close, adj_close):
    """Reconstruct adjusted open from the adj_close/close cumulative factor.

    Same mechanic as Vridmomentet's _adjust_ohlc: EODHD only gives an
    adjusted close, so open/high/low are backed out via the ratio.
    """
    factor = adj_close / raw_close
    return raw_open * factor


def build_panel(universe_label) -> Panel:
    raw_open = _load_field(universe_label, "open")
    raw_close = _load_field(universe_label, "close")
    high = _load_field(universe_label, "high")
    low = _load_field(universe_label, "low")
    adj_close = _load_field(universe_label, "adjusted_close")
    volume = _load_field(universe_label, "volume")

    # Widen all frames to a shared union index/columns without forward-fill:
    # absence of data means absence of eligibility, not a stale carried price.
    idx = raw_close.index
    cols = raw_close.columns
    raw_open = raw_open.reindex(index=idx, columns=cols)
    high = high.reindex(index=idx, columns=cols)
    low = low.reindex(index=idx, columns=cols)
    adj_close = adj_close.reindex(index=idx, columns=cols)
    volume = volume.reindex(index=idx, columns=cols)

    adj_open = _adjust_open(raw_open, raw_close, adj_close)

    tickers = list(cols)
    return Panel(
        tickers=tickers,
        raw_close=raw_close, raw_open=raw_open,
        high=high, low=low,
        adj_close=adj_close, adj_open=adj_open,
        volume=volume,
    )


def load_primary_panel() -> Panel:
    return build_panel("primary")


def load_secondary_panel() -> Panel:
    return build_panel("secondary")
