"""Wide OHLCV panels and the derived quantities the signal/portfolio/backtest
layers need (adjusted OHLC, dollar volume, log returns, ADV, realized vol,
point-in-time eligibility).

Mirrors the shape of Oglegrinden's `data.py::Panel`, generalized from ~25
ETFs to an ~900-name individual-equity universe and extended to carry
volume (Oglegrinden's ETF study never needed dollar volume; Vridmomentet's
signal is defined on it).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vridmomentet import config
from vridmomentet.universe import MembershipInterval, PointInTimeMembership, UniverseProvider


def _adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame | None:
    """Reconstruct split/dividend-adjusted open/high/low from raw OHLC using
    the same day's adjusted_close/close factor (EODHD gives adjusted close
    only). Same approximation Oglegrinden documents for its adjusted_open:
    immaterial over the holding periods this strategy uses.

    Returns None if this ticker's fetched data is missing "close" or
    "adjusted_close" -- both required to compute the adjustment factor at
    all -- so the caller can drop just this one ticker rather than crash
    the whole panel build on a single ragged row (a real, if rare,
    possibility across ~1200 tickers of real vendor data, some of it for
    obscure/delisted names).
    """
    if "close" not in df.columns or "adjusted_close" not in df.columns:
        return None
    out = df.copy()
    factor = (out["adjusted_close"] / out["close"]).replace([np.inf, -np.inf], np.nan)
    for col in ("open", "high", "low"):
        if col in out.columns:
            out[f"adj_{col}"] = out[col] * factor
    out["adj_close"] = out["adjusted_close"]
    return out


@dataclass
class Panel:
    """Wide (date x ticker) frames, all point-in-time (no forward fill across
    a name's own not-yet-listed/delisted gaps -- those stay NaN).
    """

    close: pd.DataFrame
    adj_close: pd.DataFrame
    adj_open: pd.DataFrame
    volume: pd.DataFrame
    membership: PointInTimeMembership

    dollar_volume: pd.DataFrame = field(init=False)
    log_returns: pd.DataFrame = field(init=False)
    adv20: pd.DataFrame = field(init=False)
    adv60: pd.DataFrame = field(init=False)
    vol60: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        # Raw (unadjusted) close, matching Oglegrinden's convention: dollar
        # volume should reflect the price actually paid that day, not a
        # split/dividend-adjusted figure.
        self.dollar_volume = self.close * self.volume
        self.log_returns = np.log(self.adj_close / self.adj_close.shift(1))
        # ADV20 is the brief's own eligibility-filter window (config.ADV_LOOKBACK_DAYS,
        # "ADV20 > 20 MUSD"); ADV60/vol60 are Panel's own general-purpose derived
        # quantities (consumed by signal normalization and portfolio sizing
        # respectively, which happen to also default to 60d, but Panel doesn't
        # couple to either of those config sections specifically).
        self.adv20 = self.dollar_volume.rolling(config.ADV_LOOKBACK_DAYS, min_periods=15).mean()
        self.adv60 = self.dollar_volume.rolling(60, min_periods=45).mean()
        self.vol60 = self.log_returns.rolling(60, min_periods=45).std(ddof=1)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def eligible_on(self, as_of: pd.Timestamp, price_min: float, adv_min: float) -> pd.Index:
        """Point-in-time eligible names: in the index as of `as_of`, priced
        above `price_min`, with trailing ADV20 above `adv_min`, and with
        history through `as_of`.
        """
        if as_of not in self.close.index:
            return pd.Index([])
        member_tickers = set(self.membership.constituents_as_of(as_of.date()))
        row_close = self.close.loc[as_of]
        row_adv = self.adv20.loc[as_of]
        ok = (row_close > price_min) & (row_adv > adv_min)
        names = ok[ok].index
        names = [t for t in names if t in member_tickers]
        return pd.Index(sorted(names))


def build_panel(provider: UniverseProvider, tickers: list[str], start: _dt.date, end: _dt.date) -> Panel:
    raw = provider.prices(tickers, start, end)
    if not raw:
        raise RuntimeError("no price data fetched for any ticker in the requested universe/date range")

    adjusted: dict[str, pd.DataFrame] = {}
    for t, df in raw.items():
        result = _adjust_ohlc(df)
        if result is None:
            print(f"[data] WARNING: dropping {t}: missing close/adjusted_close in fetched data")
            continue
        adjusted[t] = result
    if not adjusted:
        raise RuntimeError("no ticker had usable close/adjusted_close data after adjustment")

    all_dates = sorted(set().union(*[df.index for df in adjusted.values()]))
    idx = pd.DatetimeIndex(all_dates)

    def _wide(col: str) -> pd.DataFrame:
        # .get(col) rather than [col]: a ticker missing e.g. "volume" (or,
        # after adjustment, "adj_open" if "open" itself was absent) should
        # contribute an all-NaN column for that field, not crash panel
        # construction for the other ~1200 tickers that do have it.
        series = {}
        for t, df in adjusted.items():
            series[t] = df[col].reindex(idx) if col in df.columns else pd.Series(np.nan, index=idx)
        return pd.DataFrame(series, index=idx).sort_index()

    close = _wide("close")
    adj_close = _wide("adj_close")
    adj_open = _wide("adj_open")
    volume = _wide("volume")

    return Panel(
        close=close,
        adj_close=adj_close,
        adj_open=adj_open,
        volume=volume,
        membership=provider.membership(),
    )
