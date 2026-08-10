"""
Synthetic minute-bar generator with an injectable, KNOWN periodicity and
directional drift. This exists to answer one question before a single dollar
of real API budget or trading capital is at stake: does the detector actually
recover ground truth?

It is not a backtest and produces no claim about real markets -- see
tests/test_signal.py for how it's used, and README.md for why this step
comes first.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from .bars import session_minute_grid

N_MINUTES = len(session_minute_grid())


def _trading_days(start: dt.date, n_days: int) -> list:
    days = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def make_symbol_dense(
    rng: np.random.Generator,
    n_days: int,
    start: dt.date = dt.date(2024, 1, 2),
    base_volume: float = 800.0,
    inject_tau: Optional[int] = None,
    inject_from_day: int = 0,
    inject_strength: float = 1.5,
    inject_phase: int = 0,
    direction_bias: float = 0.15,
    common_tau: Optional[int] = None,
    common_strength: float = 0.6,
    noise_sigma: float = 0.30,
    ret_sigma: float = 0.0006,
) -> pd.DataFrame:
    """One symbol's worth of dense (session_date, minute_of_day, close, volume,
    ret) rows, `n_days` sessions starting at `start`.

    If inject_tau is set, from day `inject_from_day` onward every
    `inject_tau`-th minute (starting at `inject_phase`) gets a volume multiplier
    of exp(inject_strength) and a return drawn with mean `direction_bias *
    ret_sigma` instead of 0 -- i.e. a slice of a metaorder walking the price in
    one direction. If common_tau is set, EVERY day (regardless of injection
    window) gets a smaller common-to-all-symbols bump at that period, standing
    in for market-wide on-the-hour hedging rhythm that cross-sectional
    standardization is supposed to cancel out.
    """
    days = _trading_days(start, n_days)
    t = np.arange(N_MINUTES)
    u_shape = 1.0 + 0.35 * ((t - N_MINUTES / 2) / (N_MINUTES / 2)) ** 2  # mild residual U within the trimmed window

    rows = []
    for day_idx, day in enumerate(days):
        log_vol = np.log(base_volume * u_shape) + rng.normal(0, noise_sigma, N_MINUTES)
        ret = rng.normal(0, ret_sigma, N_MINUTES)

        if common_tau:
            common_mask = (t % common_tau) == 0
            log_vol[common_mask] += common_strength

        if inject_tau is not None and day_idx >= inject_from_day:
            burst_mask = (t % inject_tau) == (inject_phase % inject_tau)
            log_vol[burst_mask] += inject_strength
            ret[burst_mask] = rng.normal(direction_bias * ret_sigma, ret_sigma, burst_mask.sum())

        volume = np.exp(log_vol)
        close = 100.0 * np.exp(np.cumsum(ret))
        ret_out = ret.copy()
        ret_out[0] = 0.0  # first minute of the trimmed window has no defined intra-window return

        rows.append(pd.DataFrame({
            "session_date": day,
            "minute_of_day": t + 585,  # matches bars.session_minute_grid's 09:45 start
            "close": close,
            "volume": volume,
            "ret": ret_out,
        }))

    return pd.concat(rows, ignore_index=True)


def eod_from_dense(dense: pd.DataFrame) -> pd.DataFrame:
    """Derives a daily EOD OHLCV frame (matching eodhd_client.get_eod's
    schema) from a dense intraday grid, so integration tests exercise
    daily.py/universe.py against data that's internally consistent with the
    intraday series feeding signal.py, rather than two unrelated random
    series."""
    g = dense.groupby("session_date")
    out = pd.DataFrame({
        "date": pd.to_datetime(list(g.groups.keys())),
        "open": g["close"].first().values,
        "high": g["close"].max().values,
        "low": g["close"].min().values,
        "close": g["close"].last().values,
        "volume": g["volume"].sum().values,
    })
    out["adjusted_close"] = out["close"]
    return out.sort_values("date").reset_index(drop=True)
