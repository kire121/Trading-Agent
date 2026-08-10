"""
Turns raw EODHD 1-minute bars (UTC, extended hours, irregular -- rows exist
only where a trade printed) into a regular per-session-date minute grid
trimmed to RTH excl. first/last 15 minutes, which is the exact input the
spec's u_t detrending step expects.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import SPEC

_TZ = ZoneInfo(SPEC.session.tz)


def session_minute_grid() -> pd.DataFrame:
    """Minutes-since-midnight grid for the trimmed RTH window, e.g. 09:45..15:44
    for the default 09:30-16:00 session with 15-min trim on each side."""
    open_h, open_m = (int(x) for x in SPEC.session.rth_open.split(":"))
    close_h, close_m = (int(x) for x in SPEC.session.rth_close.split(":"))
    start_min = open_h * 60 + open_m + SPEC.session.exclude_open_minutes
    end_min = close_h * 60 + close_m - SPEC.session.exclude_close_minutes
    minutes = np.arange(start_min, end_min)  # half-open: [start, end)
    return pd.DataFrame({"minute_of_day": minutes})


def to_rth_grid(raw_1m: pd.DataFrame) -> pd.DataFrame:
    """
    raw_1m: columns ['datetime' (UTC, tz-aware), 'close', 'volume', ...] as
    returned by eodhd_client.get_intraday_1m / data_cache.cached_intraday_1m.

    Returns a dense frame indexed by (session_date, minute_of_day) covering
    every trimmed-RTH minute, with volume=0 and forward-filled close on
    minutes where no trade printed (illiquid names skip minutes rather than
    reporting a zero-volume bar).
    """
    if raw_1m.empty:
        return pd.DataFrame(columns=["session_date", "minute_of_day", "close", "volume", "ret"])

    df = raw_1m.copy()
    local = df["datetime"].dt.tz_convert(_TZ)
    df["session_date"] = local.dt.date
    df["minute_of_day"] = local.dt.hour * 60 + local.dt.minute

    grid = session_minute_grid()
    sessions = sorted(df["session_date"].unique())
    full_index = pd.MultiIndex.from_product([sessions, grid["minute_of_day"]], names=["session_date", "minute_of_day"])

    df = df.set_index(["session_date", "minute_of_day"])[["close", "volume"]]
    df = df[~df.index.duplicated(keep="last")]
    dense = df.reindex(full_index)

    dense["close"] = dense.groupby(level="session_date")["close"].ffill()
    dense["close"] = dense.groupby(level="session_date")["close"].bfill()
    dense["volume"] = dense["volume"].fillna(0.0)

    dense["ret"] = dense.groupby(level="session_date")["close"].transform(lambda s: np.log(s).diff())
    dense.loc[dense.groupby(level="session_date").head(1).index, "ret"] = 0.0

    return dense.reset_index()


def sessions_with_full_coverage(dense: pd.DataFrame, min_nonzero_frac: float = 0.5) -> list:
    """Session dates where the name actually traded through most of the
    trimmed window -- guards against thin names producing an all-zero
    cepstrum that would otherwise look like a (spurious) flat spectrum."""
    if dense.empty:
        return []
    frac_nonzero = dense.groupby("session_date")["volume"].apply(lambda v: (v > 0).mean())
    return sorted(frac_nonzero[frac_nonzero >= min_nonzero_frac].index.tolist())
