"""
Fasflocken (PH-1) -- per-sector signal pipeline glue.

Wires universe.UniverseProvider -> signals.py/twin.py into one daily R_s(t)
/ Corr_s(t) series per sector, plus their weekly Z-scores, respecting
point-in-time constituent membership.

Point-in-time membership note: within a single sector, a name almost
always has one contiguous membership interval (it joins, and later either
stays or gets removed once -- re-entry after removal is rare). We
therefore query `provider.constituents(sector, ·)` at a coarse cadence
(`membership_refresh_days`, default ~1 trading week, matching the
strategy's own rebalance cadence) and forward-fill that snapshot until
the next query, rather than hitting the provider every single day. This
is a deliberate accuracy/cost trade-off, not a look-ahead: every snapshot
is a point-in-time query for a date <= the days it's applied to.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fasflocken.config import SignalParams
from fasflocken.signals import (
    causal_bandpass,
    rolling_analytic_phase,
    kuramoto_order_parameter,
    resample_weekly_last,
    rolling_zscore,
)
from fasflocken.twin import mean_pairwise_correlation
from fasflocken.universe import UniverseProvider


def build_membership_mask(
    provider: UniverseProvider,
    sector: str,
    tickers: list[str],
    dates: pd.DatetimeIndex,
    membership_refresh_days: int = 5,
) -> pd.DataFrame:
    """Boolean (T, N) mask: True where `ticker` is a point-in-time constituent
    of `sector` on that date, refreshed every `membership_refresh_days`.
    """
    mask = pd.DataFrame(False, index=dates, columns=tickers)
    refresh_points = list(range(0, len(dates), max(1, membership_refresh_days)))
    if refresh_points[-1] != len(dates) - 1:
        refresh_points.append(len(dates) - 1)

    for i, idx in enumerate(refresh_points):
        as_of = dates[idx].date()
        members = set(provider.constituents(sector, as_of))
        end_idx = refresh_points[i + 1] if i + 1 < len(refresh_points) else len(dates)
        cols = [t for t in tickers if t in members]
        if cols:
            mask.iloc[idx:end_idx, mask.columns.get_indexer(cols)] = True
    return mask


@dataclass
class SectorSignal:
    sector: str
    R_daily: pd.Series
    Z_weekly: pd.Series
    corr_daily: pd.Series
    Zc_weekly: pd.Series
    n_constituents_daily: pd.Series


def compute_sector_signal(
    provider: UniverseProvider,
    sector: str,
    start: _dt.date,
    end: _dt.date,
    params: SignalParams,
    membership_refresh_days: int = 5,
) -> SectorSignal:
    """Full daily R_s(t)/Corr_s(t) and weekly Z_s(t)/Zc_s(t) for one sector.

    Every step here is causal in t (see signals.py docstring): computing
    the whole history in one vectorized pass is equivalent to computing it
    online week by week during the backtest loop.
    """
    dates = provider.trading_calendar(start, end)

    universe_probe_dates = pd.date_range(start, end, freq="MS")
    probe_tickers: set[str] = set()
    for d in list(universe_probe_dates) + [pd.Timestamp(start), pd.Timestamp(end)]:
        probe_tickers |= set(provider.constituents(sector, d.date()))
    tickers = sorted(probe_tickers)

    if not tickers:
        empty = pd.Series(dtype=float, index=dates)
        empty_w = pd.Series(dtype=float)
        return SectorSignal(sector, empty, empty_w, empty.copy(), empty_w.copy(), empty.copy())

    # astype(float) defensively: a provider returning even one non-numeric
    # (e.g. object-dtype, from an empty/unresolvable ticker column) column
    # would otherwise make np.log() fail across the whole DataFrame.
    prices = provider.prices(tickers, start, end).reindex(index=dates, columns=tickers).astype(float)
    log_prices = np.log(prices)
    returns = log_prices.diff()

    mask = build_membership_mask(provider, sector, tickers, dates, membership_refresh_days)
    returns_masked = returns.where(mask)

    bandpassed = causal_bandpass(returns_masked, params.band_low_days, params.band_high_days, params.filter_order)
    phase = rolling_analytic_phase(bandpassed, params.analytic_window)

    n_constituents_daily = mask.sum(axis=1).astype(float)
    n_constituents_daily.name = "n_constituents"

    R_daily = kuramoto_order_parameter(phase, params.min_constituents)
    R_daily.index = dates
    R_daily.name = sector

    corr_daily = mean_pairwise_correlation(bandpassed, params.analytic_window)
    corr_daily.name = sector

    R_weekly = resample_weekly_last(R_daily)
    Z_weekly = rolling_zscore(R_weekly, params.z_lookback_weeks)
    Z_weekly.name = sector

    corr_weekly = resample_weekly_last(corr_daily)
    Zc_weekly = rolling_zscore(corr_weekly, params.z_lookback_weeks)
    Zc_weekly.name = sector

    return SectorSignal(sector, R_daily, Z_weekly, corr_daily, Zc_weekly, n_constituents_daily)


def zscore_from_R_daily(R_daily: pd.Series, z_lookback_weeks: int) -> pd.Series:
    """Recompute just the weekly Z-score from an already-computed daily R_s(t)
    (or Corr_s(t)) series, for a different z_lookback -- avoids rerunning the
    Hilbert pipeline when only the z-lookback grid dimension changes (see
    grid_search.py).
    """
    R_weekly = resample_weekly_last(R_daily)
    z = rolling_zscore(R_weekly, z_lookback_weeks)
    z.name = R_daily.name
    return z


def build_z_panel(signals: dict[str, SectorSignal], twin: bool = False) -> pd.DataFrame:
    """Wide (weekly_date x sector) panel of Z_s (or Zc_s if twin=True)."""
    series = {s: (sig.Zc_weekly if twin else sig.Z_weekly) for s, sig in signals.items()}
    return pd.DataFrame(series).sort_index()
