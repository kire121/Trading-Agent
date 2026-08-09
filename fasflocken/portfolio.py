"""
Fasflocken (PH-1) -- portfolio construction.

Turns a cross-section of sector Z-scores into tradable ETF weights:

  1. eligible_sectors        -- point-in-time gating on ETF inception (XLRE/XLC).
  2. select_legs              -- rank -> top/bottom N with the "leaves top/bottom
                                  4" hysteresis rule.
  3. equal_dollar_direction   -- +1/N per long leg, -1/N per short leg (dollar
                                  neutral, base gross = 200%).
  4. vol_target_scale         -- scales the base book to hit 8% annualized vol
                                  using a 60d covariance matrix over the *book*
                                  (not per-leg), per the "Tidspilen lesson"
                                  cited in the spec.

Modeling assumption (spec is silent on the exact mechanics, flagged here
and in the README): "equal dollar per leg" is read as defining the full,
un-levered book -- N long legs at +1/N each, N short legs at -1/N each,
which is *already* 200% gross. Vol targeting can therefore only ever
scale this book **down** (k in [0, 1]); the "max brutto 200%" cap is
mathematically identical to k <= 1 in this construction, not a separate
binding constraint. This is the more conservative of the two readings
consistent with the text (the other being a smaller un-levered base that
vol-targeting can lever *up* toward the 200% cap); see README.md.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fasflocken.config import GICS_SECTORS, SECTOR_INCEPTION, SECTOR_TO_ETF


def eligible_sectors(as_of: _dt.date, sectors: tuple[str, ...] = GICS_SECTORS) -> list[str]:
    """Sectors whose SPDR ETF has already launched as of `as_of`."""
    return [s for s in sectors if SECTOR_INCEPTION[s] <= as_of]


@dataclass
class HysteresisState:
    long_sectors: frozenset[str] = field(default_factory=frozenset)
    short_sectors: frozenset[str] = field(default_factory=frozenset)


def select_legs(
    z: pd.Series,
    state: HysteresisState,
    n_legs: int,
    hysteresis_band: int,
    eligible: list[str] | None = None,
) -> tuple[HysteresisState, list[str], list[str]]:
    """Rank sectors by Z ascending; pick n_legs longs (lowest Z) / shorts
    (highest Z), retaining a currently-held sector as long as it stays
    within the top/bottom `hysteresis_band` band.

    z: Series of Z_s indexed by GICS sector name; NaN entries are dropped
    (not enough history yet -- e.g. XLC/XLRE pre-inception, or a sector
    that hasn't cleared its 104-week burn-in).
    """
    z = z.dropna()
    if eligible is not None:
        z = z[z.index.isin(eligible)]

    n_avail = len(z)
    if n_avail < 2 * hysteresis_band:
        raise ValueError(
            f"only {n_avail} eligible sectors with valid Z, need >= {2 * hysteresis_band} "
            f"for a hysteresis band of {hysteresis_band}"
        )
    if n_legs > hysteresis_band:
        raise ValueError("n_legs cannot exceed the hysteresis_band")

    ranked = z.sort_values(kind="mergesort")  # ascending: lowest Z first
    long_band = set(ranked.index[:hysteresis_band])
    short_band = set(ranked.index[-hysteresis_band:])

    lowest_first = list(ranked.index[:hysteresis_band])
    highest_first = list(ranked.index[-hysteresis_band:][::-1])

    kept_long = [s for s in lowest_first if s in state.long_sectors and s in long_band]
    new_long = list(kept_long)
    for s in lowest_first:
        if len(new_long) >= n_legs:
            break
        if s not in new_long:
            new_long.append(s)

    kept_short = [s for s in highest_first if s in state.short_sectors and s in short_band]
    new_short = list(kept_short)
    for s in highest_first:
        if len(new_short) >= n_legs:
            break
        if s not in new_short:
            new_short.append(s)

    new_long = new_long[:n_legs]
    new_short = new_short[:n_legs]

    new_state = HysteresisState(long_sectors=frozenset(new_long), short_sectors=frozenset(new_short))
    return new_state, new_long, new_short


def equal_dollar_direction(long_sectors: list[str], short_sectors: list[str]) -> pd.Series:
    """+1/N per long leg, -1/N per short leg, indexed by ETF ticker. Gross == 2.0."""
    n_long, n_short = len(long_sectors), len(short_sectors)
    weights: dict[str, float] = {}
    for s in long_sectors:
        weights[SECTOR_TO_ETF[s]] = weights.get(SECTOR_TO_ETF[s], 0.0) + 1.0 / n_long
    for s in short_sectors:
        weights[SECTOR_TO_ETF[s]] = weights.get(SECTOR_TO_ETF[s], 0.0) - 1.0 / n_short
    return pd.Series(weights)


def vol_target_scale(
    base_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    target_vol_ann: float,
    max_gross: float,
    trading_days_per_year: int = 252,
) -> float:
    """Leverage multiplier k such that k * base_weights realizes ~target_vol_ann,
    subject to k * gross(base_weights) <= max_gross and k >= 0.

    cov_matrix must be a daily-return covariance matrix (e.g. trailing 60d)
    covering at least the tickers in base_weights. Returns 0.0 if realized
    vol can't be estimated (missing/degenerate covariance).
    """
    tickers = base_weights.index
    missing = [t for t in tickers if t not in cov_matrix.index or t not in cov_matrix.columns]
    if missing:
        return 0.0
    sigma = cov_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    w = base_weights.to_numpy(dtype=float)
    if np.isnan(sigma).any():
        return 0.0

    daily_var = float(w @ sigma @ w)
    if not np.isfinite(daily_var) or daily_var <= 0:
        return 0.0
    realized_vol_ann = np.sqrt(daily_var * trading_days_per_year)

    base_gross = float(np.abs(w).sum())
    if base_gross <= 0:
        return 0.0

    k_vol = target_vol_ann / realized_vol_ann
    k_gross_cap = max_gross / base_gross
    return float(max(0.0, min(k_vol, k_gross_cap)))


def trailing_cov_matrix(etf_log_returns: pd.DataFrame, as_of: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """Daily-return covariance matrix over the `window_days` trailing
    observations up to and including `as_of` (point-in-time: no data after
    `as_of` is used).
    """
    hist = etf_log_returns.loc[etf_log_returns.index <= as_of].tail(window_days)
    return hist.cov() if len(hist) >= 2 else pd.DataFrame()


def compute_turnover(prev_weights: pd.Series, final_weights: pd.Series) -> float:
    """sum(|Delta w|) across the union of previously- and newly-held ETFs."""
    all_etfs = prev_weights.index.union(final_weights.index)
    delta = final_weights.reindex(all_etfs, fill_value=0.0) - prev_weights.reindex(all_etfs, fill_value=0.0)
    return float(delta.abs().sum())


def build_target_weights(
    z: pd.Series,
    state: HysteresisState,
    n_legs: int,
    hysteresis_band: int,
    cov_matrix: pd.DataFrame,
    target_vol_ann: float,
    max_gross: float,
    eligible: list[str] | None = None,
) -> tuple[HysteresisState, pd.Series, dict]:
    """One full weekly rebalance step: rank -> hysteresis -> equal-dollar -> vol target.

    Returns (new_state, final_weights_by_etf, info) where info carries the
    intermediate long/short leg lists and the vol-target scale k for
    diagnostics/logging.
    """
    new_state, long_sectors, short_sectors = select_legs(z, state, n_legs, hysteresis_band, eligible)
    base_weights = equal_dollar_direction(long_sectors, short_sectors)
    k = vol_target_scale(base_weights, cov_matrix, target_vol_ann, max_gross)
    final_weights = base_weights * k
    info = {
        "long_sectors": long_sectors,
        "short_sectors": short_sectors,
        "k": k,
        "base_gross": float(base_weights.abs().sum()),
        "final_gross": float(final_weights.abs().sum()),
    }
    return new_state, final_weights, info
