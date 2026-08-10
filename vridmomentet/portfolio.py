"""Quintile long/short portfolio construction.

Brief: "Long topp-kvintil av s, short botten-kvintil; marknadsneutral ...
Sizing: w_i propto 1/sigma_i(60d) inom varje ben, 50% brutto per ben, tak
2% per namn ... Full ersattning av portfoljen varje vecka; inga stoppar,
ingen diskretion."

Market-neutrality here is exact by construction (not a consequence of
demeaning, unlike Oglegrinden's continuous reversal weights): the long leg
always sums to exactly +gross_per_leg and the short leg to exactly
-gross_per_leg, so the book's net exposure is exactly zero every week,
regardless of how the per-name cap perturbs the within-leg distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vridmomentet.config import PortfolioParams


def assign_bucket(s: pd.Series, n_buckets: int) -> pd.Series:
    """Cross-sectional bucket 0 (lowest s) .. n_buckets-1 (highest s), via
    rank-based cuts so ties/sparse tails don't break qcut. NaN stays NaN.
    """
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    ranks = valid.rank(method="first")
    buckets = np.floor((ranks - 1) / len(valid) * n_buckets).clip(upper=n_buckets - 1)
    # When there are fewer names than buckets (N < n_buckets), the formula
    # above can leave the single highest-ranked name short of bucket
    # n_buckets-1 (e.g. N=3, n_buckets=5 gives it bucket 3, not 4) --
    # force it explicitly so select_legs()'s top bucket is never silently
    # empty. A no-op whenever N >= n_buckets (already correct there).
    buckets.loc[ranks.idxmax()] = n_buckets - 1
    out = pd.Series(np.nan, index=s.index)
    out.loc[valid.index] = buckets
    return out


def select_legs(s: pd.Series, n_buckets: int) -> tuple[list[str], list[str]]:
    """(long_names, short_names) = (top bucket, bottom bucket) of a single
    date's cross-sectional signal, freshly selected -- no state, no
    hysteresis/retention, matching the brief's "full replacement" rule.
    """
    buckets = assign_bucket(s, n_buckets)
    long_names = sorted(buckets[buckets == n_buckets - 1].index)
    short_names = sorted(buckets[buckets == 0].index)
    return long_names, short_names


def inverse_vol_weights(names: list[str], vol: pd.Series) -> pd.Series:
    """1/sigma_i, normalized to sum to 1.0 across `names`. A name with
    missing or non-positive vol is dropped (can't size it).
    """
    if not names:
        return pd.Series(dtype=float)
    v = vol.reindex(names)
    v = v[(v > 0) & v.notna()]
    if v.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / v
    return inv / inv.sum()


def cap_and_renormalize(w: pd.Series, cap: float, tol: float = 1e-9, max_iter: int = 200) -> pd.Series:
    """Iterative water-filling: repeatedly clamp any name whose weight would
    exceed `cap` in magnitude to exactly `cap`, and redistribute the
    remaining budget proportionally among the still-free names, until
    stable. The cap is a hard constraint; the gross target (sum(|w|) before
    capping) is soft -- if there are too few names to hit the target
    without breaching the cap, achieved gross falls short rather than
    violating the cap (same convention as Oglegrinden's portfolio.py).
    """
    if w.empty:
        return w
    target_gross = float(w.abs().sum())
    sign = np.sign(w)
    mag = w.abs().copy()
    free = pd.Series(True, index=w.index)

    for _ in range(max_iter):
        capped_now = free & (mag >= cap - tol)
        mag.loc[capped_now] = cap
        free.loc[capped_now] = False
        remaining_budget = target_gross - mag[~free].sum()
        free_mag_sum = mag[free].sum()
        if free.sum() == 0 or free_mag_sum <= 0:
            break
        scale = remaining_budget / free_mag_sum
        if scale <= 1.0 + tol:
            mag.loc[free] = mag.loc[free] * max(scale, 0.0)
            break
        mag.loc[free] = mag.loc[free] * scale
    else:
        mag = np.minimum(mag, cap)

    mag = mag.clip(upper=cap)
    return sign * mag


@dataclass
class WeeklyPortfolio:
    weights: pd.Series           # ticker -> signed weight, long+short combined
    long_names: list[str]
    short_names: list[str]
    long_weights: pd.Series
    short_weights: pd.Series


def build_target_weights(
    s: pd.Series,
    vol: pd.Series,
    params: PortfolioParams = PortfolioParams(),
) -> WeeklyPortfolio:
    long_names, short_names = select_legs(s, params.n_buckets)

    if len(long_names) < params.min_names_per_leg or len(short_names) < params.min_names_per_leg:
        empty = pd.Series(dtype=float)
        return WeeklyPortfolio(weights=empty, long_names=[], short_names=[], long_weights=empty, short_weights=empty)

    long_raw = inverse_vol_weights(long_names, vol) * params.gross_per_leg
    short_raw = inverse_vol_weights(short_names, vol) * params.gross_per_leg

    long_w = cap_and_renormalize(long_raw, params.per_name_cap)
    short_w = cap_and_renormalize(-short_raw, params.per_name_cap)  # negative: short leg

    combined = pd.concat([long_w, short_w])
    return WeeklyPortfolio(
        weights=combined,
        long_names=list(long_w.index),
        short_names=list(short_w.index),
        long_weights=long_w,
        short_weights=short_w,
    )


def turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    all_names = prev_weights.index.union(new_weights.index)
    delta = new_weights.reindex(all_names, fill_value=0.0) - prev_weights.reindex(all_names, fill_value=0.0)
    return float(delta.abs().sum())
