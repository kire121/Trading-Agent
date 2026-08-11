"""Position construction: raw signal -> jointly-scaled, capped, banded weights.

x_{i,t+1} = ghat(w_{i,t}) / sigma_i(60d)

Design choices made explicit (the spec names these but does not pin down
exact formulas -- see README "Design choices & assumptions"):

  - "gemensam skalning ... loest gemensamt" (jointly-solved common scaling)
    is implemented as a fresh, analytic, cross-sectional solve *every week*
    (not a single historically-fit constant): assuming ~zero cross-asset
    correlation, portfolio vol ~= sqrt(sum((w_i * sigma_weekly_i)^2)); the
    common scalar k is the smaller of the vol-target-implied scalar and the
    200% gross-cap-implied scalar, so whichever constraint binds in a given
    week determines k for that week (a "calibration path", not one rescale
    fit once and frozen).
  - 10% gross-per-name cap is applied after the common scaling, by clipping.
  - No-trade band: a name's weight only changes if the proposed change
    exceeds 0.15 * (this week's gross / number of active names); otherwise
    last week's held weight is carried forward.
  - Names with no valid signal this week (burn-in / <4-day / no refit yet)
    get a target weight of 0, subject to the same no-trade band.
  - The joint cross-sectional scaling/capping/banding step groups rows by
    the ISO calendar week of the *signal* date (t_signal), not by each
    row's own literal `execution_date`. Individual assets can have slightly
    different execution_date values within the same economic week (e.g. one
    ETF missing a single trading day), and grouping by the raw date would
    silently split such an asset into its own single-name "cross-section"
    for that week, distorting its vol-target scaling and gross cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_vol_est(weights: np.ndarray, sigma_weekly: np.ndarray) -> float:
    """Zero-cross-correlation analytic portfolio vol estimate."""
    return float(np.sqrt(np.sum((weights * sigma_weekly) ** 2)))


def solve_common_scale(x_raw: np.ndarray, sigma_weekly: np.ndarray,
                        target_vol: float, max_gross: float) -> float:
    gross_raw = float(np.sum(np.abs(x_raw)))
    if gross_raw <= 0:
        return 0.0
    vol_raw = portfolio_vol_est(x_raw, sigma_weekly)
    k_vol = (target_vol / vol_raw) if vol_raw > 0 else np.inf
    k_gross = max_gross / gross_raw
    return float(min(k_vol, k_gross))


def apply_no_trade_band(proposed: dict[str, float], held: dict[str, float],
                         band: float) -> dict[str, float]:
    """A name's weight only changes if |proposed - held| >= band; otherwise
    the previously-held weight is carried forward unchanged. Pure function,
    isolated from the scaling/capping logic for direct unit testing."""
    out = {}
    for asset, proposed_w in proposed.items():
        prev = held.get(asset, 0.0)
        out[asset] = prev if abs(proposed_w - prev) < band else proposed_w
    return out


def build_positions(scored_panel: pd.DataFrame, target_vol: float, max_gross: float,
                     max_gross_per_name: float, no_trade_band: float,
                     signal_col: str = "ghat") -> pd.DataFrame:
    """scored_panel columns required: asset, t_signal, execution_date,
    <signal_col>, sigma_60d, table_id. One row per (asset, week). Returns the
    same rows with an added `weight` column (the position held over the
    *next* week, i.e. the week whose return is `next_week_return` / target
    `z_next`).

    `signal_col` lets the twin strategies (T1-T4, see twins.py) reuse this
    exact same scaling/capping/no-trade-band machinery with their own raw
    signal in place of ghat, so twin vs. real-strategy comparisons are
    risk-matched by construction.
    """
    df = scored_panel.copy()
    df["sigma_weekly"] = df["sigma_60d"] * np.sqrt(5.0)
    # x_raw = signal / sigma_60d, per the spec formula. (ghat itself is estimated on the
    # sigma_weekly-standardised target z, so this ratio carries a fixed sqrt(5) scale
    # baked in uniformly across assets/weeks -- it is absorbed into the common scalar k
    # solved below and has no effect on relative cross-sectional weights.)
    df["x_raw"] = np.where(
        df[signal_col].notna() & (df["sigma_60d"] > 0),
        df[signal_col] / df["sigma_60d"],
        np.nan,
    )

    iso = df["t_signal"].dt.isocalendar()
    df["week_bucket"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    held: dict[str, float] = {}
    weight_col = np.zeros(len(df))

    for _bucket, week_rows in df.groupby("week_bucket", sort=True):
        idx = week_rows.index.to_numpy()
        assets = week_rows["asset"].to_numpy()
        x_raw = week_rows["x_raw"].to_numpy(dtype=float)
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        sigma_weekly = week_rows["sigma_weekly"].to_numpy(dtype=float)
        sigma_weekly = np.nan_to_num(sigma_weekly, nan=0.0)

        k = solve_common_scale(x_raw, sigma_weekly, target_vol, max_gross)
        w_scaled = np.clip(k * x_raw, -max_gross_per_name, max_gross_per_name)

        n_active = max(1, int(np.sum(week_rows[signal_col].notna())))
        gross_this_week = float(np.sum(np.abs(w_scaled)))
        band = no_trade_band * (gross_this_week / n_active)

        proposed = {asset: w_scaled[j] for j, asset in enumerate(assets)}
        held_after = apply_no_trade_band(proposed, held, band)
        w_final = np.array([held_after[asset] for asset in assets])
        held.update(held_after)

        weight_col[idx] = w_final

    df["weight"] = weight_col
    return df
