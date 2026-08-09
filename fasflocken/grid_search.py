"""
Fasflocken (PH-1) -- declared grid + DSR correction + rejection rule.

Grid: band {3-15, 5-20, 10-40}d x window {60, 90, 120}d x z-lookback
{52, 104, 156}w x legs {2, 3, 4} = 81 cells (config.BAND_GRID /
WINDOW_GRID / Z_LOOKBACK_GRID_WEEKS / LEGS_GRID). Hysteresis band is a
fixed rule (top/bottom 4), not part of the grid.

Performance note: R_s(t)/Corr_s(t) depend only on (band, window), not on
z-lookback or legs. run_grid therefore runs the expensive Hilbert
pipeline once per (band, window) pair (9x, not 81x) and reuses the daily
R/Corr series for the z-lookback x legs sub-grid (9 cheap cells each).

evaluate_rejection encodes the five rejection criteria from the spec
verbatim, run against the pre-registered default cell (config.DEFAULT_PARAMS)
using statistics gathered from the whole grid.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from fasflocken.config import (
    BAND_GRID,
    WINDOW_GRID,
    Z_LOOKBACK_GRID_WEEKS,
    LEGS_GRID,
    GICS_SECTORS,
    SignalParams,
    CostModel,
    DEFAULT_COSTS,
    VOL_TARGET_ANN,
    MAX_GROSS,
)
from fasflocken.pipeline import compute_sector_signal, zscore_from_R_daily
from fasflocken.backtest import run_backtest, annualized_sharpe, annualized_return, annualized_vol, max_drawdown
from fasflocken.universe import UniverseProvider


def declared_grid_cells(
    band_grid=BAND_GRID, window_grid=WINDOW_GRID, zlb_grid=Z_LOOKBACK_GRID_WEEKS, legs_grid=LEGS_GRID,
    hysteresis_band: int = 4,
) -> list[SignalParams]:
    cells = []
    for band, window, zlb, legs in product(band_grid, window_grid, zlb_grid, legs_grid):
        cells.append(
            SignalParams(
                band_low_days=band[0], band_high_days=band[1], filter_order=2,
                analytic_window=window, z_lookback_weeks=zlb, n_legs=legs, hysteresis_band=hysteresis_band,
            )
        )
    return cells


def _cell_id(p: SignalParams) -> str:
    return f"b{p.band_low_days}-{p.band_high_days}_w{p.analytic_window}_z{p.z_lookback_weeks}_l{p.n_legs}"


@dataclass
class GridResult:
    results: pd.DataFrame
    weekly_returns_by_cell: dict = field(repr=False)


def run_grid(
    provider: UniverseProvider,
    start: _dt.date,
    end: _dt.date,
    target_vol_ann: float = VOL_TARGET_ANN,
    max_gross: float = MAX_GROSS,
    cost_model: CostModel = DEFAULT_COSTS,
    sectors: tuple[str, ...] = GICS_SECTORS,
    hysteresis_band: int = 4,
    band_grid=BAND_GRID,
    window_grid=WINDOW_GRID,
    zlb_grid=Z_LOOKBACK_GRID_WEEKS,
    legs_grid=LEGS_GRID,
    progress: callable = None,
) -> GridResult:
    rows = []
    returns_by_cell: dict[str, pd.Series] = {}

    for band in band_grid:
        for window in window_grid:
            probe = SignalParams(
                band_low_days=band[0], band_high_days=band[1], analytic_window=window,
                z_lookback_weeks=zlb_grid[0], n_legs=legs_grid[0], hysteresis_band=hysteresis_band,
            )
            sigs = {s: compute_sector_signal(provider, s, start, end, probe) for s in sectors}

            for zlb in zlb_grid:
                z_panel = pd.DataFrame({s: zscore_from_R_daily(sigs[s].R_daily, zlb) for s in sectors}).sort_index()

                for legs in legs_grid:
                    cell_params = SignalParams(
                        band_low_days=band[0], band_high_days=band[1], analytic_window=window,
                        z_lookback_weeks=zlb, n_legs=legs, hysteresis_band=hysteresis_band,
                    )
                    bt = run_backtest(
                        provider, start, end, cell_params, target_vol_ann, max_gross,
                        cost_model=cost_model, sectors=sectors, z_override=z_panel,
                    )
                    cid = _cell_id(cell_params)
                    returns_by_cell[cid] = bt.weekly_returns
                    rows.append(
                        {
                            "cell_id": cid,
                            "band_low": band[0], "band_high": band[1], "window": window,
                            "z_lookback_weeks": zlb, "n_legs": legs,
                            "sharpe": annualized_sharpe(bt.weekly_returns),
                            "ann_return": annualized_return(bt.weekly_returns),
                            "ann_vol": annualized_vol(bt.weekly_returns),
                            "max_dd": max_drawdown(bt.weekly_returns),
                            "avg_turnover": float(bt.turnover.mean()) if len(bt.turnover) else float("nan"),
                            "n_obs": int(bt.weekly_returns.notna().sum()),
                        }
                    )
                    if progress is not None:
                        progress(rows[-1])

    return GridResult(results=pd.DataFrame(rows), weekly_returns_by_cell=returns_by_cell)


def neighborhood_isolation_check(
    grid_df: pd.DataFrame,
    target: SignalParams,
    sharpe_col: str = "sharpe",
    neighbor_frac_threshold: float = 0.3,
) -> dict:
    """Flags a result as an isolated gridcell fluke: the target cell is
    profitable but its one-dimension-away neighbors (same band/window/
    z-lookback/legs except one axis moved to an adjacent grid value)
    mostly aren't.
    """
    target_row = grid_df[
        (grid_df["band_low"] == target.band_low_days)
        & (grid_df["band_high"] == target.band_high_days)
        & (grid_df["window"] == target.analytic_window)
        & (grid_df["z_lookback_weeks"] == target.z_lookback_weeks)
        & (grid_df["n_legs"] == target.n_legs)
    ]
    if target_row.empty:
        return {"isolated": None, "reason": "target cell not found in grid"}
    target_sharpe = float(target_row[sharpe_col].iloc[0])

    def _differs_in_one_axis(row) -> bool:
        axes = [
            (row["band_low"], row["band_high"]) != (target.band_low_days, target.band_high_days),
            row["window"] != target.analytic_window,
            row["z_lookback_weeks"] != target.z_lookback_weeks,
            row["n_legs"] != target.n_legs,
        ]
        return sum(axes) == 1

    neighbors = grid_df[grid_df.apply(_differs_in_one_axis, axis=1)]
    if len(neighbors) == 0:
        return {"isolated": None, "reason": "no neighbors found"}

    frac_positive_neighbors = float((neighbors[sharpe_col] > 0).mean())
    isolated = bool(target_sharpe > 0 and frac_positive_neighbors < neighbor_frac_threshold)
    return {
        "isolated": isolated,
        "target_sharpe": target_sharpe,
        "n_neighbors": int(len(neighbors)),
        "frac_positive_neighbors": frac_positive_neighbors,
        "threshold": neighbor_frac_threshold,
    }


def evaluate_rejection(
    dsr_result: dict,
    bootstrap_p_value: float,
    delta_sharpe_vs_twin: float,
    sign_stability_result: dict,
    isolation_result: dict,
    p_value_threshold: float = 0.10,
    delta_sharpe_threshold: float = 0.15,
) -> dict:
    """The spec's five rejection criteria, each independently sufficient to
    kill the hypothesis. Returns per-criterion booleans plus the overall
    verdict (reject if ANY criterion fires).
    """
    dsr_gap = dsr_result.get("deflated_sharpe_gap", float("nan"))
    # NaN (e.g. a degenerate zero-vol return series) can't be shown to clear
    # the bar, so it's treated as a failure rather than silently passing.
    dsr_fails = bool(np.isnan(dsr_gap) or dsr_gap <= 0)
    reasons = {
        "dsr_leq_zero": dsr_fails,
        "bootstrap_p_geq_threshold": bool(
            not np.isnan(bootstrap_p_value) and bootstrap_p_value >= p_value_threshold
        ),
        "delta_sharpe_vs_twin_below_threshold": bool(
            np.isnan(delta_sharpe_vs_twin) or delta_sharpe_vs_twin < delta_sharpe_threshold
        ),
        "sign_unstable_across_subperiods": bool(not sign_stability_result.get("sign_stable", False)),
        "isolated_gridcell": bool(isolation_result.get("isolated") is True),
    }
    reject = bool(any(reasons.values()))
    return {"reject": reject, "reasons": reasons}
