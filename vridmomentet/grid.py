"""The declared neighborhood/robustness grid.

Brief: "Grannskap att testa: n in {10, 15, 20, 30, 40}, decilvariant,
tanh(R/sigma) i stallet for sign(R)." Full cross product: 5 windows x 2
bucket schemes (quintile/decile) x 2 direction transforms (sign/tanh) = 20
declared variants, feeding the deflated Sharpe ratio's trial-Sharpe pool
(grid.py, not the twins -- twins are separate, "known and uninteresting"
competitor signals the primary must beat outright, not alternate
re-specifications of the same signal being corrected for multiple testing).

Same amortized 3-tier-loop pattern as the sibling branches' grid searches
(Oglegrinden's grid.py, Fasflocken's grid_search.py): the expensive step
(the rolling Levy-area panel) depends only on `window` and is computed
once per window value (5x, not 20x); everything downstream of it (the
transform, the cross-sectional z-score, the bucket scheme, the backtest
itself) is cheap and re-run per grid cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from vridmomentet.backtest import BacktestResult, run_backtest
from vridmomentet.config import (
    BUCKET_GRID,
    TRANSFORM_GRID,
    WINDOW_GRID,
    CostModel,
    PortfolioParams,
    SignalParams,
)
from vridmomentet.data import Panel
from vridmomentet.signal import (
    SignalResult,
    cross_sectional_zscore,
    direction_factor,
    formation_return,
    rolling_levy_area,
    signed_dollar_volume,
)
from vridmomentet.stats import sharpe_ratio


def declared_grid_cells(
    window_grid: tuple = WINDOW_GRID, bucket_grid: tuple = BUCKET_GRID, transform_grid: tuple = TRANSFORM_GRID,
) -> list[dict]:
    cells = []
    for window, bucket, transform in product(window_grid, bucket_grid, transform_grid):
        n_buckets = 5 if bucket == "quintile" else 10
        cells.append({
            "cell_id": f"w{window}_{bucket}_{transform}",
            "window": window, "bucket": bucket, "n_buckets": n_buckets, "transform": transform,
        })
    return cells


@dataclass
class GridResult:
    table: pd.DataFrame
    weekly_returns_by_cell: dict = field(default_factory=dict, repr=False)
    backtests_by_cell: dict = field(default_factory=dict, repr=False)


def run_grid(
    panel: Panel,
    decision_dates: pd.DatetimeIndex,
    cost_model: CostModel = CostModel(),
    price_min: float = 5.0,
    adv_min: float = 20_000_000.0,
    window_grid: tuple = WINDOW_GRID,
    bucket_grid: tuple = BUCKET_GRID,
    transform_grid: tuple = TRANSFORM_GRID,
    winsor_lo: float = 0.01,
    winsor_hi: float = 0.99,
    tanh_scale_days: int = 20,
    execution: str = "monday_close",
) -> GridResult:
    u = signed_dollar_volume(panel)
    rows = []
    weekly_returns_by_cell: dict[str, pd.Series] = {}
    backtests_by_cell: dict[str, BacktestResult] = {}

    for window in window_grid:
        # Expensive step: once per window value.
        q = -rolling_levy_area(panel.log_returns, u, window)
        r_n = formation_return(panel.log_returns, window)
        z_q = cross_sectional_zscore(q, winsor_lo, winsor_hi)

        for transform in transform_grid:
            direction = direction_factor(panel.log_returns, window, transform, tanh_scale_days)
            s = z_q * direction
            signal = SignalResult(u=u, q=q, r_n=r_n, s=s)

            for bucket in bucket_grid:
                n_buckets = 5 if bucket == "quintile" else 10
                cell_id = f"w{window}_{bucket}_{transform}"
                params = PortfolioParams(n_buckets=n_buckets)
                bt = run_backtest(panel, signal, decision_dates, params, cost_model, execution, price_min, adv_min)

                weekly_returns_by_cell[cell_id] = bt.weekly_returns
                backtests_by_cell[cell_id] = bt
                stat = sharpe_ratio(bt.weekly_returns)
                rows.append({
                    "cell_id": cell_id, "window": window, "bucket": bucket, "transform": transform,
                    "sharpe": stat, "n_obs": int(bt.weekly_returns.notna().sum()),
                    "avg_turnover": float(bt.weekly_turnover.mean()) if len(bt.weekly_turnover) else float("nan"),
                    "is_primary": (window == SignalParams().window_days and bucket == "quintile" and transform == "sign"),
                })

    table = pd.DataFrame(rows)
    return GridResult(table=table, weekly_returns_by_cell=weekly_returns_by_cell, backtests_by_cell=backtests_by_cell)


def neighborhood_isolation_check(grid_df: pd.DataFrame, target_cell_id: str, sharpe_col: str = "sharpe", neighbor_frac_threshold: float = 0.3) -> dict:
    """Flags a lone-winner fluke: the target cell is profitable but its
    one-axis-away neighbors (same window/bucket/transform except one moved)
    mostly aren't.
    """
    target_row = grid_df[grid_df["cell_id"] == target_cell_id]
    if target_row.empty:
        return {"isolated": None, "reason": "target cell not found"}
    target = target_row.iloc[0]
    target_sharpe = float(target[sharpe_col])

    def differs_in_one_axis(row) -> bool:
        axes_diff = [row["window"] != target["window"], row["bucket"] != target["bucket"], row["transform"] != target["transform"]]
        return sum(axes_diff) == 1

    neighbors = grid_df[grid_df.apply(differs_in_one_axis, axis=1)]
    frac_positive = float((neighbors[sharpe_col] > 0).mean()) if len(neighbors) else float("nan")
    isolated = bool(target_sharpe > 0 and not np.isnan(frac_positive) and frac_positive < neighbor_frac_threshold)
    return {
        "isolated": isolated, "target_sharpe": target_sharpe, "n_neighbors": len(neighbors),
        "frac_positive_neighbors": frac_positive, "threshold": neighbor_frac_threshold,
    }
