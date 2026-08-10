from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vridmomentet.config import CostModel
from vridmomentet.data import Panel
from vridmomentet.grid import declared_grid_cells, neighborhood_isolation_check, run_grid
from vridmomentet.universe import MembershipInterval, PointInTimeMembership


class TestDeclaredGridCells:
    def test_cross_product_size(self):
        cells = declared_grid_cells(window_grid=(10, 20), bucket_grid=("quintile", "decile"), transform_grid=("sign", "tanh"))
        assert len(cells) == 8

    def test_cell_ids_are_unique(self):
        cells = declared_grid_cells()
        ids = [c["cell_id"] for c in cells]
        assert len(ids) == len(set(ids))

    def test_decile_maps_to_10_buckets(self):
        cells = declared_grid_cells(window_grid=(20,), bucket_grid=("decile",), transform_grid=("sign",))
        assert cells[0]["n_buckets"] == 10


def _synthetic_panel(n_days=200, n_names=60, seed=0) -> tuple[Panel, pd.DatetimeIndex]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    names = [f"T{i}" for i in range(n_names)]
    close = pd.DataFrame({n: 100 * np.cumprod(1 + rng.normal(scale=0.015, size=n_days)) for n in names}, index=dates)
    volume = pd.DataFrame({n: rng.uniform(2e6, 5e6, size=n_days) for n in names}, index=dates)
    start, end = dates.min().date(), dates.max().date() + dt.timedelta(days=1)
    membership = PointInTimeMembership(
        [MembershipInterval(ticker=t, source="sp500", start=start, end=end, point_in_time=True) for t in names]
    )
    panel = Panel(close=close, adj_close=close.copy(), adj_open=close.copy(), volume=volume, membership=membership)
    from vridmomentet.backtest import weekly_decision_dates
    decisions = weekly_decision_dates(dates)
    return panel, decisions


class TestRunGrid:
    def test_produces_one_row_per_declared_cell(self):
        panel, decisions = _synthetic_panel()
        result = run_grid(
            panel, decisions, cost_model=CostModel(0, 0), price_min=0, adv_min=0,
            window_grid=(10, 20), bucket_grid=("quintile", "decile"), transform_grid=("sign", "tanh"),
        )
        assert len(result.table) == 8
        assert set(result.weekly_returns_by_cell.keys()) == set(result.table["cell_id"])

    def test_exactly_one_cell_marked_primary(self):
        panel, decisions = _synthetic_panel()
        result = run_grid(
            panel, decisions, cost_model=CostModel(0, 0), price_min=0, adv_min=0,
            window_grid=(10, 20), bucket_grid=("quintile", "decile"), transform_grid=("sign", "tanh"),
        )
        assert result.table["is_primary"].sum() == 1
        primary = result.table[result.table["is_primary"]].iloc[0]
        assert primary["window"] == 20 and primary["bucket"] == "quintile" and primary["transform"] == "sign"


class TestNeighborhoodIsolationCheck:
    def test_isolated_winner_is_flagged(self):
        grid_df = pd.DataFrame([
            {"cell_id": "target", "window": 20, "bucket": "quintile", "transform": "sign", "sharpe": 1.0},
            {"cell_id": "n1", "window": 10, "bucket": "quintile", "transform": "sign", "sharpe": -0.5},
            {"cell_id": "n2", "window": 30, "bucket": "quintile", "transform": "sign", "sharpe": -0.3},
            {"cell_id": "n3", "window": 20, "bucket": "decile", "transform": "sign", "sharpe": -0.2},
            {"cell_id": "n4", "window": 20, "bucket": "quintile", "transform": "tanh", "sharpe": -0.4},
        ])
        result = neighborhood_isolation_check(grid_df, "target")
        assert result["isolated"] is True

    def test_robust_winner_is_not_flagged(self):
        grid_df = pd.DataFrame([
            {"cell_id": "target", "window": 20, "bucket": "quintile", "transform": "sign", "sharpe": 1.0},
            {"cell_id": "n1", "window": 10, "bucket": "quintile", "transform": "sign", "sharpe": 0.8},
            {"cell_id": "n2", "window": 30, "bucket": "quintile", "transform": "sign", "sharpe": 0.6},
            {"cell_id": "n3", "window": 20, "bucket": "decile", "transform": "sign", "sharpe": 0.7},
            {"cell_id": "n4", "window": 20, "bucket": "quintile", "transform": "tanh", "sharpe": 0.5},
        ])
        result = neighborhood_isolation_check(grid_df, "target")
        assert result["isolated"] is False
