"""Declared parameter grid for the deflated Sharpe ratio test.

Grid: corr_window {60,120} x gate_percentile {50,60,70} x formation {5d,10d}
x direction {primary,mirror} = 24 base variants, plus the two declared
extra variants: SPY beta-hedge of residual beta (crossed by direction,
+2) and correlation on SPY-residual returns (crossed by corr_window and
direction, +4) for N=30 total, matching the brief's "N ~ 30". Everything
here is trial-generation + scoring; DSR itself is computed in stats.py
from the resulting Sharpe ratios.
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from oglegrinden.data import Panel
from oglegrinden.signal import build_gate_bundle, regate
from oglegrinden.backtest import run_backtest, hedge_to_beta, benchmark_weekly_returns
from oglegrinden.universe import BENCHMARK
from oglegrinden.stats import summary_stats

GATE_PERCENTILES = [50, 60, 70]
CORR_WINDOWS = [60, 120]
FORMATION_DAYS = [5, 10]
DIRECTIONS = ["primary", "mirror"]

DEFAULT_CORR_WINDOW = 60
DEFAULT_PERCENTILE = 60.0
DEFAULT_FORMATION_DAYS = 5


def build_grid_spec() -> List[dict]:
    variants = []
    for cw in CORR_WINDOWS:
        for pct in GATE_PERCENTILES:
            for fd in FORMATION_DAYS:
                for direction in DIRECTIONS:
                    variants.append(
                        {
                            "id": f"base_cw{cw}_p{pct}_f{fd}_{direction}",
                            "kind": "base",
                            "corr_window": cw,
                            "upper": float(pct),
                            "lower": float(pct - 20),
                            "formation_days": fd,
                            "direction": direction,
                        }
                    )
    for direction in DIRECTIONS:
        variants.append(
            {
                "id": f"beta_hedge_{direction}",
                "kind": "beta_hedge",
                "corr_window": DEFAULT_CORR_WINDOW,
                "upper": DEFAULT_PERCENTILE,
                "lower": DEFAULT_PERCENTILE - 20,
                "formation_days": DEFAULT_FORMATION_DAYS,
                "direction": direction,
            }
        )
    for cw in CORR_WINDOWS:
        for direction in DIRECTIONS:
            variants.append(
                {
                    "id": f"residual_corr_cw{cw}_{direction}",
                    "kind": "residual_corr",
                    "corr_window": cw,
                    "upper": DEFAULT_PERCENTILE,
                    "lower": DEFAULT_PERCENTILE - 20,
                    "formation_days": DEFAULT_FORMATION_DAYS,
                    "direction": direction,
                }
            )
    return variants


@dataclass
class GridRun:
    spec: List[dict]
    results: dict = field(default_factory=dict)  # id -> BacktestResult
    returns_series: dict = field(default_factory=dict)  # id -> pd.Series (post beta-hedge where applicable)
    table: pd.DataFrame = None


def run_grid(
    panel: Panel,
    all_fridays: pd.DatetimeIndex,
    min_history_years: float = 3.0,
    cost_kwargs: dict = None,
) -> GridRun:
    cost_kwargs = cost_kwargs or {}
    spec = build_grid_spec()

    # Precompute the expensive per-corr-window bundles once each: raw
    # (non-residualized) and residualized, reused across every threshold
    # variant that shares a corr_window via cheap `regate`.
    raw_bundles = {
        cw: build_gate_bundle(panel, corr_window=cw, min_history_years=min_history_years)
        for cw in CORR_WINDOWS
    }
    residual_bundles = {
        cw: build_gate_bundle(panel, corr_window=cw, min_history_years=min_history_years, residualize=True)
        for cw in CORR_WINDOWS
    }

    spy_returns = benchmark_weekly_returns(panel, all_fridays, benchmark=BENCHMARK)

    results = {}
    returns_series = {}
    rows = []
    for v in spec:
        base_bundle = residual_bundles[v["corr_window"]] if v["kind"] == "residual_corr" else raw_bundles[v["corr_window"]]
        bundle = regate(base_bundle, upper=v["upper"], lower=v["lower"])

        bt = run_backtest(
            panel,
            bundle,
            all_fridays,
            signal_name="L",
            direction=v["direction"],
            formation_days=v["formation_days"],
            **cost_kwargs,
        )

        returns = bt.weekly_returns
        if v["kind"] == "beta_hedge":
            returns = hedge_to_beta(returns, spy_returns)

        stats = summary_stats(returns)
        results[v["id"]] = bt
        returns_series[v["id"]] = returns
        rows.append({**v, "returns_key": v["id"], **stats})

    table = pd.DataFrame(rows)
    return GridRun(spec=spec, results=results, returns_series=returns_series, table=table)
