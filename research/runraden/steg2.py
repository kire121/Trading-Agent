"""Steg 2: OOS grid (27 cells), oracle cap, DSR, K3 (cost floor / twin
comparison) and K4 (sign flip / isolated-cell) kill criteria.

This module is only ever invoked against the OOS UCITS panel, and only if
Steg 0 (K1a/b/c) and Steg 1 (K2) both pass on the IS panel -- see
pipeline.run_steg2 and the README "Execution log" for why, in this run, it
was implemented and unit-tested but not executed against real OOS data.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config
from additive_model import WalkForwardAdditiveModel
from kill_criteria import pooled_scored_panel
from metrics import deflated_sharpe_ratio, portfolio_weekly_returns, sharpe_ratio
from positions import build_positions
from targets import attach_targets
from words import build_asset_week_panel


def build_oos_panel(prices: dict, vol_window: int) -> pd.DataFrame:
    panel = build_asset_week_panel(prices)
    panel = attach_targets(panel, prices, vol_window=vol_window)
    panel["execution_date"] = panel["next_first_date"]
    return panel


def oracle_positions(scored_panel: pd.DataFrame, target_vol: float, max_gross: float,
                      max_gross_per_name: float) -> pd.DataFrame:
    """Perfect-foresight upper bound: signal = sign(next_week_return), i.e.
    the largest Sharpe achievable by *any* directional weekly signal under
    the same risk-management pipeline (no no-trade band, since a perfect
    oracle has no reason to ever refuse a profitable trade)."""
    df = scored_panel.copy()
    df["oracle_signal"] = np.sign(df["next_week_return"]).replace(0, 1.0)
    return build_positions(df, target_vol, max_gross, max_gross_per_name, no_trade_band=0.0,
                            signal_col="oracle_signal")


def run_grid_cell(df_5d: pd.DataFrame, df_4d: pd.DataFrame, kappa: float, vol_window: int,
                   no_trade_band: float, burn_in_years: int, cost_bp: float) -> dict:
    # vol_window affects targets.attach_targets (sigma_60d), which callers
    # must have already recomputed for this cell before calling here; this
    # function assumes df_5d/df_4d already carry the sigma for `vol_window`.
    _, combined = pooled_scored_panel(df_5d, df_4d, kappa, burn_in_years)
    positions = build_positions(combined, config.TARGET_GROSS_VOL, config.MAX_GROSS,
                                 config.MAX_GROSS_PER_NAME, no_trade_band, signal_col="ghat")
    net = portfolio_weekly_returns(positions, cost_bp=cost_bp)
    sr = sharpe_ratio(net)
    return {"kappa": kappa, "vol_window": vol_window, "no_trade_band": no_trade_band,
            "sharpe": sr, "n_weeks": int(net.dropna().shape[0]), "net_returns": net,
            "positions": positions}


def run_grid(prices: dict, burn_in_years: int, cost_bp: float) -> list[dict]:
    """Grid27: kappa x vol_window x no_trade_band. Re-attaches targets (and
    therefore sigma_60d) for each distinct vol_window since that changes
    both the regression target and the position-sizing denominator."""
    results = []
    panels_by_vol_window: dict[int, pd.DataFrame] = {}
    for vw in config.GRID_VOL_WINDOW:
        panels_by_vol_window[vw] = build_oos_panel(prices, vw)

    for kappa, vw, band in itertools.product(config.GRID_KAPPA, config.GRID_VOL_WINDOW,
                                              config.GRID_NO_TRADE_BAND):
        panel = panels_by_vol_window[vw]
        df_5d = panel[panel["table_id"] == "5d"].reset_index(drop=True)
        df_4d = panel[panel["table_id"] == "4d"].reset_index(drop=True)
        cell = run_grid_cell(df_5d, df_4d, kappa, vw, band, burn_in_years, cost_bp)
        results.append(cell)
    return results


def k3_cost_and_twin_check(primary_cell: dict, twin_sharpes: dict[str, float],
                            prices: dict, burn_in_years: int) -> dict:
    """Dead if the primary cell loses (net Sharpe) to any of T1-T3 at matched
    vol target, or DSR <= 0, or the sign flips under 1.5x cost stress."""
    stressed = run_grid_cell(
        *_tables_for_cell(prices, primary_cell["vol_window"], burn_in_years),
        kappa=primary_cell["kappa"], vol_window=primary_cell["vol_window"],
        no_trade_band=primary_cell["no_trade_band"], burn_in_years=burn_in_years,
        cost_bp=config.ONE_WAY_COST_BP * config.COST_STRESS_MULTIPLIER,
    )
    dsr = deflated_sharpe_ratio(primary_cell["net_returns"], n_trials=config.DSR_N_TRIALS)

    loses_to_a_twin = any(
        (not np.isnan(primary_cell["sharpe"])) and (not np.isnan(s)) and primary_cell["sharpe"] <= s
        for s in twin_sharpes.values()
    )
    sign_flip_under_stress = bool(
        np.sign(primary_cell["sharpe"]) != np.sign(stressed["sharpe"])
        if not (np.isnan(primary_cell["sharpe"]) or np.isnan(stressed["sharpe"])) else True
    )
    dead = bool(loses_to_a_twin or (not np.isnan(dsr["dsr"]) and dsr["dsr"] <= 0) or sign_flip_under_stress)
    return {
        "dsr": dsr, "stressed_sharpe": stressed["sharpe"], "loses_to_a_twin": loses_to_a_twin,
        "sign_flip_under_stress": sign_flip_under_stress, "passed": not dead,
    }


def _tables_for_cell(prices: dict, vol_window: int, burn_in_years: int):
    panel = build_oos_panel(prices, vol_window)
    df_5d = panel[panel["table_id"] == "5d"].reset_index(drop=True)
    df_4d = panel[panel["table_id"] == "4d"].reset_index(drop=True)
    return df_5d, df_4d


def k4_sign_flip_and_isolation_check(grid_results: list[dict], primary_key: tuple) -> dict:
    """Dead if the primary cell's sign flips between the first and second
    half of the OOS sample, or if the grid's edge is isolated to a single
    best cell (neighbouring cells in the grid show no sign agreement)."""
    primary = next(r for r in grid_results
                    if (r["kappa"], r["vol_window"], r["no_trade_band"]) == primary_key)
    net = primary["net_returns"].dropna()
    if len(net) < 20:
        return {"passed": False, "reason": "insufficient OOS observations"}
    mid = len(net) // 2
    first_half_sr = sharpe_ratio(net.iloc[:mid])
    second_half_sr = sharpe_ratio(net.iloc[mid:])
    sign_flip = bool(
        not np.isnan(first_half_sr) and not np.isnan(second_half_sr)
        and np.sign(first_half_sr) != np.sign(second_half_sr)
    )

    sharpes = np.array([r["sharpe"] for r in grid_results if not np.isnan(r["sharpe"])])
    primary_sign = np.sign(primary["sharpe"]) if not np.isnan(primary["sharpe"]) else 0
    same_sign_frac = float(np.mean(np.sign(sharpes) == primary_sign)) if len(sharpes) else 0.0
    isolated = bool(same_sign_frac < 0.5)  # edge not even directionally shared by most of the grid

    passed = bool(not sign_flip and not isolated)
    return {
        "sign_flip": sign_flip, "first_half_sharpe": first_half_sr,
        "second_half_sharpe": second_half_sr, "same_sign_fraction_of_grid": same_sign_frac,
        "isolated": isolated, "passed": passed,
    }
