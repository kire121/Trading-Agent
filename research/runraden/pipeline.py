"""End-to-end orchestration: load IS panel -> Steg 0 (K1a/b/c) -> Steg 1 (K2)
-> gated Steg 2 (OOS grid/DSR/K3/K4, only if IS survives).

Run as a script: `python pipeline.py` (writes a JSON report to
research/runraden/results/).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd

import config
import kill_criteria as kc
import twins as twins_mod
from additive_model import WalkForwardAdditiveModel
from eodhd_client import fetch_panel
from metrics import portfolio_weekly_returns, sharpe_ratio, max_drawdown
from positions import build_positions
from targets import attach_targets
from words import build_asset_week_panel


def load_is_panel(force_refresh: bool = False) -> tuple[dict, pd.DataFrame]:
    tickers = config.is_universe_flat()
    prices_raw = fetch_panel(tickers, start="1990-01-01", force_refresh=force_refresh)
    prices = {t: df["adjusted_close"].dropna() for t, df in prices_raw.items() if "adjusted_close" in df}
    panel = build_asset_week_panel(prices)
    panel = attach_targets(panel, prices, vol_window=config.PRIMARY_VOL_WINDOW)
    panel["execution_date"] = panel["next_first_date"]
    return prices, panel


def split_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_5d = panel[panel["table_id"] == "5d"].reset_index(drop=True)
    df_4d = panel[panel["table_id"] == "4d"].reset_index(drop=True)
    return df_5d, df_4d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (pd.Period,)):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def run_steg0_steg1(prices: dict, panel: pd.DataFrame, n_draws: int = 200,
                     cost_bp: float = config.ONE_WAY_COST_BP) -> dict:
    df_5d, df_4d = split_tables(panel)
    report: dict = {"n_rows_5d": len(df_5d), "n_rows_4d": len(df_4d), "n_rows_total": len(panel)}

    t0 = time.time()
    models, combined = kc.pooled_scored_panel(df_5d, df_4d, config.PRIMARY_KAPPA, config.BURN_IN_YEARS)
    report["fit_seconds"] = time.time() - t0
    report["n_scored_oof_rows"] = int(combined[["ghat", "z_next"]].dropna().shape[0])

    real_positions = build_positions(
        combined, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
        config.PRIMARY_NO_TRADE_BAND, signal_col="ghat",
    )
    additive_positions = build_positions(
        combined, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
        config.PRIMARY_NO_TRADE_BAND, signal_col="additive_pred",
    )

    t4_signal = twins_mod.market_timing_signal(
        combined, prices, config.PRIMARY_KAPPA, config.PRIMARY_VOL_WINDOW, config.BURN_IN_YEARS,
    )
    combined_t4 = combined.copy()
    combined_t4["t4_signal"] = t4_signal
    t4_positions = build_positions(
        combined_t4, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
        config.PRIMARY_NO_TRADE_BAND, signal_col="t4_signal",
    )

    combined_t2 = combined.copy()
    combined_t2["t2_signal"] = twins_mod.reversal_signal(combined_t2)
    t2_positions = build_positions(
        combined_t2, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
        config.PRIMARY_NO_TRADE_BAND, signal_col="t2_signal",
    )
    combined_t3 = combined.copy()
    combined_t3["t3_signal"] = twins_mod.tsmom_signal(combined_t3)
    t3_positions = build_positions(
        combined_t3, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
        config.PRIMARY_NO_TRADE_BAND, signal_col="t3_signal",
    )

    net_real = portfolio_weekly_returns(real_positions, cost_bp=cost_bp)
    net_t1 = portfolio_weekly_returns(additive_positions, cost_bp=cost_bp)
    net_t2 = portfolio_weekly_returns(t2_positions, cost_bp=cost_bp)
    net_t3 = portfolio_weekly_returns(t3_positions, cost_bp=cost_bp)
    net_t4 = portfolio_weekly_returns(t4_positions, cost_bp=cost_bp)

    report["descriptive_is_performance"] = {
        "runraden_full": {"sharpe": sharpe_ratio(net_real), "max_dd": max_drawdown(net_real),
                           "n_weeks": int(net_real.dropna().shape[0])},
        "T1_additive_only": {"sharpe": sharpe_ratio(net_t1), "max_dd": max_drawdown(net_t1)},
        "T2_reversal_1w": {"sharpe": sharpe_ratio(net_t2), "max_dd": max_drawdown(net_t2)},
        "T3_tsmom_1w": {"sharpe": sharpe_ratio(net_t3), "max_dd": max_drawdown(net_t3)},
        "T4_market_timing": {"sharpe": sharpe_ratio(net_t4), "max_dd": max_drawdown(net_t4)},
    }

    t0 = time.time()
    k1a = kc.k1a_redundancy_screen(df_5d, df_4d, config.PRIMARY_KAPPA, config.BURN_IN_YEARS,
                                    n_draws=n_draws)
    report["K1a_redundancy_screen"] = k1a
    report["k1a_seconds"] = time.time() - t0

    t0 = time.time()
    k1b = kc.k1b_common_factor_screen(combined, real_positions, t4_positions, cost_bp,
                                       n_draws=n_draws)
    report["K1b_common_factor_screen"] = k1b
    report["k1b_seconds"] = time.time() - t0

    t0 = time.time()
    k1c = kc.k1c_ubiquity(df_5d, df_4d, combined, real_positions, config.PRIMARY_KAPPA, cost_bp)
    report["K1c_ubiquity"] = k1c
    report["k1c_seconds"] = time.time() - t0

    k2 = kc.k2_incremental_value(combined)
    report["K2_incremental_value"] = k2

    step0_passed = bool(k1a["passed"] and k1b["passed"] and k1c["passed"])
    step1_passed = bool(k2["passed"])
    report["step0_passed"] = step0_passed
    report["step1_passed"] = step1_passed
    report["is_survives_to_step2"] = bool(step0_passed and step1_passed)
    return report


def main():
    print("[pipeline] Loading IS panel from EODHD ...")
    prices, panel = load_is_panel()
    print(f"[pipeline] Panel built: {len(panel)} asset-week rows, "
          f"{panel['asset'].nunique()} assets, "
          f"{panel['t_signal'].min()} .. {panel['t_signal'].max()}")

    report = run_steg0_steg1(prices, panel, n_draws=200)
    report["universe_n_tickers"] = len(prices)
    report["universe_tickers"] = sorted(prices.keys())

    out_path = f"{config.RESULTS_DIR}/steg0_steg1_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"[pipeline] Report written to {out_path}")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))},
                      indent=2, default=_json_default))
    return report


if __name__ == "__main__":
    main()
