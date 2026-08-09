"""End-to-end runner: fetch data, build signals, run the primary strategy
and all declared variants/null tests, write a results report.

Usage:
    python -m oglegrinden.run
"""

import json
import os

import numpy as np
import pandas as pd

from oglegrinden.universe import ALL_TICKERS, SECTOR_ETFS, BENCHMARK, MIN_ADV_USD
from oglegrinden.data import load_universe, Panel
from oglegrinden.signal import build_gate_bundle, weekly_fridays
from oglegrinden.backtest import run_backtest, always_on_baseline, benchmark_weekly_returns, adjusted_open, next_trading_day
from oglegrinden.portfolio import formation_weights
from oglegrinden.grid import run_grid, DEFAULT_CORR_WINDOW, DEFAULT_PERCENTILE, DEFAULT_FORMATION_DAYS
from oglegrinden.stats import (
    summary_stats,
    sharpe_ratio,
    deflated_sharpe_ratio,
    block_bootstrap_gate_test,
    gated_vs_always_on,
    beats_all_twins,
    twin_gate_regression,
    max_pnl_concentration,
    subperiod_sign_check,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OOS_START = pd.Timestamp("2018-01-01")
SUBPERIODS = [
    (pd.Timestamp("2000-01-01"), pd.Timestamp("2007-12-31")),
    (pd.Timestamp("2008-01-01"), pd.Timestamp("2012-12-31")),
    (pd.Timestamp("2013-01-01"), pd.Timestamp("2017-12-31")),
]


def _to_jsonable(obj):
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def cross_sectional_momentum_benchmark(panel: Panel, all_fridays: pd.DatetimeIndex, lookback=252, skip=21) -> pd.Series:
    """Simple 12-1 month cross-sectional momentum proxy (long recent
    winners, short recent losers) on the same ETF universe, used only as a
    diversification reference since no "Tidspilen" strategy exists in this
    codebase to compare against directly.
    """
    adj_open = adjusted_open(panel)
    trading_calendar = panel.close.index
    rets = []
    for i, t in enumerate(all_fridays):
        eligible = [tk for tk in panel.eligible_on(t, min_adv=MIN_ADV_USD) if tk != BENCHMARK]
        window = panel.log_returns.loc[:t, eligible].tail(lookback)
        if window.shape[0] < lookback:
            rets.append(0.0)
            continue
        mom = window.iloc[:-skip].sum(axis=0).dropna()
        if len(mom) < 6:
            rets.append(0.0)
            continue
        s = (mom - mom.mean())
        w = s / s.abs().sum()

        entry_date = next_trading_day(trading_calendar, t)
        next_t = all_fridays[i + 1] if i + 1 < len(all_fridays) else None
        exit_date = next_trading_day(trading_calendar, next_t) if next_t is not None else None
        if entry_date is None or exit_date is None:
            rets.append(0.0)
            continue
        entry_px = adj_open.loc[entry_date, w.index]
        exit_px = adj_open.loc[exit_date, w.index]
        valid = entry_px.notna() & exit_px.notna() & (entry_px > 0)
        w = w[valid]
        if w.empty:
            rets.append(0.0)
            continue
        simple_ret = exit_px[w.index] / entry_px[w.index] - 1.0
        rets.append(float((w * simple_ret).sum()))
    return pd.Series(rets, index=all_fridays, name="momentum_benchmark")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/9] Loading universe data (cached)...")
    histories = load_universe(ALL_TICKERS)
    panel = Panel(histories)
    print(f"  {len(panel.tickers)} tickers, {panel.close.index.min().date()} - {panel.close.index.max().date()}")

    all_fridays = weekly_fridays(panel.close.index)
    print(f"  {len(all_fridays)} weekly decision points")

    print("[2/9] Building primary (60d) gate bundle...")
    bundle60 = build_gate_bundle(panel, corr_window=DEFAULT_CORR_WINDOW)
    print(f"  {len(bundle60.raw)} weeks with a computable topology signal "
          f"({bundle60.raw.index.min().date()} - {bundle60.raw.index.max().date()})")

    print("[3/9] Running primary strategy + mirror + always-on baseline...")
    primary = run_backtest(panel, bundle60, all_fridays, signal_name="L", direction="primary",
                            formation_days=DEFAULT_FORMATION_DAYS)
    mirror = run_backtest(panel, bundle60, all_fridays, signal_name="L", direction="mirror",
                           formation_days=DEFAULT_FORMATION_DAYS)
    always_on = always_on_baseline(panel, all_fridays, formation_days=DEFAULT_FORMATION_DAYS)

    print("[4/9] Running twin gates (rho_bar, absorption_ratio, index_vol)...")
    twins = {}
    for sig in ["rho_bar", "absorption_ratio", "index_vol"]:
        twins[sig] = run_backtest(panel, bundle60, all_fridays, signal_name=sig, direction="primary",
                                   formation_days=DEFAULT_FORMATION_DAYS)

    print("[5/9] Running declared parameter grid (~30 variants)...")
    grid_run = run_grid(panel, all_fridays)
    grid_run.table.to_csv(os.path.join(RESULTS_DIR, "grid_table.csv"), index=False)

    print("[6/9] Computing DSR, block bootstrap, twin-gate regression, PnL concentration...")
    trial_sharpes_period = (grid_run.table["sharpe"] / np.sqrt(52)).values
    primary_sharpe_period = sharpe_ratio(primary.weekly_returns, annualize=False)
    dsr_full = deflated_sharpe_ratio(
        observed_sharpe_per_period=primary_sharpe_period,
        trial_sharpes_per_period=trial_sharpes_period,
        n_obs=len(primary.weekly_returns),
        skewness=summary_stats(primary.weekly_returns)["skew"],
        kurtosis=summary_stats(primary.weekly_returns)["kurtosis"],
    )

    oos_mask = primary.weekly_returns.index >= OOS_START
    primary_oos = primary.weekly_returns[oos_mask]
    grid_oos_sharpes = []
    for _id, ret in grid_run.returns_series.items():
        r_oos = ret[ret.index >= OOS_START]
        grid_oos_sharpes.append(sharpe_ratio(r_oos, annualize=False))
    dsr_oos = deflated_sharpe_ratio(
        observed_sharpe_per_period=sharpe_ratio(primary_oos, annualize=False),
        trial_sharpes_per_period=grid_oos_sharpes,
        n_obs=len(primary_oos),
        skewness=summary_stats(primary_oos)["skew"],
        kurtosis=summary_stats(primary_oos)["kurtosis"],
    )

    boot = block_bootstrap_gate_test(bundle60.gates[("L", "primary")], always_on.weekly_returns, n_boot=1000, block_size=13.0)

    beats_baseline = gated_vs_always_on(primary.weekly_returns, always_on.weekly_returns)

    reg = twin_gate_regression(
        always_on.weekly_returns,
        bundle60.smoothed["L"],
        bundle60.smoothed["rho_bar"],
        bundle60.smoothed["absorption_ratio"],
    )
    reg_no_model = {k: v for k, v in reg.items() if k != "model"}

    # Literal rule (c): "H1 maste sla alla tre [twins], annars tillfor
    # topologin inget" -- a direct Sharpe comparison of the fully
    # backtested strategies, distinct from (and in addition to) the
    # regression-control test above.
    twin_returns = {k: v.weekly_returns for k, v in twins.items()}
    twin_comparison_full = beats_all_twins(primary.weekly_returns, twin_returns)
    twin_comparison_oos = beats_all_twins(
        primary_oos,
        {k: v[v.index >= OOS_START] for k, v in twin_returns.items()},
    )

    concentration = max_pnl_concentration(primary.weekly_returns, window=8)

    subperiod = subperiod_sign_check(
        always_on.weekly_returns,
        bundle60.smoothed["L"],
        bundle60.smoothed["rho_bar"],
        bundle60.smoothed["absorption_ratio"],
        SUBPERIODS,
    )

    print("[7/9] Computing diversification stats (SPY beta, momentum-benchmark correlation)...")
    spy_returns = benchmark_weekly_returns(panel, all_fridays, benchmark=BENCHMARK)
    aligned = pd.DataFrame({"strategy": primary.weekly_returns, "spy": spy_returns}).dropna()
    spy_beta = float(np.cov(aligned["strategy"], aligned["spy"])[0, 1] / np.var(aligned["spy"])) if len(aligned) > 10 else float("nan")
    spy_corr = float(aligned["strategy"].corr(aligned["spy"]))

    mom_bench = cross_sectional_momentum_benchmark(panel, all_fridays)
    aligned_mom = pd.DataFrame({"strategy": primary.weekly_returns, "mom": mom_bench}).dropna()
    mom_corr = float(aligned_mom["strategy"].corr(aligned_mom["mom"])) if len(aligned_mom) > 10 else float("nan")

    print("[8/9] Assembling rejection-criteria verdicts...")
    rejection = {
        # Spec's rejection criterion "DSR <= 0" is read as the underlying
        # z-statistic (SR_hat - SR0) <= 0 -- the observed OOS Sharpe fails
        # to clear the noise ceiling implied by the grid's own dispersion.
        # The probability-scale DSR (dsr_oos["dsr"], a CDF value in [0,1])
        # is reported alongside for interpretability; z<=0 <=> dsr<=0.5.
        "dsr_oos_le_zero": dsr_oos["z"] <= 0.0,
        "dsr_oos_z": dsr_oos["z"],
        "dsr_oos_value": dsr_oos["dsr"],
        # Regression-control reading of rule (c): does b_L survive
        # controlling for rho_bar/absorption ratio?
        "twin_regression_insignificant": reg_no_model["p_L"] > 0.05,
        "twin_gate_p_L": reg_no_model["p_L"],
        # Literal reading of rule (c): does H1 beat all three fully
        # backtested twin gates on Sharpe? ("annars tillfor topologin
        # inget" -- if not, the topology signal adds nothing.)
        "h1_beats_all_twins_full_sample": twin_comparison_full["beats_all_twins"],
        "h1_beats_all_twins_oos": twin_comparison_oos["beats_all_twins"],
        "twin_explains_all": reg_no_model["p_L"] > 0.05 or not twin_comparison_full["beats_all_twins"],
        "pnl_concentration_gt_50pct": (concentration["share"] or 0) > 0.5,
        "pnl_concentration_share": concentration["share"],
        "subperiod_sign_flip": None,  # filled below
    }
    signs = [r.get("b_L") for r in subperiod if not r.get("insufficient_data")]
    if len(signs) >= 2:
        rejection["subperiod_sign_flip"] = not all((s > 0) == (signs[0] > 0) for s in signs)
    rejection["subperiods_with_data"] = sum(1 for r in subperiod if not r.get("insufficient_data"))

    always_on_stats = summary_stats(always_on.weekly_returns)
    primary_stats = summary_stats(primary.weekly_returns)
    mirror_stats = summary_stats(mirror.weekly_returns)
    twin_stats = {k: summary_stats(v.weekly_returns) for k, v in twins.items()}

    results = {
        "universe": {"n_tickers": len(SECTOR_ETFS), "tickers": SECTOR_ETFS, "benchmark": BENCHMARK,
                     "date_range": [str(panel.close.index.min().date()), str(panel.close.index.max().date())]},
        "primary": primary_stats,
        "mirror": mirror_stats,
        "always_on": always_on_stats,
        "twin_gates": twin_stats,
        "beats_always_on": beats_baseline["beats_both"],
        "dsr_full_sample": dsr_full,
        "dsr_oos": dsr_oos,
        "block_bootstrap": {k: v for k, v in boot.items() if k != "boot_sharpes"},
        "twin_gate_regression": reg_no_model,
        "twin_gate_literal_comparison": {"full_sample": twin_comparison_full, "oos": twin_comparison_oos},
        "pnl_concentration": concentration,
        "subperiod_sign_check": subperiod,
        "rejection_criteria": rejection,
        "diversification": {
            "spy_beta": spy_beta,
            "spy_corr": spy_corr,
            "momentum_benchmark_corr": mom_corr,
            "tidspilen_corr": None,
            "tidspilen_note": "No such strategy exists in this codebase; cannot be computed here.",
        },
        "grid": {"n_variants": len(grid_run.table)},
    }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, default=str)

    primary.weekly_returns.to_csv(os.path.join(RESULTS_DIR, "primary_weekly_returns.csv"))
    mirror.weekly_returns.to_csv(os.path.join(RESULTS_DIR, "mirror_weekly_returns.csv"))
    always_on.weekly_returns.to_csv(os.path.join(RESULTS_DIR, "always_on_weekly_returns.csv"))
    bundle60.raw.to_csv(os.path.join(RESULTS_DIR, "raw_signal_cw60.csv"))
    bundle60.smoothed.to_csv(os.path.join(RESULTS_DIR, "smoothed_signal_cw60.csv"))
    bundle60.percentile.to_csv(os.path.join(RESULTS_DIR, "percentile_signal_cw60.csv"))
    bundle60.gates[("L", "primary")].to_frame("gate_on").to_csv(os.path.join(RESULTS_DIR, "gate_state_primary.csv"))

    print("[9/9] Done. Results written to", RESULTS_DIR)
    return results, panel, bundle60, primary, mirror, always_on, twins, grid_run


if __name__ == "__main__":
    main()
