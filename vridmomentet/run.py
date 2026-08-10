"""End-to-end runner: fetch data, run the three-stage rocket, write
results/results.json.

The brief's own text names "Oglegrindens protokoll" but is cut off before
specifying the three stages for Vridmomentet itself ("Backtestskiss
(trestegsraket per Oglegrindens protokoll)" is the brief's last sentence).
Oglegrinden's own run.py, on inspection, is actually a single linear
9-step pipeline that always runs to completion -- it does not implement a
runtime early-exit gate. The one place in this whole research program a
"cheap stage 1 that can kill the idea before expensive stages run" idea is
actually stated is the brief's OWN "Forvantad svaghet" section: "Daglig
upplosning for grov ... da ar IC ~ 0 och ideen dor billigt i steg 1."

We take that at face value and implement a genuine, load-bearing stage 1:

  Stage 1 (cheap): compute the primary signal; run the pre-registered
    shuffle-null check (estimator-level) and the IC-vs-forward-return
    check (the literal "IC ~ 0" criterion). No portfolio construction, no
    costs, no walk-forward loop.
  Stage 2 (moderate): full costed weekly walk-forward backtest of the
    primary strategy (both execution conventions) and all three active
    twins; benchmark/diversification stats.
  Stage 3 (expensive): the 20-cell neighborhood grid, deflated Sharpe
    ratio, block bootstrap of the primary weekly returns, the PEAD-proxy
    exclusion re-backtest, sub-period sign stability, PnL concentration,
    neighborhood isolation, and the assembled rejection verdict.

Matching this repository's house convention (Oglegrinden, Fasflocken,
irreversibility_lab all run every stage to completion and report negative
results as a valid, primary outcome, not an early abort): by default all
three stages always run, in order, regardless of stage 1's outcome. Pass
--fast-exit to genuinely stop after a stage-1 kill, for cheap iteration
during development; the committed results/ output is always a full run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time

import numpy as np
import pandas as pd

from vridmomentet import config, grid, stats
from vridmomentet.backtest import BacktestResult, benchmark_weekly_returns, run_backtest, weekly_decision_dates
from vridmomentet.data import Panel, build_panel
from vridmomentet.portfolio import PortfolioParams
from vridmomentet.signal import SignalParams, compute_signal
from vridmomentet.twins import compute_all_twins
from vridmomentet.universe import EODHDProvider, UniverseProvider

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

_t_start = time.time()
_log_lines: list[str] = []


def _log(msg: str) -> None:
    elapsed = time.time() - _t_start
    line = f"[{elapsed:7.1f}s] {msg}"
    print(line, flush=True)
    _log_lines.append(line)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp, _dt.date, _dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vridmomentet: Levy-area-ranked price/volume rotation")
    p.add_argument("--start", type=lambda s: _dt.date.fromisoformat(s), default=config.HISTORY_START)
    p.add_argument("--end", type=lambda s: _dt.date.fromisoformat(s), default=_dt.date.today())
    p.add_argument("--universe-cap", type=int, default=None, help="cap the universe to the first N tickers (fast iteration)")
    p.add_argument("--sp400", action="store_true", default=True, help="include the (non-point-in-time) S&P 400 leg")
    p.add_argument("--no-sp400", dest="sp400", action="store_false")
    p.add_argument("--fast-exit", action="store_true", help="stop after stage 1 if its kill criterion fires")
    p.add_argument("--skip-grid", action="store_true", help="skip the 20-cell neighborhood grid (fast iteration)")
    p.add_argument("--bootstrap-reps", type=int, default=config.N_BOOTSTRAP_DRAWS)
    p.add_argument("--shuffle-windows", type=int, default=300, help="how many (ticker,date) windows to sample for the stage-1 shuffle null")
    p.add_argument("--output-dir", type=str, default=RESULTS_DIR)
    return p


def fetch_universe_panel(provider: UniverseProvider, start: _dt.date, end: _dt.date, sp400: bool, universe_cap: int | None) -> tuple[Panel, pd.Series]:
    membership = provider.membership()
    tickers = membership.all_tickers()
    if not sp400:
        sp500_only = membership.tickers_from_source("sp500")
        tickers = [t for t in tickers if t in sp500_only]
    tickers = sorted(tickers)
    if universe_cap:
        tickers = tickers[:universe_cap]
    _log(f"fetching {len(tickers)} tickers + SPY benchmark ({start} .. {end})...")
    panel = build_panel(provider, tickers, start, end)
    _log(f"panel built: {panel.close.shape[1]} names x {panel.close.shape[0]} trading days")

    spy_prices = provider.prices(["SPY"], start, end)
    spy_close = spy_prices["SPY"]["adjusted_close"] if "SPY" in spy_prices else pd.Series(dtype=float)
    return panel, spy_close


def run_stage1(panel: Panel, signal_params: SignalParams, n_shuffle_windows: int, n_shuffle_reps: int) -> dict:
    _log("STAGE 1: computing primary signal + pre-registered null checks...")
    signal = compute_signal(panel, signal_params)

    decisions = weekly_decision_dates(panel.dates)
    fwd_return = panel.log_returns.rolling(signal_params.forward_horizon_days, min_periods=signal_params.forward_horizon_days).sum().shift(-signal_params.forward_horizon_days)

    ic = stats.ic_stage1_check(signal.s, fwd_return, decisions)
    _log(f"  mean weekly IC = {ic['mean_ic']:.4f}, t = {ic['t_stat']:.2f}, n_weeks = {ic['n_weeks']}")

    shuffle = stats.shuffle_null_check(panel, signal_params.window_days, n_shuffle_windows, n_shuffle_reps, config.SHUFFLE_BLOCK_SIZES)
    for bs, res in shuffle["by_block_size"].items():
        _log(f"  shuffle null (block={bs}): {res['fraction_exceeding_95th_pct_null']:.1%} of sampled windows exceed their own null's 95th pct (n={res['n_windows_tested']})")

    ic_t = ic["t_stat"]
    kill = bool(np.isnan(ic_t) or abs(ic_t) < config.DEFAULT_REJECTION.ic_t_stat_min)
    verdict = "STAGE 1 KILL: |t(IC)| < threshold, consistent with brief's own predicted failure mode (IC ~ 0)" if kill else "STAGE 1 PASS: IC distinguishable from zero, proceeding"
    _log(f"  {verdict}")

    return {"signal": signal, "decision_dates": decisions, "forward_return": fwd_return, "ic": ic, "shuffle_null": shuffle, "stage1_kill": kill}


def run_stage2(panel: Panel, spy_close: pd.Series, signal, decisions: pd.DatetimeIndex, portfolio_params: PortfolioParams, cost_model, twin_params) -> dict:
    _log("STAGE 2: full costed backtest (primary, execution variant, twins, benchmark)...")

    primary_bt = run_backtest(panel, signal, decisions, portfolio_params, cost_model, "monday_close", config.PRICE_MIN_USD, config.ADV_MIN_USD)
    _log(f"  primary (monday_close): Sharpe={stats.sharpe_ratio(primary_bt.weekly_returns):.3f}, n_weeks={len(primary_bt.weekly_returns)}")

    variant_bt = run_backtest(panel, signal, decisions, portfolio_params, cost_model, "monday_open", config.PRICE_MIN_USD, config.ADV_MIN_USD)
    _log(f"  variant (monday_open): Sharpe={stats.sharpe_ratio(variant_bt.weekly_returns):.3f}")

    twin_signals = compute_all_twins(panel, twin_params)
    twin_backtests: dict[str, BacktestResult] = {}
    for name, twin_s in twin_signals.items():
        from vridmomentet.signal import SignalResult
        twin_signal = SignalResult(u=None, q=None, r_n=None, s=twin_s)
        bt = run_backtest(panel, twin_signal, decisions, portfolio_params, cost_model, "monday_close", config.PRICE_MIN_USD, config.ADV_MIN_USD)
        twin_backtests[name] = bt
        _log(f"  twin[{name}]: Sharpe={stats.sharpe_ratio(bt.weekly_returns):.3f}")

    spy_returns = benchmark_weekly_returns(spy_close, panel.dates, decisions) if len(spy_close) else pd.Series(dtype=float)
    trend_proxy_returns = twin_backtests["momentum"].weekly_returns if "momentum" in twin_backtests else None
    div = stats.diversification_stats(primary_bt.weekly_returns, spy_returns, trend_proxy_returns)
    _log(f"  beta to SPY = {div['beta_to_benchmark']:.3f}, corr to momentum-proxy = {div['corr_to_trend_proxy']:.3f}, corr to Tidspilen = N/A (no such strategy exists in this codebase)")

    twins_comparison = stats.beats_all_twins(primary_bt.weekly_returns, {k: v.weekly_returns for k, v in twin_backtests.items()})
    _log(f"  beats_all_twins = {twins_comparison['beats_all_twins']}")

    return {
        "primary_bt": primary_bt, "variant_bt": variant_bt, "twin_backtests": twin_backtests,
        "spy_returns": spy_returns, "diversification": div, "twins_comparison": twins_comparison,
    }


def run_stage3(panel: Panel, signal, decisions, primary_bt: BacktestResult, args, cost_model) -> dict:
    _log("STAGE 3: robustness (grid/DSR/bootstrap/PEAD/sign-stability/concentration)...")

    if args.skip_grid:
        _log("  --skip-grid set, skipping the neighborhood grid and DSR")
        grid_result, dsr_full, dsr_oos, isolation = None, {}, {}, {}
    else:
        grid_result = grid.run_grid(panel, decisions, cost_model, config.PRICE_MIN_USD, config.ADV_MIN_USD)
        _log(f"  grid: {len(grid_result.table)} cells run")

        trial_sharpes_period_full = (grid_result.table["sharpe"] / np.sqrt(config.WEEKS_PER_YEAR)).tolist()
        primary_period_sharpe_full = stats.sharpe_ratio(primary_bt.weekly_returns, annualize=False)
        dsr_full = stats.deflated_sharpe_ratio(primary_period_sharpe_full, trial_sharpes_period_full, n_obs=len(primary_bt.weekly_returns))
        _log(f"  DSR (full sample): z={dsr_full['z']:.2f}, dsr={dsr_full['dsr']:.4f}")

        oos_returns = primary_bt.weekly_returns[primary_bt.weekly_returns.index >= pd.Timestamp(config.OOS_START)]
        primary_period_sharpe_oos = stats.sharpe_ratio(oos_returns, annualize=False)
        trial_sharpes_period_oos = [sharpe_of_period(r, config.OOS_START) for r in grid_result.weekly_returns_by_cell.values()]
        dsr_oos = stats.deflated_sharpe_ratio(primary_period_sharpe_oos, trial_sharpes_period_oos, n_obs=len(oos_returns))
        _log(f"  DSR (OOS): z={dsr_oos['z']:.2f}, dsr={dsr_oos['dsr']:.4f}")

        primary_cell_id = grid_result.table.loc[grid_result.table["is_primary"], "cell_id"].iloc[0]
        isolation = grid.neighborhood_isolation_check(grid_result.table, primary_cell_id)
        _log(f"  neighborhood isolation: isolated={isolation['isolated']}, frac_positive_neighbors={isolation.get('frac_positive_neighbors')}")

    bootstrap = stats.block_bootstrap_sharpe_pvalue(primary_bt.weekly_returns, n_boot=args.bootstrap_reps, block_size=config.BOOTSTRAP_BLOCK_WEEKS)
    _log(f"  block bootstrap: p={bootstrap['p_value']:.4f}")

    pead_mask = stats.pead_exclusion_mask(panel, window=config.DEFAULT_SIGNAL_PARAMS.window_days)
    s_excl = stats.apply_exclusion_mask(signal.s, pead_mask)
    from vridmomentet.signal import SignalResult
    pead_signal = SignalResult(u=signal.u, q=signal.q, r_n=signal.r_n, s=s_excl)
    pead_bt = run_backtest(panel, pead_signal, decisions, PortfolioParams(), cost_model, "monday_close", config.PRICE_MIN_USD, config.ADV_MIN_USD)
    pead_delta_sharpe = stats.sharpe_ratio(pead_bt.weekly_returns) - stats.sharpe_ratio(primary_bt.weekly_returns)
    _log(f"  PEAD-exclusion Sharpe delta: {pead_delta_sharpe:+.3f} (excl. Sharpe={stats.sharpe_ratio(pead_bt.weekly_returns):.3f})")

    concentration = stats.max_pnl_concentration(primary_bt.weekly_returns, window=config.PNL_CONCENTRATION_WINDOW_WEEKS)
    _log(f"  PnL concentration (best {config.PNL_CONCENTRATION_WINDOW_WEEKS}wk window): {concentration['share']:.1%}")

    sign = stats.sign_stability(primary_bt.weekly_returns, config.SUBPERIODS)
    _log(f"  sub-period sign stability: {sign['sign_stable']}")

    return {
        "grid_result": grid_result, "dsr_full": dsr_full, "dsr_oos": dsr_oos, "isolation": isolation,
        "bootstrap": bootstrap, "pead_bt": pead_bt, "pead_delta_sharpe": pead_delta_sharpe,
        "concentration": concentration, "sign_stability": sign,
    }


def sharpe_of_period(returns: pd.Series, since: _dt.date) -> float:
    sub = returns[returns.index >= pd.Timestamp(since)]
    return stats.sharpe_ratio(sub, annualize=False)


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    provider = EODHDProvider()
    panel, spy_close = fetch_universe_panel(provider, args.start, args.end, args.sp400, args.universe_cap)

    signal_params = config.DEFAULT_SIGNAL_PARAMS
    s1 = run_stage1(panel, signal_params, args.shuffle_windows, config.N_SHUFFLE_REPS)

    if args.fast_exit and s1["stage1_kill"]:
        _log("--fast-exit set and stage 1 kill criterion fired: stopping here.")
        report = {"stage1": _strip_heavy(s1), "verdict": "STAGE_1_KILL", "log": _log_lines}
        _write(report, args.output_dir)
        return report

    s2 = run_stage2(panel, spy_close, s1["signal"], s1["decision_dates"], config.DEFAULT_PORTFOLIO_PARAMS, config.DEFAULT_COSTS, config.DEFAULT_TWIN_PARAMS)
    s3 = run_stage3(panel, s1["signal"], s1["decision_dates"], s2["primary_bt"], args, config.DEFAULT_COSTS)

    rejection = stats.evaluate_rejection(
        s3["dsr_oos"], s3["bootstrap"], s1["shuffle_null"], s2["twins_comparison"],
        s3["concentration"], s3["sign_stability"], s3["pead_delta_sharpe"],
    )
    verdict = "REJECT" if rejection["reject"] else "DO NOT REJECT (survives pre-registered falsification suite)"
    if s1["stage1_kill"]:
        verdict = "REJECT (stage 1 kill: " + verdict + ")"
    _log(f"FINAL VERDICT: {verdict}")
    _log(f"  reasons: {rejection['reasons']}")

    report = {
        "universe": {"n_names": int(panel.close.shape[1]), "n_days": int(panel.close.shape[0]),
                     "start": args.start, "end": args.end, "sp400_included": args.sp400,
                     "membership_note": provider.sector_coverage_note()},
        "stage1": _strip_heavy(s1),
        "stage2": {
            "primary": stats.summary_stats(s2["primary_bt"].weekly_returns),
            "variant_monday_open": stats.summary_stats(s2["variant_bt"].weekly_returns),
            "twins": {k: stats.summary_stats(v.weekly_returns) for k, v in s2["twin_backtests"].items()},
            "diversification": s2["diversification"], "twins_comparison": s2["twins_comparison"],
        },
        "stage3": {
            "grid_table": s3["grid_result"].table.to_dict(orient="records") if s3["grid_result"] is not None else None,
            "dsr_full": s3["dsr_full"], "dsr_oos": s3["dsr_oos"], "isolation": s3["isolation"],
            "bootstrap": s3["bootstrap"], "pead_delta_sharpe": s3["pead_delta_sharpe"],
            "pead_excluded_summary": stats.summary_stats(s3["pead_bt"].weekly_returns),
            "concentration": s3["concentration"], "sign_stability": s3["sign_stability"],
        },
        "rejection_criteria": rejection,
        "verdict": verdict,
        "log": _log_lines,
    }
    _write(report, args.output_dir)

    if s3["grid_result"] is not None:
        s3["grid_result"].table.to_csv(os.path.join(args.output_dir, "grid_table.csv"), index=False)
    s2["primary_bt"].weekly_returns.to_csv(os.path.join(args.output_dir, "primary_weekly_returns.csv"), header=["net_return"])
    for name, bt in s2["twin_backtests"].items():
        bt.weekly_returns.to_csv(os.path.join(args.output_dir, f"twin_{name}_weekly_returns.csv"), header=["net_return"])

    _log(f"done. results written to {args.output_dir}")
    return report


def _strip_heavy(s1: dict) -> dict:
    return {
        "ic": {k: v for k, v in s1["ic"].items() if k != "ic_series"},
        "ic_series": s1["ic"]["ic_series"],
        "shuffle_null": s1["shuffle_null"],
        "stage1_kill": s1["stage1_kill"],
    }


def _write(report: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, default=str)


if __name__ == "__main__":
    main()
