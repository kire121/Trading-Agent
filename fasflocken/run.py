"""
Fasflocken (PH-1) -- CLI entry point.

Defaults to SyntheticUniverseProvider (no credentials needed). Pass
--provider eodhd to run against real data via EODHDProvider, which reads
EODHD_API_KEY / EODHD_API_TOKEN from the environment -- see universe.py /
README.md's "Using real data (EODHD)" section for what that provider
actually does and its one disclosed coverage gap.

Usage:
    python -m fasflocken.run demo
    python -m fasflocken.run grid --start 2015-01-01 --end 2019-12-31
    python -m fasflocken.run full --start 2015-01-01 --end 2019-12-31
    python -m fasflocken.run full --provider eodhd --n-sectors 11 \
        --start 2002-01-01 --eval-start 2004-01-01 --end 2026-08-09
"""

from __future__ import annotations

import argparse
import datetime as _dt

from fasflocken.config import DEFAULT_PARAMS, GICS_SECTORS
from fasflocken.universe import SyntheticUniverseProvider, EODHDProvider
from fasflocken.backtest import (
    run_backtest,
    annualized_sharpe,
    annualized_return,
    annualized_vol,
    max_drawdown,
    since,
)
from fasflocken.pipeline import compute_sector_signal, build_z_panel
from fasflocken import stats
from fasflocken import grid_search


def _parse_date(s: str) -> _dt.date:
    return _dt.datetime.strptime(s, "%Y-%m-%d").date()


def _build_provider(args):
    if args.provider == "eodhd":
        return EODHDProvider(cache_dir=args.cache_dir, max_workers=args.max_workers)
    sectors = tuple(list(GICS_SECTORS)[: args.n_sectors])
    return SyntheticUniverseProvider(
        start=args.start, end=args.end, n_per_sector=args.n_per_sector, seed=args.seed, sectors=sectors
    )


def cmd_demo(args) -> None:
    sectors = tuple(list(GICS_SECTORS)[: args.n_sectors])
    provider = _build_provider(args)
    bt = run_backtest(provider, args.start, args.end, DEFAULT_PARAMS, sectors=sectors)
    r = since(bt.weekly_returns, args.eval_start)

    print(f"Fasflocken demo backtest ({args.start} .. {args.end}), {len(sectors)} sectors, {args.provider} data")
    if args.eval_start:
        print(f"  (performance stats from {args.eval_start} onward, excluding burn-in)")
    print(f"  weeks traded        : {len(r)}")
    print(f"  Sharpe (net, ann.)   : {annualized_sharpe(r):.3f}")
    print(f"  Return (ann.)        : {annualized_return(r):.3%}")
    print(f"  Vol (ann.)           : {annualized_vol(r):.3%}")
    print(f"  Max drawdown (cum lr): {max_drawdown(r):.3%}")
    print(f"  Avg weekly turnover  : {bt.turnover.mean():.3f}")
    print(f"  Avg vol-target k     : {bt.k_scale.mean():.3f}")


def cmd_grid(args) -> None:
    sectors = tuple(list(GICS_SECTORS)[: args.n_sectors])
    provider = _build_provider(args)

    def _progress(row):
        print(f"  cell {row['cell_id']:<28} sharpe={row['sharpe']:.3f}")

    result = grid_search.run_grid(
        provider, args.start, args.end, sectors=sectors, progress=_progress if args.verbose else None,
        eval_start=args.eval_start,
    )
    print(result.results.sort_values("sharpe", ascending=False).to_string(index=False))


def _save_full_results(output_dir, args, main_returns, grid, boot, delta_sr, oracle_ret, dsr, sign, isolation, verdict) -> None:
    """Persist the `full` run's results to disk -- stdout-only meant this was
    otherwise lost the moment the process exited, with no way to inspect the
    81-cell grid breakdown after the fact.
    """
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    grid.results.to_csv(os.path.join(output_dir, "grid_results.csv"), index=False)
    main_returns.to_csv(os.path.join(output_dir, "main_weekly_returns.csv"), header=["net_return"])

    summary = {
        "provider": args.provider,
        "start": str(args.start),
        "end": str(args.end),
        "eval_start": str(args.eval_start) if args.eval_start else None,
        "n_sectors": args.n_sectors,
        "bootstrap_draws": args.bootstrap_draws,
        "net_sharpe": annualized_sharpe(main_returns),
        "bootstrap_p_value": boot.p_value,
        "delta_sharpe_vs_twin": delta_sr,
        "oracle_sharpe": annualized_sharpe(oracle_ret),
        "dsr": dsr,
        "sign_stability": sign,
        "isolation": isolation,
        "verdict": verdict,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


def cmd_full(args) -> None:
    sectors = tuple(list(GICS_SECTORS)[: args.n_sectors])
    provider = _build_provider(args)
    eval_start = args.eval_start

    print("1/6 computing sector signals (default grid cell)...")
    sector_signals = {s: compute_sector_signal(provider, s, args.start, args.end, DEFAULT_PARAMS) for s in sectors}
    z_panel = build_z_panel(sector_signals)
    main = run_backtest(provider, args.start, args.end, DEFAULT_PARAMS, sectors=sectors, z_override=z_panel)
    main_returns = since(main.weekly_returns, eval_start)
    print(f"    net Sharpe = {annualized_sharpe(main_returns):.3f}")

    print("2/6 running declared grid...")
    grid = grid_search.run_grid(provider, args.start, args.end, sectors=sectors, eval_start=eval_start)
    trial_sharpes = grid.results["sharpe"].to_numpy()  # annualized -- see stats.deflated_sharpe_ratio's docstring

    print("3/6 circular block bootstrap (this is the slow one)...")
    boot = stats.circular_block_bootstrap_pvalue(
        provider, args.start, args.end, DEFAULT_PARAMS, n_draws=args.bootstrap_draws, sectors=sectors, z_panel=z_panel
    )
    print(f"    p-value = {boot.p_value:.4f}")

    print("4/6 boring-twin comparison...")
    twin = stats.run_twin_backtest(provider, args.start, args.end, DEFAULT_PARAMS, sectors=sectors, sector_signals=sector_signals)
    twin_returns = since(twin.weekly_returns, eval_start)
    delta_sr = annualized_sharpe(main_returns) - annualized_sharpe(twin_returns)
    print(f"    delta Sharpe vs twin = {delta_sr:.3f}")

    print("5/6 oracle cap + sign stability + DSR...")
    oracle_ret = since(stats.oracle_backtest(provider, args.start, args.end, n_legs=DEFAULT_PARAMS.n_legs, sectors=sectors), eval_start)
    dsr = stats.deflated_sharpe_ratio(main_returns, trial_sharpes)
    sign = stats.sign_stability(main_returns)
    print(f"    oracle Sharpe        = {annualized_sharpe(oracle_ret):.3f}")
    print(f"    DSR gap              = {dsr['deflated_sharpe_gap']:.3f} (psr={dsr['psr']:.4f})")
    print(f"    sign stable          = {sign['sign_stable']}")

    print("6/6 grid isolation check + verdict...")
    isolation = grid_search.neighborhood_isolation_check(grid.results, DEFAULT_PARAMS)
    verdict = grid_search.evaluate_rejection(dsr, boot.p_value, delta_sr, sign, isolation)
    print(f"    isolated gridcell?   = {isolation.get('isolated')}")
    print()
    print("=" * 60)
    print("VERDICT:", "REJECT" if verdict["reject"] else "SURVIVES null-hypothesis baseline")
    for reason, fired in verdict["reasons"].items():
        print(f"  [{'X' if fired else ' '}] {reason}")
    print("=" * 60)

    if args.output_dir:
        _save_full_results(
            args.output_dir, args, main_returns, grid, boot, delta_sr, oracle_ret, dsr, sign, isolation, verdict
        )
        print(f"Results written to {args.output_dir}/")

    if args.provider == "synthetic":
        print(
            "Note: this is a synthetic-data smoke run, not a real backtest -- "
            "pass --provider eodhd for real point-in-time data."
        )


def _common_args_parser() -> argparse.ArgumentParser:
    """Shared options, attached to every subcommand so they can follow it
    on the command line (e.g. `run.py demo --start ...`), not just precede it.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", type=_parse_date, default=_dt.date(2015, 1, 1))
    common.add_argument("--end", type=_parse_date, default=_dt.date(2019, 12, 31))
    common.add_argument(
        "--eval-start", type=_parse_date, default=None,
        help="only compute performance stats (Sharpe/DSR/etc.) from this date onward, excluding an earlier "
             "burn-in period included in --start; defaults to --start (no slicing)",
    )
    common.add_argument(
        "--provider", choices=["synthetic", "eodhd"], default="synthetic",
        help="synthetic (default, no credentials needed) or eodhd (real data, needs EODHD_API_KEY)",
    )
    common.add_argument("--seed", type=int, default=7, help="synthetic provider only")
    common.add_argument("--n-per-sector", type=int, default=20, help="synthetic provider only")
    common.add_argument("--cache-dir", type=str, default=None, help="eodhd provider only; default: system temp dir")
    common.add_argument("--max-workers", type=int, default=6, help="eodhd provider only; thread pool size for fan-out fetches")
    common.add_argument("--n-sectors", type=int, default=9, help="use the first N of the 11 GICS sectors (default 9, always-on)")
    common.add_argument("--bootstrap-draws", type=int, default=200)
    common.add_argument(
        "--output-dir", type=str, default=None,
        help="`full` command only: write grid_results.csv, main_weekly_returns.csv, and summary.json here",
    )
    common.add_argument("--verbose", action="store_true")
    return common


def build_arg_parser() -> argparse.ArgumentParser:
    common = _common_args_parser()
    p = argparse.ArgumentParser(description="Fasflocken (PH-1) sector phase-coherence strategy", parents=[common])

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", parents=[common]).set_defaults(func=cmd_demo)
    sub.add_parser("grid", parents=[common]).set_defaults(func=cmd_grid)
    sub.add_parser("full", parents=[common]).set_defaults(func=cmd_full)
    return p


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
