"""
Fasflocken (PH-1) -- CLI entry point.

No real market-data credentials are configured in this environment (see
universe.py / README.md), so every mode here runs against
SyntheticUniverseProvider by default. Swap in NorgateProvider /
SharadarProvider / EODHDProvider (once configured with real credentials)
by passing --provider and the relevant env vars; the rest of the pipeline
is provider-agnostic.

Usage:
    python -m fasflocken.run demo
    python -m fasflocken.run grid --start 2015-01-01 --end 2019-12-31
    python -m fasflocken.run full --start 2015-01-01 --end 2019-12-31
"""

from __future__ import annotations

import argparse
import datetime as _dt

from fasflocken.config import DEFAULT_PARAMS, GICS_SECTORS
from fasflocken.universe import SyntheticUniverseProvider
from fasflocken.backtest import run_backtest, annualized_sharpe, annualized_return, annualized_vol, max_drawdown
from fasflocken.pipeline import compute_sector_signal, build_z_panel
from fasflocken import stats
from fasflocken import grid_search


def _parse_date(s: str) -> _dt.date:
    return _dt.datetime.strptime(s, "%Y-%m-%d").date()


def _demo_provider(start: _dt.date, end: _dt.date, seed: int, n_per_sector: int, sectors) -> SyntheticUniverseProvider:
    return SyntheticUniverseProvider(start=start, end=end, n_per_sector=n_per_sector, seed=seed, sectors=sectors)


def cmd_demo(args) -> None:
    sectors = tuple(list(GICS_SECTORS)[: args.n_sectors])
    provider = _demo_provider(args.start, args.end, args.seed, args.n_per_sector, sectors)
    bt = run_backtest(provider, args.start, args.end, DEFAULT_PARAMS, sectors=sectors)

    print(f"Fasflocken demo backtest ({args.start} .. {args.end}), {len(sectors)} sectors, synthetic data")
    print(f"  weeks traded        : {len(bt.weekly_returns)}")
    print(f"  Sharpe (net, ann.)   : {annualized_sharpe(bt.weekly_returns):.3f}")
    print(f"  Return (ann.)        : {annualized_return(bt.weekly_returns):.3%}")
    print(f"  Vol (ann.)           : {annualized_vol(bt.weekly_returns):.3%}")
    print(f"  Max drawdown (cum lr): {max_drawdown(bt.weekly_returns):.3%}")
    print(f"  Avg weekly turnover  : {bt.turnover.mean():.3f}")
    print(f"  Avg vol-target k     : {bt.k_scale.mean():.3f}")


def cmd_grid(args) -> None:
    sectors = tuple(list(GICS_SECTORS)[: args.n_sectors])
    provider = _demo_provider(args.start, args.end, args.seed, args.n_per_sector, sectors)

    def _progress(row):
        print(f"  cell {row['cell_id']:<28} sharpe={row['sharpe']:.3f}")

    result = grid_search.run_grid(provider, args.start, args.end, sectors=sectors, progress=_progress if args.verbose else None)
    print(result.results.sort_values("sharpe", ascending=False).to_string(index=False))


def cmd_full(args) -> None:
    sectors = tuple(list(GICS_SECTORS)[: args.n_sectors])
    provider = _demo_provider(args.start, args.end, args.seed, args.n_per_sector, sectors)

    print("1/6 computing sector signals (default grid cell)...")
    sector_signals = {s: compute_sector_signal(provider, s, args.start, args.end, DEFAULT_PARAMS) for s in sectors}
    z_panel = build_z_panel(sector_signals)
    main = run_backtest(provider, args.start, args.end, DEFAULT_PARAMS, sectors=sectors, z_override=z_panel)
    print(f"    net Sharpe = {annualized_sharpe(main.weekly_returns):.3f}")

    print("2/6 running declared grid...")
    grid = grid_search.run_grid(provider, args.start, args.end, sectors=sectors)
    trial_sharpes = grid.results["sharpe"].to_numpy()

    print("3/6 circular block bootstrap (this is the slow one)...")
    boot = stats.circular_block_bootstrap_pvalue(
        provider, args.start, args.end, DEFAULT_PARAMS, n_draws=args.bootstrap_draws, sectors=sectors, z_panel=z_panel
    )
    print(f"    p-value = {boot.p_value:.4f}")

    print("4/6 boring-twin comparison...")
    twin = stats.run_twin_backtest(provider, args.start, args.end, DEFAULT_PARAMS, sectors=sectors, sector_signals=sector_signals)
    delta_sr = stats.delta_sharpe_vs_twin(main, twin)
    print(f"    delta Sharpe vs twin = {delta_sr:.3f}")

    print("5/6 oracle cap + sign stability + DSR...")
    oracle_ret = stats.oracle_backtest(provider, args.start, args.end, n_legs=DEFAULT_PARAMS.n_legs, sectors=sectors)
    dsr = stats.deflated_sharpe_ratio(main.weekly_returns, trial_sharpes)
    sign = stats.sign_stability(main.weekly_returns)
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
    print(
        "Note: this is a synthetic-data smoke run, not a real backtest -- "
        "see README.md for how to plug in real point-in-time data."
    )


def _common_args_parser() -> argparse.ArgumentParser:
    """Shared options, attached to every subcommand so they can follow it
    on the command line (e.g. `run.py demo --start ...`), not just precede it.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", type=_parse_date, default=_dt.date(2015, 1, 1))
    common.add_argument("--end", type=_parse_date, default=_dt.date(2019, 12, 31))
    common.add_argument("--seed", type=int, default=7)
    common.add_argument("--n-per-sector", type=int, default=20)
    common.add_argument("--n-sectors", type=int, default=9, help="use the first N of the 11 GICS sectors (default 9, always-on)")
    common.add_argument("--bootstrap-draws", type=int, default=200)
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
