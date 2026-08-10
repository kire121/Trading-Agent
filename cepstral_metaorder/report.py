"""
Turns a pipeline.run_pipeline() result into JSON-safe summary stats and a
markdown report.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd


def performance_stats(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    r = daily_returns.dropna()
    if len(r) < 5:
        return {"n": int(len(r)), "note": "insufficient history"}
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    ann_return = float(r.mean() * periods_per_year)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    return {
        "n": int(len(r)),
        "total_return": float(cum.iloc[-1] - 1),
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((r > 0).mean()),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {(str(k) if not isinstance(k, str) else k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Series,)):
        return _jsonable(obj.to_dict())
    if isinstance(obj, (pd.DataFrame,)):
        return None  # summarized separately; too large/not JSON-natural for the summary blob
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def summarize(result: dict, meta: dict) -> dict:
    main_perf = performance_stats(result["main_backtest"]["daily"]["net_return"])
    main_perf_gross = performance_stats(result["main_backtest"]["daily"]["gross_return"])
    twin_perf = {name: performance_stats(bt["daily"]["net_return"]) for name, bt in result["twin_backtests"].items()}

    avg_turnover = float(result["main_backtest"]["daily"]["turnover"].mean())
    avg_positions = float(result["main_backtest"]["daily"]["n_positions"].mean())
    max_positions = int(result["main_backtest"]["daily"]["n_positions"].max())

    summary = {
        "meta": meta,
        "universe": {
            "n_symbols_with_signal": len(result["signal_layer"]["signal_by_symbol"]),
        },
        "main_strategy_performance": {**main_perf, "gross": main_perf_gross,
                                       "avg_daily_turnover": avg_turnover,
                                       "avg_simultaneous_positions": avg_positions,
                                       "max_simultaneous_positions": max_positions},
        "twin_performance": twin_perf,
        "twin_liveness": result["twin_liveness"],
        "step0_existence": result["step0_existence"],
        "step0_breadth": result["step0_breadth"],
        "step1": {"nw": result["step1"]["nw"], "n_days": result["step1"]["n_days"],
                  "avg_cross_section": result["step1"].get("avg_cross_section"),
                  "passed": result["step1"]["passed"]},
        "step2": result.get("step2"),
        "step2_diagnostic_only": result.get("step2_diagnostic_only", False),
        "verdict": result["verdict"],
    }
    return json.loads(json.dumps(_jsonable(summary)))


def _fmt(x, pct=False, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, (int, float)):
        if pct:
            return f"{x * 100:.{digits}f}%"
        return f"{x:.{digits}f}"
    return str(x)


def to_markdown(summary: dict) -> str:
    m = summary["meta"]
    perf = summary["main_strategy_performance"]
    lines = []
    lines.append(f"# Cepstral metaorder-slicing signal -- pilot results\n")
    lines.append(f"Universe: {m.get('n_symbols')} curated liquid US names. "
                 f"Period: {m.get('intraday_start')} to {m.get('intraday_end')} "
                 f"({perf.get('n')} scored trading days). Generated: {m.get('generated_at')}.\n")

    v = summary["verdict"]
    verdict_str = "REJECTED" if v.get("rejected") else "NOT REJECTED (survives this pilot's tests)"
    lines.append(f"## Verdict: {verdict_str}\n")
    lines.append(f"Stage reached: {v.get('stage_reached')}\n")
    if v.get("reasons"):
        lines.append("Reasons:")
        for r in v["reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("## Step 0: existence + breadth\n")
    e = summary["step0_existence"]
    lines.append(f"- Existence (permutation null): exceedance fraction = {_fmt(e.get('exceedance_fraction'), pct=True)} "
                 f"vs required >= {_fmt(e.get('required_fraction'), pct=True)} "
                 f"(n={e.get('n_sampled')} stock-days sampled) -> **{_fmt(e.get('passed'))}**")
    b = summary["step0_breadth"]
    lines.append(f"- PC1 share of S_bar panel: {_fmt(b.get('pc1_share'))} (must be < 0.40) -> **{_fmt(b.get('pc1_ok'))}**")
    lines.append(f"- Cross-sectional dispersion: median={_fmt(b.get('median_real_dispersion'))} vs "
                 f"block-null p95={_fmt(b.get('block_null_p95_dispersion'))} -> **{_fmt(b.get('dispersion_ok'))}**")
    lines.append(f"- Avg simultaneous positions: {_fmt(b.get('avg_simultaneous_positions'), digits=1)} "
                 f"(spec wants >= {b.get('min_required_positions')}; not meaningful at pilot scale, see README) "
                 f"-> {_fmt(b.get('positions_ok'))}\n")

    lines.append("## Step 1: Fama-MacBeth redundancy screen\n")
    s1 = summary["step1"]
    nw = s1.get("nw", {})
    lines.append(f"- Incremental coefficient on S_bar, NW t-stat = {_fmt(nw.get('t_stat'))} "
                 f"(need |t| >= 2.0), over {s1.get('n_days')} days, avg cross-section "
                 f"{_fmt(s1.get('avg_cross_section'), digits=1)} names/day -> **{_fmt(s1.get('passed'))}**\n")

    if summary.get("step2"):
        s2 = summary["step2"]
        if summary.get("step2_diagnostic_only"):
            lines.append("## Step 2 (DIAGNOSTIC ONLY -- Step 0/1 already rejected; not part of the verdict)\n")
            lines.append("Run anyway to confirm the full battery executes correctly end-to-end on real data. "
                         "Per the pre-registered rejection rule, Step 0/1 failing already kills the idea "
                         "regardless of what follows.\n")
        else:
            lines.append("## Step 2: costs, deflated Sharpe, robustness, twins\n")
        dsr = s2.get("dsr", {})
        lines.append(f"- Net Sharpe (base variant, per-period): {_fmt(dsr.get('sr'))}, "
                     f"annualized: {_fmt(dsr.get('sr_annual'))}")
        lines.append(f"- Deflated Sharpe Ratio: {_fmt(dsr.get('dsr'))} (need > 0.5) over "
                     f"{dsr.get('n_trials')} trials -> **{_fmt(dsr.get('dsr_above_half'))}**")
        sc = s2.get("sign_consistency", {})
        lines.append(f"- Sign consistency: missing in {sc.get('n_missing_consistency')} of 4 IS subperiods "
                     f"-> **{_fmt(sc.get('passed'))}**")
        tr = s2.get("twin_race", {})
        lines.append(f"- Baseline race, 5 legs (3 named twins + 2 null-hypothesis baselines), "
                     f"main Sharpe={_fmt(tr.get('main_sharpe'))}: **{_fmt(tr.get('passed'))}**")
        for name, info in tr.get("twins", {}).items():
            lines.append(f"  - vs {name}: alive={info.get('alive')}, comparator Sharpe={_fmt(info.get('twin_sharpe'))}, "
                         f"beaten_by_main={info.get('beaten_by_main')}")
        if s2.get("beta_to_spy"):
            beta = s2["beta_to_spy"]
            lines.append(f"- Beta to SPY: {_fmt(beta.get('beta'))} (need |beta| < 0.15) -> **{_fmt(beta.get('passed'))}**")
        if s2.get("tsmom_corr"):
            tc = s2["tsmom_corr"]
            lines.append(f"- Correlation to TSMOM proxy: {_fmt(tc.get('corr'))} (need |corr| < 0.25) -> "
                         f"**{_fmt(tc.get('passed'))}**")
        lines.append("")

    lines.append("## Main strategy performance (net of pilot cost model)\n")
    lines.append(f"- N days: {perf.get('n')}, annualized return: {_fmt(perf.get('annualized_return'), pct=True)}, "
                 f"annualized vol: {_fmt(perf.get('annualized_vol'), pct=True)}, Sharpe: {_fmt(perf.get('sharpe'))}")
    lines.append(f"- Max drawdown: {_fmt(perf.get('max_drawdown'), pct=True)}, hit rate: {_fmt(perf.get('hit_rate'), pct=True)}")
    lines.append(f"- Avg daily turnover: {_fmt(perf.get('avg_daily_turnover'), pct=True)}, "
                 f"avg simultaneous positions: {_fmt(perf.get('avg_simultaneous_positions'), digits=1)}\n")

    lines.append("## Twin performance\n")
    for name, p in summary["twin_performance"].items():
        alive = summary["twin_liveness"].get(name, {}).get("alive")
        lines.append(f"- {name} (alive={alive}): Sharpe={_fmt(p.get('sharpe'))}, "
                     f"annualized return={_fmt(p.get('annualized_return'), pct=True)}")

    return "\n".join(lines) + "\n"
