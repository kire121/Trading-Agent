"""
Loads the cached pilot data (run fetch_pilot_data.py first) and runs the full
pipeline. Writes cepstral_metaorder/pilot_results/summary.json and REPORT.md.

Usage: python -m cepstral_metaorder.run_pilot
"""

import datetime as dt
import json
from pathlib import Path

from . import data_cache as dc
from . import pilot_config as pc
from . import pipeline, report

RESULTS_DIR = Path(__file__).parent / "pilot_results"


def main():
    print("Loading cached data...", flush=True)
    raw_intraday, eod_raw, market_caps = {}, {}, {}
    missing = []
    for sym in pc.PILOT_SYMBOLS:
        intr = dc.cached_intraday_1m(sym, pc.INTRADAY_START, pc.INTRADAY_END)
        eod = dc.cached_eod(sym, frm=pc.EOD_START, to=pc.EOD_END)
        if intr.empty or eod.empty:
            missing.append(sym)
            continue
        raw_intraday[sym] = intr
        eod_raw[sym] = eod
        market_caps[sym] = dc.cached_market_cap(sym)

    if missing:
        print(f"WARNING: missing/empty data for {missing}, excluding from pilot", flush=True)

    market_proxy_eod = dc.cached_eod(pc.MARKET_PROXY, frm=pc.EOD_START, to=pc.EOD_END)
    tsmom_basket_eod = {s: dc.cached_eod(s, frm=pc.EOD_START, to=pc.EOD_END) for s in pc.TSMOM_BASKET}

    print(f"Loaded {len(raw_intraday)} symbols. Running pipeline...", flush=True)
    result = pipeline.run_pipeline(
        raw_intraday, eod_raw, market_cap_by_symbol=market_caps,
        market_proxy_eod=market_proxy_eod, tsmom_basket_eod=tsmom_basket_eod,
        run_step2=True, force_step2_diagnostics=True, seed=0,
    )
    print("Pipeline complete. Verdict:", result["verdict"], flush=True)

    meta = {
        "n_symbols": len(raw_intraday),
        "symbols_excluded": missing,
        "intraday_start": pc.INTRADAY_START.isoformat(),
        "intraday_end": pc.INTRADAY_END.isoformat(),
        "eod_start": pc.EOD_START,
        "eod_end": pc.EOD_END,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    summary = report.summarize(result, meta)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = report.to_markdown(summary)
    with open(RESULTS_DIR / "REPORT.md", "w") as f:
        f.write(md)

    result["main_backtest"]["daily"].to_csv(RESULTS_DIR / "main_backtest_daily.csv")
    result["main_backtest"]["weights"].to_csv(RESULTS_DIR / "main_backtest_weights.csv")
    for name, bt in result["twin_backtests"].items():
        bt["daily"].to_csv(RESULTS_DIR / f"twin_{name}_daily.csv")
    if not result["fm_panel"].empty:
        result["fm_panel"].to_csv(RESULTS_DIR / "fm_panel.csv", index=False)

    print(f"Wrote results to {RESULTS_DIR}", flush=True)
    print(md, flush=True)


if __name__ == "__main__":
    main()
