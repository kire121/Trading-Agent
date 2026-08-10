"""The single, locked OOS reading -- run exactly once, on the full history
of the 16-country + 12-commodity panel (config.OOS_UNIVERSE), using every
parameter frozen by run_research.py (theta, z-threshold, kappa, tau floor/
cap, p_star). This is the 3rd read of the 16-lands-ETF panel specifically
(after Vindkastet and Dammluckan -- see config.py / REPORT.md), which the
DSR correction accounts for explicitly.

Nothing here is fit, tuned, or chosen by looking at this panel's own data:
priors_dict's per-instrument entries fall back to the frozen IS global
prior for every one of these tickers (none were in the IS panel), exactly
as the "instrument prior p_bar_i skattad enbart pa IS-panelen" rule
requires.

Run with: python -m research.omori.run_oos  (after run_research.py has
produced output/priors.json, output/kappa.json, output/p_star.json).
"""
import json
import os

import numpy as np

from research.omori import backtest, config, data, events, metrics, priors

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")


def run():
    panel = data.load("secondary")
    priors_dict = priors.load_priors()
    with open(os.path.join(OUTPUT_DIR, "kappa.json")) as f:
        kappa = json.load(f)["kappa"]
    with open(os.path.join(OUTPUT_DIR, "p_star.json")) as f:
        p_star = json.load(f)["p_star"]

    ef = events.EventFields(panel)
    res = backtest.run_backtest(panel, priors_dict, p_star, kappa=kappa, ef=ef)
    summ = metrics.summarize(res.daily_returns, res.events_frame())

    lands = set(config.OOS_LANDS)
    ev = res.events_frame()
    # Sensitivity check: event-level stats excluding the 3 not-fully-virgin
    # commodity tickers (GLD/SLV/USO, see config.py caveat).
    is_clean_event = ~ev["ticker"].isin(config.OOS_COMMODITIES_CONTAMINATED)

    print(f"[oos] n_events={len(res.closed_events)} sharpe={summ['sharpe']:.3f} "
          f"total_return={res.daily_returns.sum():.4f} maxdd={summ['max_drawdown']:.4f}")
    print(f"[oos] events by exit_reason:\n{ev.exit_reason.value_counts()}")
    print(f"[oos] events lands vs commodities: "
          f"{ev['ticker'].isin(lands).sum()} lands / {(~ev['ticker'].isin(lands)).sum()} commodities")
    print(f"[oos] events on contaminated (GLD/SLV/USO): {(~is_clean_event).sum()}")

    n_clean = int(is_clean_event.sum())
    clean_sharpe = np.nan
    if n_clean >= 20:
        # Reconstruct daily returns excluding contaminated-ticker events'
        # contribution is not separable post-hoc from the shared daily
        # series without re-running; instead report clean-subset event-level
        # net-return stats as the sensitivity check (documented in REPORT.md).
        clean_net = ev.loc[is_clean_event, "net_return"]
        clean_sharpe = float(clean_net.mean() / clean_net.std(ddof=1)) if clean_net.std(ddof=1) > 0 else np.nan

    out = {
        "n_events": len(res.closed_events),
        "summary": summ,
        "kappa": kappa, "p_star": p_star,
        "events_lands": int(ev["ticker"].isin(lands).sum()),
        "events_commodities": int((~ev["ticker"].isin(lands)).sum()),
        "events_contaminated_commodities": int((~is_clean_event).sum()),
        "clean_subset_n_events": n_clean,
        "clean_subset_event_level_sharpe_proxy": clean_sharpe,
        "min_events_oos_met": len(res.closed_events) >= config.MIN_EVENTS_OOS,
    }
    with open(os.path.join(OUTPUT_DIR, "oos_results_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    res.daily_returns.to_csv(os.path.join(OUTPUT_DIR, "oos_daily_returns.csv"))
    ev.to_csv(os.path.join(OUTPUT_DIR, "oos_events.csv"), index=False)
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
