"""One-off supplementary analysis: charts + IS/OOS breakdown table for the
written report. Run after run.py has populated oglegrinden/results/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oglegrinden.universe import ALL_TICKERS
from oglegrinden.data import load_universe, Panel
from oglegrinden.signal import build_gate_bundle, weekly_fridays
from oglegrinden.backtest import run_backtest, always_on_baseline
from oglegrinden.grid import DEFAULT_CORR_WINDOW, DEFAULT_FORMATION_DAYS
from oglegrinden.stats import summary_stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OOS_START = pd.Timestamp("2018-01-01")


def main():
    histories = load_universe(ALL_TICKERS)
    panel = Panel(histories)
    all_fridays = weekly_fridays(panel.close.index)
    bundle60 = build_gate_bundle(panel, corr_window=DEFAULT_CORR_WINDOW)

    primary = run_backtest(panel, bundle60, all_fridays, signal_name="L", direction="primary",
                            formation_days=DEFAULT_FORMATION_DAYS)
    mirror = run_backtest(panel, bundle60, all_fridays, signal_name="L", direction="mirror",
                           formation_days=DEFAULT_FORMATION_DAYS)
    always_on = always_on_baseline(panel, all_fridays, formation_days=DEFAULT_FORMATION_DAYS)
    twins = {
        sig: run_backtest(panel, bundle60, all_fridays, signal_name=sig, direction="primary",
                           formation_days=DEFAULT_FORMATION_DAYS)
        for sig in ["rho_bar", "absorption_ratio", "index_vol"]
    }

    # --- IS/OOS breakdown table -------------------------------------------------
    signal_start = bundle60.raw.index.min()
    rows = []
    series_map = {"primary (H1)": primary.weekly_returns, "mirror": mirror.weekly_returns,
                   "always_on": always_on.weekly_returns, **{f"twin:{k}": v.weekly_returns for k, v in twins.items()}}
    for name, s in series_map.items():
        is_mask = (s.index >= signal_start) & (s.index < OOS_START)
        oos_mask = s.index >= OOS_START
        is_stats = summary_stats(s[is_mask])
        oos_stats = summary_stats(s[oos_mask])
        rows.append({
            "strategy": name,
            "is_start": str(signal_start.date()),
            "is_sharpe": is_stats["sharpe"], "is_ann_return": is_stats["annualized_return"],
            "is_maxdd": is_stats["max_drawdown"], "is_n": is_stats["n_obs"],
            "oos_sharpe": oos_stats["sharpe"], "oos_ann_return": oos_stats["annualized_return"],
            "oos_maxdd": oos_stats["max_drawdown"], "oos_n": oos_stats["n_obs"],
        })
    is_oos_table = pd.DataFrame(rows)
    is_oos_table.to_csv(os.path.join(RESULTS_DIR, "is_oos_table.csv"), index=False)
    print(is_oos_table.to_string(index=False))

    # --- Equity curve chart -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, s in series_map.items():
        wealth = (1 + s.fillna(0)).cumprod()
        ax.plot(wealth.index, wealth.values, label=name, linewidth=1.3)
    ax.axvline(OOS_START, color="gray", linestyle="--", linewidth=1, label="OOS start (2018)")
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("Oglegrinden: cumulative growth of $1 (net of costs), all variants")
    ax.set_ylabel("Wealth (log scale)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "equity_curves.png"), dpi=130)
    plt.close(fig)

    # --- L_t / percentile / gate-state chart --------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(bundle60.raw.index, bundle60.raw["L"], color="tab:blue", linewidth=0.8, label="raw L_t")
    axes[0].plot(bundle60.smoothed.index, bundle60.smoothed["L"], color="tab:red", linewidth=1.2, label="smoothed L_t (4wk median)")
    axes[0].set_ylabel("Total H1 persistence")
    axes[0].legend(fontsize=8)

    axes[1].plot(bundle60.percentile.index, bundle60.percentile["L"], color="tab:purple", linewidth=1.0)
    axes[1].axhline(60, color="green", linestyle="--", linewidth=0.8)
    axes[1].axhline(40, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Expanding percentile")

    gate = bundle60.gates[("L", "primary")].reindex(all_fridays).ffill().fillna(False)
    axes[2].fill_between(gate.index, 0, gate.astype(int), step="post", color="tab:green", alpha=0.5)
    axes[2].set_ylabel("Gate ON (primary)")
    axes[2].set_ylim(0, 1.1)
    axes[2].axvline(OOS_START, color="gray", linestyle="--", linewidth=1)

    fig.suptitle("Oglegrinden signal: H1 persistence, percentile, gate state (60d correlation window)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "signal_and_gate.png"), dpi=130)
    plt.close(fig)

    print("Charts written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
