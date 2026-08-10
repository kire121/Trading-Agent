"""
Vindkastet -- Step D: sign-replication check on an untouched secondary surface
(single-country equity ETFs -- a structurally different cross-section from the
primary multi-asset universe, never used in prior 2018-2026 OOS work).

Re-runs the same event-count, v*-stability and conditional-IC diagnostics as
run_grid.py / run_gate_checks.py, on data/prices_secondary.csv.
"""
import numpy as np
import pandas as pd

import core
from run_gate_checks import week_to_week_cos, conditional_ic


def main():
    prices = pd.read_csv("data/prices_secondary.csv", index_col=0, parse_dates=True)
    returns = core.log_returns(prices)
    panel = core.build_signal_panel(returns, K=3, ridge_param=0.1)

    print("=== Secondary universe: alignment distribution ===")
    print(panel["alignment"].describe())

    print("\n=== Secondary universe: event counts ===")
    for a_th in [0.5, 0.65]:
        for q in [0.90, 0.95]:
            trig = core.trigger_series(panel, quantile=q, align_thresh=a_th, eta_thresh=2.0)
            print(f"quantile={q} align={a_th} n_trig={trig.sum()}")

    print("\n=== Secondary universe: v*-stability ===")
    cos_realized = week_to_week_cos(panel)
    print(f"median |cos|={np.median(cos_realized):.4f} mean={cos_realized.mean():.4f}")

    print("\n=== Secondary universe: conditional IC ===")
    fs, rp, av = conditional_ic(panel, returns, K=3)
    print(f"n={len(fs)} unconditional IC={np.corrcoef(fs, rp)[0, 1]:.4f}")
    for pct in [0.5, 0.75, 0.9, 0.95]:
        thr = np.quantile(av, pct)
        mask = av >= thr
        ic = np.corrcoef(fs[mask], rp[mask])[0, 1]
        print(f"  conditional IC (align top {(1 - pct) * 100:.0f}%, n={mask.sum()}): {ic:.4f}")


if __name__ == "__main__":
    main()
