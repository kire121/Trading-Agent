"""
Vindkastet -- Step A: build the signal panel for every (K, ridge) cell of the
pre-registered grid and count trigger events across the full
quantile x alignment x K x ridge grid (24 variants).

Writes:
  output/grid_event_counts.csv
  output/panels_cache.pkl  (gitignored -- regenerate by rerunning this script)
"""
import itertools
import pickle
import time

import pandas as pd

import core

QUANTILES = [0.90, 0.95]
ALIGNS = [0.5, 0.65]
KS = [2, 3, 5]
RIDGES = [0.05, 0.1]


def main():
    prices = pd.read_csv("data/prices_primary.csv", index_col=0, parse_dates=True)
    returns = core.log_returns(prices)

    results = []
    panels_cache = {}
    t0 = time.time()
    for K, ridge in itertools.product(KS, RIDGES):
        panel = core.build_signal_panel(returns, K=K, ridge_param=ridge)
        panels_cache[(K, ridge)] = panel
        for q, a in itertools.product(QUANTILES, ALIGNS):
            trig = core.trigger_series(panel, quantile=q, align_thresh=a, eta_thresh=2.0)
            results.append(dict(K=K, ridge=ridge, quantile=q, align=a, n_trig=int(trig.sum())))
        print(f"done K={K} ridge={ridge} elapsed={time.time() - t0:.1f}s")

    df = pd.DataFrame(results).sort_values("n_trig", ascending=False)
    print(df.to_string())
    df.to_csv("output/grid_event_counts.csv", index=False)
    with open("output/panels_cache.pkl", "wb") as f:
        pickle.dump(panels_cache, f)
    print("Max events across entire 24-variant grid:", df["n_trig"].max(),
          "(required >= 100 per protocol Regel 4 / expected 150-285 total)")


if __name__ == "__main__":
    main()
