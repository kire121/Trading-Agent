"""
Vindkastet -- Step C: descriptive (non-decisive) backtest of the best available
grid cells, plus a diversification cross-check vs ACWI and a TSMOM trend-proxy.

These numbers are reported for transparency only: given the event counts found
in run_grid.py (max 12 across the whole pre-registered grid), nothing here is
statistically powered. The go/no-go verdict is decided in run_gate_checks.py.
"""
import pickle

import numpy as np
import pandas as pd

import backtest
import core


def trend_proxy_returns(prices, returns):
    """Academic substitute for the (unavailable) SG Trend index: a simple
    12m-sign, vol-scaled, daily-rebalanced TSMOM portfolio on the same universe."""
    mom_sign = np.sign(np.log(prices).diff(252))
    vol20 = returns.rolling(20).std()
    w = mom_sign.div(vol20).replace([np.inf, -np.inf], np.nan)
    w = w.div(w.abs().sum(axis=1), axis=0)
    return (w.shift(1) * returns).sum(axis=1)


def main():
    prices = pd.read_csv("data/prices_primary.csv", index_col=0, parse_dates=True)
    bench = pd.read_csv("data/prices_bench.csv", index_col=0, parse_dates=True)
    returns = core.log_returns(prices)
    with open("output/panels_cache.pkl", "rb") as f:
        panels_cache = pickle.load(f)

    print("=== Backtest across top grid cells by event count ===")
    grid_df = pd.read_csv("output/grid_event_counts.csv")
    top = grid_df.sort_values("n_trig", ascending=False).head(8)
    rows = []
    for _, r in top.iterrows():
        K, ridge, q, a = int(r["K"]), r["ridge"], r["quantile"], r["align"]
        panel = panels_cache[(K, ridge)]
        trig = core.trigger_series(panel, quantile=q, align_thresh=a, eta_thresh=2.0)
        trades, results = backtest.run_backtest(panel, prices, returns, trig, K=K)
        row = dict(K=K, ridge=ridge, quantile=q, align=a, n_trig=int(r["n_trig"]))
        if not trades.empty:
            res2 = results[2]
            row.update(n_trades=res2["n_trades"], net_mean_2bp=res2["net_mean"],
                       hit_rate=res2["hit_rate"], sharpe_event_2bp=res2["sharpe_per_event"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(summary.to_string())
    summary.to_csv("output/grid_backtest_summary.csv", index=False)

    print("\n=== Best-cell (K=2, ridge=0.1, q=0.90, align=0.5) trade log ===")
    panel = panels_cache[(2, 0.1)]
    trig = core.trigger_series(panel, quantile=0.90, align_thresh=0.50, eta_thresh=2.0)
    trades, results = backtest.run_backtest(panel, prices, returns, trig, K=2)
    print(trades[["entry", "exit", "ret", "net_ret_2bp", "net_ret_5bp", "replaced"]])
    trades.to_csv("output/best_grid_trades.csv", index=False)

    print("\n=== Diversification check (n=12, descriptive only) ===")
    trend_ret = trend_proxy_returns(prices, returns)
    acwi_ret = np.log(bench["ACWI"]).diff()
    rows = []
    for _, row in trades.iterrows():
        rows.append(dict(entry=row["entry"], strat_ret=row["net_ret_2bp"],
                          acwi_ret=acwi_ret.loc[row["entry"]:row["exit"]].sum(),
                          trend_ret=trend_ret.loc[row["entry"]:row["exit"]].sum()))
    comp = pd.DataFrame(rows)
    print(comp)
    print("corr strat vs ACWI:", comp["strat_ret"].corr(comp["acwi_ret"]), " (kill if |rho|>0.2)")
    print("corr strat vs trend proxy:", comp["strat_ret"].corr(comp["trend_ret"]), " (kill if |rho|>0.2)")
    comp.to_csv("output/diversification_check.csv", index=False)
    trend_ret.to_csv("output/trend_proxy_returns.csv")


if __name__ == "__main__":
    main()
