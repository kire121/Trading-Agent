"""
Robustness sweep (window x tail-cutoff x quintile/tertile), sub-period
stability, PnL-concentration kill check, and the two diversification checks
(beta vs SPY proxy, correlation vs the TSMOM proxy).
"""
import numpy as np
import pandas as pd

from . import metrics
from .portfolio import BacktestConfig, run_backtest
from .signal import build_d_l_panel, LEFT_TAIL_MASK, U_GRID
from .universe import Universe, eligibility_panel, month_end_dates

WINDOWS = [84, 126, 189]
TAIL_CUTOFFS = [0.10, 0.20, 0.30]
LEG_TYPES = ["quintile", "tertile"]


def _tail_mask_for_cutoff(cutoff):
    return (U_GRID > 0.02) & (U_GRID <= cutoff)


def run_variant(universe: Universe, curr_window, prev_window, tail_cutoff, leg_type,
                 eval_dates=None, adv_threshold=20e6, min_names=10):
    if eval_dates is None:
        eval_dates = month_end_dates(universe.returns.index)
    mask = _tail_mask_for_cutoff(tail_cutoff)
    idx = universe.returns.index
    dates = [d for d in eval_dates if d in idx]
    positions = [idx.get_loc(d) for d in dates]
    ret = universe.returns.fillna(0.0)

    out = {}
    for col in ret.columns:
        from .signal import rolling_d_l
        out[col] = rolling_d_l(ret[col].values, curr_window, prev_window, tail_mask=mask, at_positions=positions)
    d_l = pd.DataFrame(out, index=pd.DatetimeIndex(dates))

    elig = eligibility_panel(universe, d_l, adv_threshold)
    cfg = BacktestConfig(curr_window=curr_window, prev_window=prev_window, min_names=min_names,
                          n_legs=leg_type)
    res = run_backtest(ret, d_l, elig, universe.adv, cfg, rebalance_dates=dates)
    return d_l, elig, res


def robustness_sweep(universe: Universe, min_names=10):
    """Full window x cutoff x leg-type grid. Returns a results table and the
    list of net-return series (for sign-stability / DSR-trial logging)."""
    rows = []
    series = {}
    for w in WINDOWS:
        for cutoff in TAIL_CUTOFFS:
            for leg in LEG_TYPES:
                d_l, elig, res = run_variant(universe, w, w, cutoff, leg, min_names=min_names)
                sh = metrics.sharpe(res["net_returns"])
                key = f"w{w}_cut{cutoff}_{leg}"
                rows.append({
                    "window": w, "tail_cutoff": cutoff, "leg_type": leg,
                    "sharpe_net": sh, "sharpe_gross": metrics.sharpe(res["gross_returns"]),
                    "ann_ret_net": metrics.ann_return(res["net_returns"]),
                    "turnover": res["turnover"]["turnover"].mean(),
                })
                series[key] = res["net_returns"]
    table = pd.DataFrame(rows)
    return table, series


def sign_stability(table: pd.DataFrame, base_mask=None):
    sh = table["sharpe_net"].dropna()
    if len(sh) == 0:
        return {"all_positive": False, "frac_positive": np.nan, "n": 0}
    frac_pos = (sh > 0).mean()
    return {"all_positive": bool((sh > 0).all()), "frac_positive": float(frac_pos), "n": len(sh)}


def sub_period_checks(returns: pd.Series):
    r = returns.dropna()
    half = len(r) // 2
    first_half, second_half = r.iloc[:half], r.iloc[half:]
    ex_covid = r[~((r.index >= "2020-02-15") & (r.index <= "2020-04-30"))]
    return {
        "first_half_sharpe": metrics.sharpe(first_half),
        "second_half_sharpe": metrics.sharpe(second_half),
        "sign_stable_halves": bool(np.sign(metrics.ann_return(first_half) or 0) ==
                                    np.sign(metrics.ann_return(second_half) or 0)),
        "ex_covid_sharpe": metrics.sharpe(ex_covid),
        "pnl_quarter_concentration": metrics.pnl_quarter_concentration(r),
        "pnl_concentration_kill": bool((metrics.pnl_quarter_concentration(r) or 0) > 0.5),
    }


def diversification_checks(strategy_returns: pd.Series, spy_returns: pd.Series, tsmom_returns: pd.Series):
    df = pd.concat([strategy_returns.rename("strat"), spy_returns.rename("spy")], axis=1).dropna()
    beta = np.nan
    if len(df) > 30 and df["spy"].var() > 0:
        beta = np.polyfit(df["spy"].values, df["strat"].values, 1)[0]

    df2 = pd.concat([strategy_returns.rename("strat"), tsmom_returns.rename("tsmom")], axis=1).dropna()
    corr = df2["strat"].corr(df2["tsmom"]) if len(df2) > 30 else np.nan

    return {
        "beta_vs_spy": float(beta) if pd.notna(beta) else np.nan,
        "beta_pass": bool(abs(beta) < 0.15) if pd.notna(beta) else False,
        "corr_vs_tsmom": float(corr) if pd.notna(corr) else np.nan,
        "corr_pass": bool(abs(corr) < 0.25) if pd.notna(corr) else False,
    }
