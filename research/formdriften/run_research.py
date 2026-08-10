"""
Orchestrates the full Formdriften research pipeline end to end:
redundancy screen (gates everything) -> estimator null (mandatory fast-exit)
-> block permutation -> noise floor -> robustness sweep (IS trials for DSR)
-> IS/OOS lock -> kill-criteria evaluation -> report.

Each expensive stage is cached to disk (pickle) so the pipeline can be
resumed/inspected without recomputation. Run with:
    python3 -m research.formdriften.run_research [stage]
stage in {data, base, redundancy, estnull, blockperm, noisefloor,
          robustness, diversify, all} -- default "all".
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from . import metrics, nulls, robustness
from .data_fetch import ETF_UNIVERSE, FX_UNIVERSE
from .portfolio import BacktestConfig, run_backtest
from .tsmom import tsmom_returns
from .universe import build_signal_and_eligibility, load_universe, month_end_dates

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

IS_END = pd.Timestamp("2017-12-31")
OOS_START = pd.Timestamp("2018-01-01")


def _cache(name):
    return os.path.join(OUT_DIR, f"{name}.pkl")


def save(name, obj):
    with open(_cache(name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(_cache(name), "rb") as f:
        return pickle.load(f)


def has(name):
    return os.path.exists(_cache(name))


def stage_data():
    etf_u = load_universe("ETF", ETF_UNIVERSE, is_fx=False)
    fx_u = load_universe("FX", FX_UNIVERSE, is_fx=True)
    spy_u = load_universe("SPY", ["SPY"], is_fx=False)
    save("universes", {"etf": etf_u, "fx": fx_u, "spy": spy_u})
    return etf_u, fx_u, spy_u


def stage_base():
    u = load("universes")
    etf_u, fx_u = u["etf"], u["fx"]

    d_l_etf, elig_etf = build_signal_and_eligibility(etf_u)
    d_l_fx, elig_fx = build_signal_and_eligibility(fx_u)

    cfg_etf = BacktestConfig(min_names=10, n_legs="quintile")
    cfg_fx = BacktestConfig(min_names=6, n_legs="tertile")

    res_etf = run_backtest(etf_u.returns.fillna(0), d_l_etf, elig_etf, etf_u.adv, cfg_etf,
                            rebalance_dates=list(d_l_etf.index))
    res_fx = run_backtest(fx_u.returns.fillna(0), d_l_fx, elig_fx, fx_u.adv, cfg_fx,
                           rebalance_dates=list(d_l_fx.index))

    out = {
        "d_l_etf": d_l_etf, "elig_etf": elig_etf, "cfg_etf": cfg_etf, "res_etf": res_etf,
        "d_l_fx": d_l_fx, "elig_fx": elig_fx, "cfg_fx": cfg_fx, "res_fx": res_fx,
    }
    save("base", out)
    return out


def stage_redundancy():
    u = load("universes")
    base = load("base")
    screen_etf = nulls.redundancy_screen(u["etf"].returns, base["d_l_etf"], base["elig_etf"])
    twin_sharpes_etf = nulls.twin_portfolio_sharpes(u["etf"].returns, screen_etf["twin_panels"],
                                                      base["elig_etf"], base["cfg_etf"], u["etf"].adv)
    screen_fx = nulls.redundancy_screen(u["fx"].returns, base["d_l_fx"], base["elig_fx"])
    twin_sharpes_fx = nulls.twin_portfolio_sharpes(u["fx"].returns, screen_fx["twin_panels"],
                                                     base["elig_fx"], base["cfg_fx"], u["fx"].adv)
    out = {"etf": {"screen": screen_etf, "twins": twin_sharpes_etf},
           "fx": {"screen": screen_fx, "twins": twin_sharpes_fx}}
    save("redundancy", out)
    return out


def stage_estnull(n_reps=25, n_synth=2000):
    u = load("universes")
    base = load("base")
    t0 = time.time()
    null_etf = nulls.estimator_null(u["etf"].returns, n_reps=n_reps, n_synthetic_cross_sections=n_synth)
    ev_etf = nulls.evaluate_estimator_null(base["d_l_etf"], base["elig_etf"], null_etf)
    print(f"[estnull] ETF done in {time.time()-t0:.1f}s")
    t0 = time.time()
    null_fx = nulls.estimator_null(u["fx"].returns, n_reps=n_reps, n_synthetic_cross_sections=n_synth)
    ev_fx = nulls.evaluate_estimator_null(base["d_l_fx"], base["elig_fx"], null_fx)
    print(f"[estnull] FX done in {time.time()-t0:.1f}s")
    out = {"etf": ev_etf, "fx": ev_fx}
    save("estnull", out)
    return out


def stage_blockperm(n_perms=150):
    u = load("universes")
    base = load("base")
    t0 = time.time()
    bp_etf = nulls.block_permutation_test(u["etf"].returns, base["d_l_etf"], base["elig_etf"],
                                           u["etf"].adv, base["cfg_etf"], n_perms=n_perms)
    print(f"[blockperm] ETF done in {time.time()-t0:.1f}s, p={bp_etf['pvalue']:.4f}")
    t0 = time.time()
    bp_fx = nulls.block_permutation_test(u["fx"].returns, base["d_l_fx"], base["elig_fx"],
                                          u["fx"].adv, base["cfg_fx"], n_perms=n_perms)
    print(f"[blockperm] FX done in {time.time()-t0:.1f}s, p={bp_fx['pvalue']:.4f}")
    out = {"etf": bp_etf, "fx": bp_fx}
    save("blockperm", out)
    return out


def stage_noisefloor(n_reps=100):
    u = load("universes")
    base = load("base")
    t0 = time.time()
    nf_etf = nulls.noise_floor_test(u["etf"].returns, base["d_l_etf"], base["elig_etf"],
                                     u["etf"].adv, base["cfg_etf"], n_reps=n_reps)
    print(f"[noisefloor] ETF done in {time.time()-t0:.1f}s, p={nf_etf['pvalue']:.4f}")
    t0 = time.time()
    nf_fx = nulls.noise_floor_test(u["fx"].returns, base["d_l_fx"], base["elig_fx"],
                                    u["fx"].adv, base["cfg_fx"], n_reps=n_reps)
    print(f"[noisefloor] FX done in {time.time()-t0:.1f}s, p={nf_fx['pvalue']:.4f}")
    out = {"etf": nf_etf, "fx": nf_fx}
    save("noisefloor", out)
    return out


def stage_robustness():
    u = load("universes")
    t0 = time.time()
    table_etf, series_etf = robustness.robustness_sweep(u["etf"], min_names=10)
    print(f"[robustness] ETF sweep done in {time.time()-t0:.1f}s")
    t0 = time.time()
    table_fx, series_fx = robustness.robustness_sweep(u["fx"], min_names=6)
    print(f"[robustness] FX sweep done in {time.time()-t0:.1f}s")
    out = {"etf": {"table": table_etf, "series": series_etf},
           "fx": {"table": table_fx, "series": series_fx}}
    save("robustness", out)
    return out


def stage_diversify():
    u = load("universes")
    base = load("base")
    spy_ret = u["spy"].returns["SPY"].fillna(0.0)
    tsmom_ret, _ = tsmom_returns(u["etf"].returns.fillna(0), base["elig_etf"])
    div_etf_full = robustness.diversification_checks(base["res_etf"]["net_returns"], spy_ret, tsmom_ret)
    div_etf_oos = robustness.diversification_checks(
        base["res_etf"]["net_returns"][base["res_etf"]["net_returns"].index >= OOS_START],
        spy_ret, tsmom_ret,
    )
    out = {"full": div_etf_full, "oos": div_etf_oos, "tsmom_returns": tsmom_ret}
    save("diversify", out)
    return out


def is_oos_dsr(base, robustness_out):
    """Deflated Sharpe Ratio: IS trials = the 18-variant robustness sweep
    scored ONLY on the in-sample window; primary config's OOS Sharpe is the
    figure DSR-deflates against."""
    results = {}
    for uni in ["etf", "fx"]:
        net = base[f"res_{uni}"]["net_returns"]
        is_ret = net[net.index <= IS_END]
        oos_ret = net[net.index >= OOS_START]

        trial_series = robustness_out[uni]["series"]
        is_trial_sharpes = []
        for key, s in trial_series.items():
            s_is = s[s.index <= IS_END].dropna()
            if len(s_is) > 20 and s_is.std() > 0:
                is_trial_sharpes.append(s_is.mean() / s_is.std())  # per-period (daily), unannualized

        oos_r = oos_ret.dropna()
        sr_hat_daily = oos_r.mean() / oos_r.std() if oos_r.std() > 0 else np.nan
        skew = oos_r.skew() if len(oos_r) > 3 else 0.0
        kurt = oos_r.kurtosis() + 3 if len(oos_r) > 3 else 3.0  # scipy convention -> Pearson kurtosis
        dsr = metrics.deflated_sharpe_ratio(sr_hat_daily, len(oos_r), is_trial_sharpes, skew, kurt) \
            if len(is_trial_sharpes) > 2 and np.isfinite(sr_hat_daily) else np.nan

        results[uni] = {
            "is_sharpe_ann": metrics.sharpe(is_ret),
            "oos_sharpe_ann": metrics.sharpe(oos_ret),
            "oos_ann_return": metrics.ann_return(oos_ret),
            "oos_nw_tstat": metrics.newey_west_tstat(oos_ret),
            "n_is_trials": len(is_trial_sharpes),
            "dsr": dsr,
        }
    return results


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    stages = {
        "data": stage_data, "base": stage_base, "redundancy": stage_redundancy,
        "estnull": stage_estnull, "blockperm": stage_blockperm, "noisefloor": stage_noisefloor,
        "robustness": stage_robustness, "diversify": stage_diversify,
    }
    if stage == "all":
        for name, fn in stages.items():
            t0 = time.time()
            fn()
            print(f"=== stage {name} done in {time.time()-t0:.1f}s ===")
    else:
        stages[stage]()
