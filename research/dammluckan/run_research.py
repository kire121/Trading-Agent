"""
Dammluckan -- full research pipeline orchestration.

Stages (run in order, each cached to output/<stage>.pkl so re-runs are
resumable): calibration -> gate0 (event count + concentration) -> gate1
(estimator-level IC null test + redundancy screen) -> primary (IS/OOS-1
backtest of the primary cell, Donchian twin, anti-twin, null batteries #3/#4,
DSR, sub-period sign consistency, PnL concentration, TSMOM correlation) ->
grid (27-variant DSR deflation + neighborhood check) -> oos2 (secondary,
16-country-ETF confirmation surface, touched exactly once, using every
parameter frozen from the primary-universe IS calibration) -> verdict.
"""
import os
import pickle
import time
import sys

import numpy as np
import pandas as pd

from . import config
from . import data
from . import signal as signal_mod
from . import backtest
from . import twins
from . import battery
from . import robustness
from . import metrics
from . import grid as grid_mod
from . import tsmom
from . import run_calibration

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")


def _cache_path(stage):
    return os.path.join(OUT_DIR, f"stage_{stage}.pkl")


def _save(stage, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(_cache_path(stage), "wb") as f:
        pickle.dump(obj, f)


def _load(stage):
    path = _cache_path(stage)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _log(t0, msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}")
    sys.stdout.flush()


PRIMARY_CELL = dict(n=config.N_DEFAULT, pctl=config.THETA_PCTL_DEFAULT, h=config.H_DEFAULT)


def stage_gate0(panel, calib, t0, force=False):
    if not force and (cached := _load("gate0")) is not None:
        return cached
    n, pctl, h = PRIMARY_CELL["n"], PRIMARY_CELL["pctl"], PRIMARY_CELL["h"]
    c = calib["c"][n]
    sig = signal_mod.build_signal(panel, n=n, c=c)
    th_high, th_low = calib["theta_high"][n][pctl], calib["theta_low"][n][pctl]
    candidates = backtest.generate_candidates(panel, sig, th_high, th_low, h)
    long_candidates = [c_ for c_ in candidates if c_.direction == 1]
    short_candidates = [c_ for c_ in candidates if c_.direction == -1]
    conc_long = metrics.event_concentration(long_candidates, panel.tickers)
    conc_short = metrics.event_concentration(short_candidates, panel.tickers)
    result = {
        "n_long": len(long_candidates), "n_short": len(short_candidates),
        "conc_long": conc_long, "conc_short": conc_short,
        "pass_count": len(long_candidates) >= config.MIN_EVENTS_PER_SIDE
                      and len(short_candidates) >= config.MIN_EVENTS_PER_SIDE,
        "pass_concentration": conc_long["max_share"] <= config.MAX_SINGLE_ASSET_EVENT_SHARE
                               and conc_short["max_share"] <= config.MAX_SINGLE_ASSET_EVENT_SHARE,
    }
    _log(t0, f"gate0: n_long={result['n_long']} n_short={result['n_short']} "
             f"max_share_long={conc_long['max_share']:.3f} ({conc_long['max_ticker']}) "
             f"max_share_short={conc_short['max_share']:.3f} ({conc_short['max_ticker']}) "
             f"pass_count={result['pass_count']} pass_conc={result['pass_concentration']}")
    _save("gate0", result)
    return result


def stage_gate1(panel, calib, t0, n_draws=config.N_BLOCK_DRAWS, force=False):
    if not force and (cached := _load("gate1")) is not None:
        return cached
    n, h = PRIMARY_CELL["n"], PRIMARY_CELL["h"]
    c = calib["c"][n]
    sig = signal_mod.build_signal(panel, n=n, c=c)
    ic_res = robustness.event_level_ic(panel, sig, h, side="both")
    _log(t0, f"gate1: real pooled event-level IC = {ic_res['ic']:.4f} (n_events={ic_res['n_events']})")
    null_res = robustness.ic_null_test(panel, sig, h, n_draws=n_draws, seed=7)
    _log(t0, f"gate1: IC null test p={null_res['p_value']:.4f} over {null_res['n_draws']} draws")
    redund = robustness.redundancy_regression(panel, sig, h)
    _log(t0, f"gate1: redundancy pooled R^2={redund['r2']:.4f} (n_obs={redund['n_obs']})")
    result = {
        "ic": ic_res["ic"], "n_events": ic_res["n_events"], "null_test": null_res,
        "redundancy": redund,
        "pass_ic": bool(np.isfinite(null_res["p_value"]) and null_res["p_value"] < config.IC_P_KILL
                        and abs(ic_res["ic"]) >= config.IC_ABS_KILL),
        "pass_redundancy": bool(np.isfinite(redund["r2"]) and redund["r2"] <= 0.5),
    }
    _save("gate1", result)
    return result


def stage_primary(panel, calib, t0, n_draws=config.N_BLOCK_DRAWS, force=False):
    if not force and (cached := _load("primary")) is not None:
        return cached
    n, pctl, h = PRIMARY_CELL["n"], PRIMARY_CELL["pctl"], PRIMARY_CELL["h"]
    c = calib["c"][n]
    sig = signal_mod.build_signal(panel, n=n, c=c)
    th_high, th_low = calib["theta_high"][n][pctl], calib["theta_low"][n][pctl]
    anti_th_high, anti_th_low = calib["anti_theta_high"][n][pctl], calib["anti_theta_low"][n][pctl]

    res = backtest.run_is_oos(panel, sig, th_high, th_low, h)
    _log(t0, f"primary: k={res['k']:.5f} achieved_IS_vol={res['achieved_vol_is']:.4f} "
             f"n_trades={len(res['trades'])} IS_sharpe={metrics.sharpe(res['returns_is']):.3f} "
             f"OOS1_sharpe={metrics.sharpe(res['returns_oos1']):.3f}")

    donchian = twins.run_donchian_twin(panel, sig, h)
    _log(t0, f"primary: Donchian twin n_trades={len(donchian['trades'])} "
             f"IS_sharpe={metrics.sharpe(donchian['returns_is']):.3f} "
             f"OOS1_sharpe={metrics.sharpe(donchian['returns_oos1']):.3f}")

    anti = twins.run_anti_twin(panel, sig, anti_th_high, anti_th_low, h)
    _log(t0, f"primary: anti-twin n_trades={len(anti['trades'])} "
             f"IS_sharpe={metrics.sharpe(anti['returns_is']):.3f} "
             f"OOS1_sharpe={metrics.sharpe(anti['returns_oos1']):.3f}")

    null3 = battery.block_permuted_returns_null(panel, n, c, th_high, th_low, h, res["k"], n_draws=n_draws, seed=11)
    _log(t0, f"primary: null#3 (block-permuted returns) {null3['n_draws']} draws done")

    null4 = battery.randomized_entry_null(panel, sig, res["admitted"], h, res["k"], n_draws=n_draws, seed=13)
    _log(t0, f"primary: null#4 (randomized entry) {null4['n_draws']} draws done")

    tsmom_ret = tsmom.tsmom_returns(panel)
    common_idx = res["returns_full"].dropna().index.intersection(tsmom_ret.dropna().index)
    tsmom_corr = res["returns_full"].reindex(common_idx).corr(tsmom_ret.reindex(common_idx))
    tsmom_corr_oos1 = (res["returns_oos1"].dropna().index.intersection(tsmom_ret.dropna().index))
    tsmom_corr_oos1_val = res["returns_oos1"].reindex(tsmom_corr_oos1).corr(tsmom_ret.reindex(tsmom_corr_oos1))
    _log(t0, f"primary: corr(TSMOM) full={tsmom_corr:.3f} OOS1={tsmom_corr_oos1_val:.3f}")

    sign_full = metrics.sub_period_sign_consistency(res["returns_full"], config.IS_START, config.OOS1_END)
    concentration_full = metrics.pnl_quarter_concentration(res["returns_full"])
    _log(t0, f"primary: sub-period sign inconsistency={sign_full['n_inconsistent']}/4, "
             f"max-quarter PnL share={concentration_full:.3f}")

    result = {
        "primary": res, "donchian": donchian, "anti_twin": anti, "null3": null3, "null4": null4,
        "tsmom_corr_full": tsmom_corr, "tsmom_corr_oos1": tsmom_corr_oos1_val,
        "sign_consistency": sign_full, "pnl_quarter_concentration": concentration_full,
        "theta_high": th_high, "theta_low": th_low, "k": res["k"], "n": n, "c": c, "h": h, "pctl": pctl,
    }
    _save("primary", result)
    return result


def stage_grid(panel, calib, t0, force=False):
    if not force and (cached := _load("grid")) is not None:
        return cached
    res = grid_mod.run_grid(panel, calib)
    primary_cell_id = f"n{PRIMARY_CELL['n']}_p{PRIMARY_CELL['pctl']}_h{PRIMARY_CELL['h']}"
    dsr_is = grid_mod.dsr_from_grid(res["table"], res["returns_by_cell"], primary_cell_id, period="is")
    dsr_oos = grid_mod.dsr_from_grid(res["table"], res["returns_by_cell"], primary_cell_id, period="oos1")
    neighborhood = grid_mod.neighborhood_isolation_check(res["table"], primary_cell_id, sharpe_col="is_sharpe")
    neighborhood_oos = grid_mod.neighborhood_isolation_check(res["table"], primary_cell_id, sharpe_col="oos1_sharpe")
    _log(t0, f"grid: DSR(IS) prob={dsr_is['dsr_prob']:.4f} excess={dsr_is['dsr_excess']:.4f}")
    _log(t0, f"grid: DSR(OOS1) prob={dsr_oos['dsr_prob']:.4f} excess={dsr_oos['dsr_excess']:.4f}")
    _log(t0, f"grid: neighborhood sign-majority IS={neighborhood['frac_same_sign']:.2f} "
             f"OOS1={neighborhood_oos['frac_same_sign']:.2f}")
    result = {"table": res["table"], "dsr_is": dsr_is, "dsr_oos1": dsr_oos,
              "neighborhood_is": neighborhood, "neighborhood_oos1": neighborhood_oos,
              "primary_cell_id": primary_cell_id}
    _save("grid", result)
    return result


def stage_oos2(calib, primary_result, t0, force=False):
    """Single confirmation run on the 16-country-ETF surface, using every
    parameter frozen from the primary-universe IS calibration: n, c, theta_i
    (recalibrated PER-ASSET on THIS universe's own IS window -- theta_i is
    explicitly an asset-specific threshold and the country ETFs are
    different assets, so a literal transplant of the primary universe's
    per-ticker theta_i values would be a category error; n, c, h, and k are
    transplanted verbatim, unchanged). Run exactly once."""
    if not force and (cached := _load("oos2")) is not None:
        return cached
    panel2 = data.load_secondary_panel()
    n, pctl, h = PRIMARY_CELL["n"], PRIMARY_CELL["pctl"], PRIMARY_CELL["h"]
    c = calib["c"][n]
    sig2 = signal_mod.build_signal(panel2, n=n, c=c)

    from . import signal as signal_mod2
    th_high2 = signal_mod2.calibrate_theta_multi(panel2, n=n, c=c, pctls=[pctl], side="high",
                                                  n_draws=config.N_THETA_CALIB_DRAWS, seed=500)[pctl]
    th_low2 = signal_mod2.calibrate_theta_multi(panel2, n=n, c=c, pctls=[pctl], side="low",
                                                 n_draws=config.N_THETA_CALIB_DRAWS, seed=600)[pctl]

    candidates = backtest.generate_candidates(panel2, sig2, th_high2, th_low2, h)
    long_c = [c_ for c_ in candidates if c_.direction == 1]
    short_c = [c_ for c_ in candidates if c_.direction == -1]

    k_frozen = primary_result["k"]
    admitted = backtest.admit_candidates(candidates)
    trades = backtest.assign_weights(panel2, admitted, k_frozen)
    rets = backtest.daily_returns(panel2, trades)

    donchian2 = twins.run_donchian_twin(panel2, sig2, h)

    result = {
        "n_long": len(long_c), "n_short": len(short_c),
        "conc_long": metrics.event_concentration(long_c, panel2.tickers),
        "conc_short": metrics.event_concentration(short_c, panel2.tickers),
        "n_trades": len(trades), "returns": rets,
        "sharpe_full": metrics.sharpe(rets), "ann_return": metrics.ann_return(rets),
        "ann_vol": metrics.ann_vol(rets),
        "donchian_sharpe": metrics.sharpe(donchian2["returns_full"]),
        "theta_high": th_high2, "theta_low": th_low2,
    }
    _log(t0, f"oos2: n_long={result['n_long']} n_short={result['n_short']} n_trades={result['n_trades']} "
             f"sharpe={result['sharpe_full']:.3f} donchian_sharpe={result['donchian_sharpe']:.3f}")
    _save("oos2", result)
    return result


def run(force=False, n_draws=config.N_BLOCK_DRAWS):
    t0 = time.time()
    calib = run_calibration.run()
    _log(t0, "calibration loaded")
    panel = data.load_primary_panel()

    gate0 = stage_gate0(panel, calib, t0, force=force)
    gate1 = stage_gate1(panel, calib, t0, n_draws=n_draws, force=force)
    primary = stage_primary(panel, calib, t0, n_draws=n_draws, force=force)
    grid_res = stage_grid(panel, calib, t0, force=force)
    oos2 = stage_oos2(calib, primary["primary"], t0, force=force)

    _log(t0, "pipeline complete")
    return {"gate0": gate0, "gate1": gate1, "primary": primary, "grid": grid_res, "oos2": oos2}


if __name__ == "__main__":
    run(force=True)
