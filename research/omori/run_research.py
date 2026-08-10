"""Orchestrates the full IS design pipeline on the (contaminated,
freely-spent) 40-ticker primary panel:

    data -> priors -> redundancy screen (kill check) -> estimator null
    (kill check) -> kappa calibration -> p_star calibration -> primary
    backtest -> step-1 IC tests (kill check) -> twins T1/T2/T3 -> grid +
    sign stability + DSR -> three-era consistency -> results_summary.json

Every stage runs to completion regardless of interim kill-check outcomes
(matching the sibling-branch convention, e.g. irreversibility_lab's rejected
pipeline still runs end to end) so the full diagnostic record is available
for REPORT.md; the FINAL verdict is assembled from all gate results at once,
not short-circuited on the first failure.

Run with: python -m research.omori.run_research
Each stage's result is cached to output/<stage>.pkl; to resume after an
interruption, import this module and call the stage_* functions directly
(each is idempotent and cheap to re-run given its cached inputs).
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from research.omori import backtest, battery, calibrate, config, data, events, grid, metrics, nulls, priors, twins

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _cache_path(name):
    return os.path.join(OUTPUT_DIR, f"{name}.pkl")


def save(name, obj):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(_cache_path(name), "rb") as f:
        return pickle.load(f)


def has(name):
    return os.path.exists(_cache_path(name))


def stage_data():
    panel = data.load("primary")
    save("panel", panel)
    return panel


def stage_priors(panel):
    t0 = time.time()
    fits = priors.terminal_fits(panel)
    priors_dict = priors.calibrate_priors_from_fits(panel, fits)
    print(f"[priors] {len(fits)} candidates, {priors_dict['n_identified']} identified, "
          f"global p_bar={priors_dict['global']:.4f} ({time.time()-t0:.0f}s)")
    save("fits", fits)
    save("priors", priors_dict)
    return fits, priors_dict


def stage_redundancy(panel, ef, fits, priors_dict):
    t0 = time.time()
    battery_df = battery.build_battery_frame(panel, ef, fits, priors_dict, kappa=config.KAPPA_DEFAULT)
    screen = battery.redundancy_screen(battery_df)
    print(f"[redundancy] r2(p_hat~battery)={screen['r2_phat_vs_battery']:.3f} "
          f"delta_r2(half_life)={screen['delta_r2_halflife']:.3f} killed={screen['killed']} "
          f"({time.time()-t0:.0f}s)")
    save("battery_df", battery_df)
    save("redundancy_screen", screen)
    return battery_df, screen


def stage_estnull(fits):
    t0 = time.time()
    result = nulls.estimator_null_test(fits)
    print(f"[estnull] real_dispersion={result['real_dispersion']:.4f} "
          f"null_p95={result['null_p95_dispersion']:.4f} passed={result['passed']} "
          f"({time.time()-t0:.0f}s)")
    save("estnull", result)
    return result


def stage_kappa(battery_df, priors_dict):
    t0 = time.time()
    result = calibrate.calibrate_kappa(battery_df)
    kappa = result["kappa"]
    prior_lookup = battery_df["ticker"].map(lambda t: priors.prior_for(t, priors_dict))
    battery_df["p_tilde"] = (battery_df["n_pos"] * battery_df["p_hat"] + kappa * prior_lookup) / \
                              (battery_df["n_pos"] + kappa)
    print(f"[kappa] selected kappa={kappa} mse_by_kappa={result['mse_by_kappa']} ({time.time()-t0:.0f}s)")
    save("battery_df", battery_df)
    save("kappa_result", result)
    return kappa, battery_df


def stage_pstar(panel, ef, priors_dict, kappa):
    t0 = time.time()
    prelim = backtest.run_backtest(panel, priors_dict, priors_dict["global"], kappa=kappa, ef=ef)
    p_star = calibrate.calibrate_p_star(prelim.events_frame())
    print(f"[pstar] preliminary n_events={len(prelim.closed_events)} p_star={p_star:.4f} "
          f"({time.time()-t0:.0f}s)")
    save("p_star", p_star)
    return p_star


def stage_backtest(panel, ef, priors_dict, kappa, p_star):
    t0 = time.time()
    res = backtest.run_backtest(panel, priors_dict, p_star, kappa=kappa, ef=ef)
    summ = metrics.summarize(res.daily_returns, res.events_frame())
    print(f"[backtest] n_events={len(res.closed_events)} sharpe={summ['sharpe']:.3f} "
          f"total_return={res.daily_returns.sum():.4f} maxdd={summ['max_drawdown']:.4f} "
          f"({time.time()-t0:.0f}s)")
    save("primary_result", res)
    save("primary_summary", summ)
    return res, summ


def stage_ic(battery_df, res, p_star):
    t0 = time.time()
    rank_ic = nulls.rank_ic_p_tilde_halflife(battery_df)
    signed_ic = nulls.signed_ic_z_forward_return(res.events_frame(), p_star)
    print(f"[ic] rank_ic(p_tilde,half_life)={rank_ic['ic']:.4f} p={rank_ic['p_value']:.4f} | "
          f"signed_ic(Z,fwd_ret)={signed_ic['ic']:.4f} p={signed_ic['p_value']:.4f} "
          f"({time.time()-t0:.0f}s)")
    save("rank_ic", rank_ic)
    save("signed_ic", signed_ic)
    return rank_ic, signed_ic


def stage_twins(panel, ef, res):
    t0 = time.time()
    ev = res.events_frame()
    primary_sharpe = metrics.annualized_sharpe(res.daily_returns)
    out = {}
    for name, fn in [("T1", twins.t1_fixed_horizon), ("T2", twins.t2_shuffled_within_instrument),
                      ("T3", twins.t3_randomized_entry)]:
        dr, net = fn(panel, ef, ev)
        out[name] = {
            "sharpe": metrics.annualized_sharpe(dr), "total_return": float(dr.sum()),
            "mean_net_return": float(np.mean(net)) if len(net) else np.nan,
            "n_events": int(len(net)),
        }
    out["primary"] = {"sharpe": primary_sharpe, "total_return": float(res.daily_returns.sum())}
    print(f"[twins] primary_sharpe={primary_sharpe:.3f} " +
          " ".join(f"{k}_sharpe={v['sharpe']:.3f}" for k, v in out.items() if k != "primary") +
          f" ({time.time()-t0:.0f}s)")
    save("twins", out)
    return out


def stage_grid(panel, priors_dict, p_star, kappa):
    t0 = time.time()
    grid_df = grid.run_grid(panel, priors_dict, p_star, kappa=kappa)
    stability = grid.sign_stability(grid_df)
    grid_df.to_csv(os.path.join(OUTPUT_DIR, "grid_table.csv"), index=False)
    print(f"[grid] {len(grid_df)} cells, sign_stable={stability['stable']} "
          f"frac_matching={stability['frac_matching']:.2f} ({time.time()-t0:.0f}s)")
    save("grid_df", grid_df)
    save("sign_stability", stability)
    return grid_df, stability


def stage_era(panel, priors_dict, p_star, kappa):
    t0 = time.time()
    era_df = grid.era_consistency(panel, priors_dict, p_star, kappa=kappa)
    print(f"[era] \n{era_df}\n({time.time()-t0:.0f}s)")
    save("era_df", era_df)
    return era_df


def run_all():
    panel = stage_data()
    ef = events.EventFields(panel)
    fits, priors_dict = stage_priors(panel)
    battery_df, screen = stage_redundancy(panel, ef, fits, priors_dict)
    estnull = stage_estnull(fits)
    kappa, battery_df = stage_kappa(battery_df, priors_dict)
    p_star = stage_pstar(panel, ef, priors_dict, kappa)
    res, summ = stage_backtest(panel, ef, priors_dict, kappa, p_star)
    rank_ic, signed_ic = stage_ic(battery_df, res, p_star)
    twins_out = stage_twins(panel, ef, res)
    grid_df, stability = stage_grid(panel, priors_dict, p_star, kappa)
    era_df = stage_era(panel, priors_dict, p_star, kappa)
    dsr = grid.dsr_from_grid(res.daily_returns, grid_df)

    gates = {
        "redundancy_killed": screen["killed"],
        "estnull_passed": estnull["passed"],
        "rank_ic_significant": rank_ic["p_value"] < config.IC_ALPHA,
        "signed_ic_significant": signed_ic["p_value"] < config.IC_ALPHA,
        "sign_stable_over_grid": stability["stable"],
        "min_events_is": len(res.closed_events) >= config.MIN_EVENTS_IS,
        "beats_t1_net": summ["sharpe"] > twins_out["T1"]["sharpe"],
    }
    summary = {
        "n_candidates": len(fits), "n_identified": priors_dict["n_identified"],
        "kappa": kappa, "p_star": p_star,
        "redundancy_screen": screen, "estimator_null": estnull,
        "rank_ic": rank_ic, "signed_ic": signed_ic,
        "primary_summary": summ, "twins": twins_out,
        "sign_stability": stability, "dsr": dsr,
        "gates": gates,
    }
    # "all gates passed" should read TRUE only for the gates that are meant
    # to be TRUE-is-good; redundancy_killed is TRUE-is-BAD, so combine
    # explicitly rather than a blanket all(...).
    passed = (not gates["redundancy_killed"]) and gates["estnull_passed"] and \
        gates["rank_ic_significant"] and gates["signed_ic_significant"] and \
        gates["sign_stable_over_grid"] and gates["min_events_is"] and gates["beats_t1_net"]
    summary["all_gates_passed"] = bool(passed)

    with open(os.path.join(OUTPUT_DIR, "is_results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== IS SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("redundancy_screen",)}, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run_all()
