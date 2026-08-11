"""Full pre-registered pipeline, run in cost order exactly as specified
(cheapest kill first):

  1. Oracle-cap test: perfect-foresight G (matched distribution) vs base.
  2. Base-engine gate (Oeglegrinden-regeln): TSMOM sleeve must have net
     alpha IS before any overlay work.
  3. Redundancy screen: R_hat vs count-EWMA + GARCH-persistence twins.
  4. Dispersion null + episode count.
  5. Step 2: net-SR increment vs T1/T2/T3, bootstrap CI, sign consistency,
     grid sign-stability, pooled-corrected DSR.
  OOS: single locked read of 2018-> on the primary cell only, only if every
      IS gate above passed.

Writes research/smittotalet/output/is_results_summary.json (and
oos_results_summary.json if IS survives). Mirrors the staged, cached
orchestration convention used by every sibling run_research.py.
"""
import json
import os

import numpy as np
import pandas as pd

from . import backtest
from . import battery
from . import config
from . import data
from . import episodes
from . import events as events_mod
from . import fetch_data
from . import grid as grid_mod
from . import metrics
from . import nulls
from . import signal as signal_mod
from . import tsmom
from . import twins as twins_mod


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def ensure_data():
    field_path = os.path.join(config.DATA_DIR, "primary_close.csv")
    if not os.path.exists(field_path):
        fetch_data.build_panel_csvs()


def run(apply_costs: bool = True, dispersion_draws: int = 200) -> dict:
    ensure_data()
    panel = data.load_primary_panel()
    panel = data.restrict(panel, start=config.HISTORY_START)

    results: dict = {"gates": {}, "config": {"grid": [c.__dict__ for c in config.GRID]}}

    # ---- Infra: solve k IS, build full-sample base book + default cell ----
    # (Building the base book and G_t is required plumbing for the oracle-cap
    # gate below, which is evaluated "netto-SR over basen" -- but it is NOT
    # itself a gate. Gate order below follows the brief's explicit
    # "kostnadsordning" literally: orakel-tak (1) BEFORE Steg-1 basmotor (2).)
    k = tsmom.solve_k_for_target_vol(panel, config.IS_START, config.IS_END)
    base_weights, base_returns = tsmom.build_base_book(panel, k, apply_costs=apply_costs)
    base_is = base_returns.loc[config.IS_START:config.IS_END]
    base_is_sharpe = metrics.sharpe(base_is)
    results["k"] = k
    results["base_is_sharpe"] = base_is_sharpe

    cell = config.DEFAULT_CELL
    returns = panel.simple_returns()
    events, x_t = events_mod.build(returns, q=cell.q)
    r_hat, g_t, lam_t = signal_mod.build(x_t, q=cell.q, tau=cell.tau, kappa=cell.kappa)
    tilted_weights, tilted_returns, g_applied = backtest.apply_overlay(panel, base_weights, g_t, apply_costs)

    # ---- Gate 1: oracle cap (cheapest, first) -------------------------------
    # IS-only, like every other gate below -- OOS (2018-> ) stays locked until
    # every IS gate has passed.
    from . import scheduling
    g_weekly = scheduling.week_end_values(g_t)
    oracle = backtest.oracle_cap_test(
        base_returns.loc[config.IS_START:config.IS_END], g_weekly.loc[config.IS_START:config.IS_END]
    )
    results["gates"]["1_oracle_cap"] = oracle
    if not oracle["passes"]:
        results["verdict"] = "DOD (Steg 1: orakeltak -- reglageklassen har inte tillrackligt tak har)"
        _write("is_results_summary.json", results)
        return results

    # ---- Gate 2: base-engine alpha (Oeglegrinden-regeln) --------------------
    gate2 = bool(np.isfinite(base_is_sharpe) and base_is_sharpe > config.BASE_ENGINE_MIN_IS_SHARPE)
    results["gates"]["2_base_engine_alpha"] = {"is_sharpe": base_is_sharpe, "passes": gate2}
    if not gate2:
        results["verdict"] = "DOD (Steg 2: basmotorn saknar netto-alfa IS -- Oeglegrinden-regeln)"
        _write("is_results_summary.json", results)
        return results

    # ---- Gate 3: redundancy screen (IS-only) --------------------------------
    batt = battery.build_battery_frame(
        r_hat.loc[config.IS_START:config.IS_END], x_t.loc[config.IS_START:config.IS_END],
        returns.loc[config.IS_START:config.IS_END], cell.tau,
    )
    redundancy = battery.redundancy_screen(batt)
    results["gates"]["3_redundancy"] = redundancy
    if redundancy["killed"]:
        results["verdict"] = "DOD (Steg 3: redundant med count-EWMA/GARCH-persistens)"
        _write("is_results_summary.json", results)
        return results

    # ---- Gate 4: dispersion null + episode count ----------------------------
    r_hat_is = r_hat.loc[config.IS_START:config.IS_END]
    x_t_is = x_t.loc[config.IS_START:config.IS_END]
    real_dispersion = float(r_hat_is.std(skipna=True))
    null_dispersion = nulls.r_hat_dispersion_null(x_t_is, cell.q, cell.tau, cell.kappa,
                                                   n_draws=dispersion_draws)
    null_p95 = float(np.quantile(null_dispersion, 0.95))
    dispersion_pass = bool(np.isfinite(real_dispersion) and real_dispersion > null_p95)

    g_weekly_is = g_weekly.loc[config.IS_START:config.IS_END]
    ep = episodes.episode_gate(r_hat_is, g_weekly_is)
    gate4 = bool(dispersion_pass and ep["episode_count_pass"] and ep["binding_share_pass"])
    results["gates"]["4_dispersion_episodes"] = {
        "real_dispersion": real_dispersion, "null_p95_dispersion": null_p95,
        "dispersion_pass": dispersion_pass, **ep, "passes": gate4,
    }
    if not gate4:
        results["verdict"] = "DOD (Steg 4: dispersionsnull eller episodfattigdom)"
        _write("is_results_summary.json", results)
        return results

    # ---- Step 5: twins, grid, DSR ------------------------------------------
    t1 = twins_mod.voltarget_twin(base_returns, g_t, cell.kappa)
    t2 = twins_mod.count_ewma_twin(x_t, g_t, cell.tau, cell.kappa)
    t3 = twins_mod.broadcast_twin(returns, g_t, cell.q, cell.tau, cell.kappa)

    twin_results = {}
    for name, twin_g in (("T1_voltarget", t1), ("T2_count_ewma", t2), ("T3_broadcast", t3)):
        _, twin_returns, _ = backtest.apply_overlay(panel, base_weights, twin_g, apply_costs)
        primary_is = tilted_returns.loc[config.IS_START:config.IS_END]
        twin_is = twin_returns.loc[config.IS_START:config.IS_END]
        diff = (primary_is - twin_is).dropna()
        lo, hi, _ = nulls.block_bootstrap_sharpe_ci(diff)
        increment = metrics.sharpe(primary_is) - metrics.sharpe(twin_is)
        sign_cons = metrics.sub_period_sign_consistency(diff, config.IS_START, config.IS_END)
        twin_results[name] = {
            "twin_sharpe_is": metrics.sharpe(twin_is),
            "primary_sharpe_is": metrics.sharpe(primary_is),
            "increment": increment,
            "increment_meets_bar": bool(np.isfinite(increment) and increment >= config.STEP2_MIN_SR_INCREMENT),
            "diff_bootstrap_ci90": [lo, hi],
            "sign_consistency": sign_cons["consistency"],
            "sign_consistency_meets_bar": bool(
                np.isfinite(sign_cons["consistency"]) and sign_cons["consistency"] >= config.STEP2_MIN_SIGN_CONSISTENCY
            ),
        }
    results["step2_twins"] = twin_results

    grid_df = grid_mod.run_grid(panel, base_weights, base_returns, config.IS_START, config.IS_END,
                                 config.OOS_START, config.OOS_END, apply_costs)
    sign_stab = grid_mod.sign_stability(grid_df)
    dsr = grid_mod.dsr_from_grid(tilted_returns.loc[config.IS_START:config.IS_END], grid_df)

    grid_sharpes = (grid_df["is_sharpe"].dropna() / np.sqrt(config.TRADING_DAYS_YEAR)).to_numpy()
    pooled_trials = np.tile(grid_sharpes, config.N_EFFECTIVE_SURFACE_READS) if len(grid_sharpes) else grid_sharpes
    sr_hat_period = metrics.sharpe(tilted_returns.loc[config.IS_START:config.IS_END]) / np.sqrt(config.TRADING_DAYS_YEAR)
    pooled_dsr = metrics.deflated_sharpe_ratio(
        sr_hat_period, tilted_returns.loc[config.IS_START:config.IS_END].dropna().shape[0], pooled_trials
    ) if len(pooled_trials) >= 2 else {"dsr_prob": np.nan, "dsr_excess": np.nan, "expected_max_sr": np.nan}

    results["grid"] = grid_df.to_dict(orient="records")
    results["grid_sign_stability"] = sign_stab
    results["dsr_own_grid"] = dsr
    results["dsr_pooled_surface_corrected"] = pooled_dsr

    kernel_robust = grid_mod.kernel_robustness_check(panel, base_weights, base_returns, cell,
                                                       config.IS_START, config.IS_END, apply_costs=apply_costs)
    results["kernel_robustness"] = kernel_robust

    step2_pass = all(
        v["increment_meets_bar"] and v["sign_consistency_meets_bar"] and v["diff_bootstrap_ci90"][0] > 0
        for v in twin_results.values()
    ) and bool(np.isfinite(pooled_dsr["dsr_excess"]) and pooled_dsr["dsr_excess"] >= config.DSR_OOS_KILL)
    results["gates"]["5_step2"] = {"passes": step2_pass}

    if not step2_pass:
        results["verdict"] = "DOD (Steg 5: klarar inte inkrement/CI/teckenkonsistens/DSR mot samtliga tvillingar)"
        _write("is_results_summary.json", results)
        return results

    results["verdict"] = "OVERLEVER STEG 1-5 IS -- laser OOS (2018- )"
    _write("is_results_summary.json", results)

    # ---- Locked OOS read (only if every IS gate passed) --------------------
    oos_results = _run_oos(panel, base_weights, base_returns, cell, apply_costs)
    _write("oos_results_summary.json", oos_results)
    results["oos"] = oos_results
    return results


def _run_oos(panel, base_weights, base_returns, cell, apply_costs):
    returns = panel.simple_returns()
    events, x_t = events_mod.build(returns, q=cell.q)
    r_hat, g_t, lam_t = signal_mod.build(x_t, q=cell.q, tau=cell.tau, kappa=cell.kappa)
    _, tilted_returns, _ = backtest.apply_overlay(panel, base_weights, g_t, apply_costs)

    oos_primary = tilted_returns.loc[config.OOS_START:config.OOS_END]
    oos_base = base_returns.loc[config.OOS_START:config.OOS_END]
    return {
        "oos_start": config.OOS_START, "oos_end": config.OOS_END,
        "base_sharpe_oos": metrics.sharpe(oos_base),
        "primary_sharpe_oos": metrics.sharpe(oos_primary),
        "increment_oos": metrics.sharpe(oos_primary) - metrics.sharpe(oos_base),
        "n_episodes_oos": len(episodes.superkritiska_episodes(r_hat.loc[config.OOS_START:config.OOS_END])),
    }


def _write(filename, obj):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, filename), "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


if __name__ == "__main__":
    out = run()
    print(json.dumps({"verdict": out.get("verdict")}, indent=2))
