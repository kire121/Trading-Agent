"""Steg 0b A/B-separation (spec SS9: estimator-null demonstrated on synthetic
data BEFORE the threshold lock -- "mallkrav B, Timglaset"). Two checks:

  1. theta=0 (no true s->r link): across 200 independent sims, the
     block-permutation-null exceedance rate (fraction of sims where the
     observed IC beats the null's own p95) should be 5%+-3%.
  2. planted theta (true weekly IC ~= 0.03): observed IC >= 0.02 AND >
     null's own p99. Run once with a constant-sign theta and once with a
     mixed-sign (episode-alternating) theta (spec: "mixed-sign-plantering
     ingar (Runraden-mallkrav 3)" -- see AVVIKELSER.md for why this is a
     disclosed de-facto interpretation, not a citable template: no such
     numbered list exists in Runraden's own repo history, the same
     citation-integrity gap research/smittotalet/README.md already
     documents for a different "Runraden mallkrav" reference).

DECLARED SCALE (AVVIKELSER.md, computational necessity, affects ONLY this
internal methodology self-check -- never the real Steg 0a-5 pipeline, which
always runs on the full 40-ticker/22-year real panel):
  - 10 synthetic assets x 1200 days per outer sim (vs. the 40x5000 used for
    band derivation), 200 outer sims.
  - 50 inner block-permutation-null draws per outer sim (vs. the
    spec-mandated 500 used verbatim for the REAL K1.3/T3 tests on actual
    data) -- 2% p-value resolution, sufficient for a 5%+-3% tolerance check.
  - demean=None (raw s, no FE-demean) for this internal check specifically:
    spec's own note "nollen bevarar FE per konstruktion -- pass i Steg 1
    bevisar att tidsvariationen bar informationen" reads as Steg 0b
    validating the CORE estimator (rolling t-stat of s), with Steg 1 (on
    the primary/demeaned cell, on real data) validating the FE-demean's
    incremental value -- also avoids a 252-day demean burn-in that would
    leave too little history at this reduced synthetic scale.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import config
from . import intrabar
from . import nulls
from . import signal
from . import synth

N_OUTER_SIMS = 200
N_ASSETS = 10
N_DAYS = 1200
N_INNER_NULL_DRAWS = 50
K = config.K_PRIMARY
Z_STAR = config.Z_STAR_PRIMARY
BLOCK_LEN = config.STEG1_NULL_BLOCK_DAYS
FWD_HORIZON_DAYS = 5


def _weekly_ic(g_wide: pd.DataFrame, fwd_ret_wide: pd.DataFrame) -> float:
    decision_dates = signal.weekly_decision_dates(g_wide.index)
    decision_dates = decision_dates[decision_dates.isin(g_wide.index)]
    g_at = g_wide.loc[decision_dates]
    fwd_ret = fwd_ret_wide.reindex(decision_dates)
    g_flat = g_at.stack()
    r_flat = fwd_ret.stack()
    common = pd.concat([g_flat, r_flat], axis=1, keys=["g", "r"]).dropna()
    if len(common) < 20:
        return np.nan
    return float(scipy_stats.spearmanr(common["g"], common["r"]).correlation)


def _one_sim_ic_and_null(seed: int, theta: float, mixed_sign: bool,
                          n_null_draws: int = N_INNER_NULL_DRAWS) -> dict:
    base_panel = synth.simulate_panel(n_assets=N_ASSETS, n_days=N_DAYS, seed=seed)
    if theta != 0.0:
        s_panel = synth.plant_effect(base_panel, seed=seed, theta=theta, mixed_sign=mixed_sign)
        z = s_panel["z"]
    else:
        s_panel = base_panel
        z = pd.DataFrame(0.0, index=base_panel["dates"], columns=base_panel["assets"])

    mi = synth.panel_to_multiindex(s_panel)
    shadow = intrabar.shadow_stats(mi["O"], mi["H"], mi["L"], mi["C"])
    s = shadow["s"]
    S = intrabar.rolling_tstat(s, K, min_valid=config.K_EFF_MIN_FRACTION)
    g = signal.compute_g(S, Z_STAR)
    g_wide = g.unstack("ticker")

    # Forward-return TARGET is constructed directly from the UNBIASED base
    # price path plus theta*z_t -- see synth.plant_effect's docstring for
    # why the bias is not baked into the simulated price path itself.
    raw_fwd = base_panel["C"].pct_change(FWD_HORIZON_DAYS).shift(-FWD_HORIZON_DAYS)
    biased_fwd = raw_fwd + theta * z

    observed_ic = _weekly_ic(g_wide, biased_fwd)

    rng = np.random.default_rng(seed + 10_000_019)
    null_ics = np.empty(n_null_draws)
    for i in range(n_null_draws):
        s_perm = nulls.block_permute_within_ticker(s, BLOCK_LEN, rng)
        S_perm = intrabar.rolling_tstat(s_perm, K, min_valid=config.K_EFF_MIN_FRACTION)
        g_perm = signal.compute_g(S_perm, Z_STAR)
        null_ics[i] = _weekly_ic(g_perm.unstack("ticker"), biased_fwd)

    null_ics = null_ics[np.isfinite(null_ics)]
    null_p95 = float(np.percentile(null_ics, 95)) if len(null_ics) else np.nan
    null_p99 = float(np.percentile(null_ics, 99)) if len(null_ics) else np.nan
    return {"observed_ic": observed_ic, "null_p95": null_p95, "null_p99": null_p99,
            "exceeds_p95": bool(np.isfinite(observed_ic) and np.isfinite(null_p95) and observed_ic > null_p95)}


def run_null_calibration(n_outer_sims: int = N_OUTER_SIMS, seed_base: int = config.GLOBAL_SEED) -> dict:
    """theta=0: false-positive (exceedance) rate across n_outer_sims should be 5%+-3%."""
    exceed = 0
    ics = []
    n_valid = 0
    for i in range(n_outer_sims):
        r = _one_sim_ic_and_null(seed_base + i, theta=0.0, mixed_sign=False)
        if np.isfinite(r["observed_ic"]):
            n_valid += 1
            ics.append(r["observed_ic"])
            if r["exceeds_p95"]:
                exceed += 1
    rate = exceed / n_valid if n_valid else float("nan")
    target = config.STEG0B_AB_NULL_EXCEEDANCE_TARGET
    tol = config.STEG0B_AB_NULL_EXCEEDANCE_TOLERANCE
    passes = bool(np.isfinite(rate) and (target - tol) <= rate <= (target + tol))
    return {"n_sims": n_outer_sims, "n_valid": n_valid, "exceedance_rate": rate,
            "target": target, "tolerance": tol, "passes": passes, "mean_ic": float(np.mean(ics)) if ics else None}


def run_planted_effect_check(theta: float, mixed_sign: bool, seed: int) -> dict:
    r = _one_sim_ic_and_null(seed, theta=theta, mixed_sign=mixed_sign)
    ic_min = config.STEG0B_AB_PLANTED_MEASURED_IC_MIN
    passes = bool(np.isfinite(r["observed_ic"]) and r["observed_ic"] >= ic_min
                  and np.isfinite(r["null_p99"]) and r["observed_ic"] > r["null_p99"])
    return {**r, "theta": theta, "mixed_sign": mixed_sign, "ic_min_required": ic_min, "passes": passes}


def calibrate_theta_for_target_ic(target_ic: float, mixed_sign: bool, seed: int,
                                   theta_grid=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02),
                                   n_calib_seeds: int = 5) -> dict:
    """Small deterministic grid search (no optimizer dependency, seed-
    averaged since a single sim's pooled IC is noisy at this reduced scale
    -- see AVVIKELSER.md) for the SMALLEST theta whose average observed IC
    clears `target_ic` with headroom -- used to pick a single theta value
    for the planted-effect checks. The spec pins the TARGET true weekly IC
    (~0.03) and the pass bar (observed IC >= 0.02), not theta itself (an
    internal generator parameter, not a spec-visible quantity)."""
    results = []
    for theta in theta_grid:
        ics = []
        for i in range(n_calib_seeds):
            r = _one_sim_ic_and_null(seed + i, theta=theta, mixed_sign=mixed_sign, n_null_draws=5)
            if np.isfinite(r["observed_ic"]):
                ics.append(r["observed_ic"])
        mean_ic = float(np.mean(ics)) if ics else np.nan
        results.append((theta, mean_ic))
        if np.isfinite(mean_ic) and mean_ic >= target_ic:
            return {"theta": theta, "achieved_ic": mean_ic, "grid": results}
    # nothing cleared the bar -> take the theta with the highest achieved IC
    valid = [(t, ic) for t, ic in results if np.isfinite(ic)]
    if not valid:
        return {"theta": theta_grid[-1], "achieved_ic": None, "grid": results}
    best = max(valid, key=lambda x: x[1])
    return {"theta": best[0], "achieved_ic": best[1], "grid": results}


def run_ab_separation(seed_base: int = config.GLOBAL_SEED) -> dict:
    calib_single = calibrate_theta_for_target_ic(config.STEG0B_AB_PLANTED_TRUE_IC, mixed_sign=False,
                                                  seed=seed_base + 900_001)
    calib_mixed = calibrate_theta_for_target_ic(config.STEG0B_AB_PLANTED_TRUE_IC, mixed_sign=True,
                                                 seed=seed_base + 900_002)

    null_calibration = run_null_calibration(seed_base=seed_base)
    planted_single = run_planted_effect_check(calib_single["theta"], mixed_sign=False, seed=seed_base + 900_101)
    planted_mixed = run_planted_effect_check(calib_mixed["theta"], mixed_sign=True, seed=seed_base + 900_102)

    passes = bool(null_calibration["passes"] and planted_single["passes"] and planted_mixed["passes"])
    return {
        "null_calibration_theta0": null_calibration,
        "planted_single_sign": planted_single,
        "planted_mixed_sign": planted_mixed,
        "theta_calibration": {"single_sign": calib_single, "mixed_sign": calib_mixed},
        "passes": passes,
    }
