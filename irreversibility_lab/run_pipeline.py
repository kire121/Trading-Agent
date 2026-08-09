"""End-to-end pipeline: load data -> robustness grid -> IS-only config lock
-> OOS evaluation -> full validation suite -> pre-registered accept/reject
verdict -> write results/report.json.

Run: python3 -m irreversibility_lab.run_pipeline
"""

import json
import time

import numpy as np
import pandas as pd

from . import config, data, signal, strategy, backtest, variants, validation, robustness


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def select_locked_config(grid_df, sign_df, primary_estimator="hvg"):
    """IS-ONLY selection, performed before any OOS data is consulted:
    among (W, threshold) cells where the PRIMARY estimator (HVG-KL) has a
    positive IS Sharpe AND keeps the same sign in all three named
    sub-periods, pick the highest IS Sharpe. If none survive, there is no
    config to lock -- the hypothesis fails at the calibration stage itself.
    """
    primary = grid_df[grid_df.estimator == primary_estimator].copy()
    sp_cols = [f"sp_{name}_sharpe" for name in config.SUBPERIODS]
    primary["sp_all_positive"] = (primary[sp_cols] > 0).all(axis=1)
    survivors = primary[(primary["is_sharpe"] > 0) & (primary["sp_all_positive"])]
    if len(survivors) == 0:
        return None, primary
    best = survivors.sort_values("is_sharpe", ascending=False).iloc[0]
    return {"W": int(best["W"]), "threshold": float(best["threshold"]), "estimator": primary_estimator}, primary


def run(n_bootstrap=config.BOOTSTRAP_RUNS, out_path="irreversibility_lab/results/report.json"):
    t_start = time.time()
    log = []

    def _log(msg):
        elapsed = time.time() - t_start
        line = f"[{elapsed:7.1f}s] {msg}"
        print(line, flush=True)
        log.append(line)

    _log("Loading universe (12 ETFs, Yahoo chart API, cached to irreversibility_lab/data/)...")
    px = data.load_universe()
    ret = data.log_returns(px)
    px = px.loc[ret.index[0]:]
    _log(f"Data loaded: {px.shape[0]} trading days, {px.shape[1]} instruments, "
         f"{px.index.min().date()} -> {px.index.max().date()}")

    _log("Running full robustness grid (W x threshold x estimator)...")
    rows, panel_cache, bt_cache = robustness.run_full_grid(px, ret)
    grid_df = robustness.flatten_grid_rows(rows)
    grid_df.to_csv("irreversibility_lab/results/grid_full.csv", index=False)
    sign_df = robustness.sign_consistency_check(grid_df)
    sign_df.to_csv("irreversibility_lab/results/sign_consistency.csv", index=False)
    _log(f"Grid done: {len(grid_df)} configs evaluated.")

    locked, primary_grid = select_locked_config(grid_df, sign_df, primary_estimator="hvg")
    if locked is None:
        _log("REJECT AT CALIBRATION STAGE: no (W, threshold) config gives the primary "
             "HVG-KL estimator a positive in-sample (2000-2018) Sharpe with the same sign "
             "held across all three pre-registered sub-periods (2000-07 / 2008-12 / 2013-18).")
        best_row = primary_grid.sort_values("is_sharpe", ascending=False).iloc[0]
        locked = {"W": int(best_row["W"]), "threshold": float(best_row["threshold"]), "estimator": "hvg"}
        _log(f"Proceeding anyway with the best-IS-Sharpe candidate for full diagnostic "
             f"reporting (NOT a valid lock under the pre-registered protocol): {locked}")
        calibration_verdict = "REJECT_AT_CALIBRATION (no config passes sub-period sign consistency)"
    else:
        _log(f"Locked config (IS-only, 2000-2018): {locked}")
        calibration_verdict = "PASS_CALIBRATION"

    W, thr, est = locked["W"], locked["threshold"], locked["estimator"]
    z_panel = panel_cache[(W, est)]
    res = variants.irreversibility_weekly_weights(px, ret, W=W, threshold=thr, estimator=est, z_panel_daily=z_panel)
    bt = backtest.run_backtest(px, ret, res["weekly_weights"])
    perf_full = backtest.performance_summary(bt["portfolio_return"], bt["equity"], bt["turnover"])
    perf_is = robustness.slice_performance(bt["portfolio_return"], bt["turnover"], config.IS_START, config.IS_END)
    perf_oos = robustness.slice_performance(bt["portfolio_return"], bt["turnover"], config.OOS_START, None)
    _log(f"Locked-config performance -- full: Sharpe={perf_full['sharpe']:.3f}, "
         f"IS: Sharpe={perf_is['sharpe']:.3f}, OOS: Sharpe={perf_oos['sharpe']:.3f}")

    _log("Baseline (b): static 50/50 trend+meanrev mix, no switching...")
    mix = variants.static_mix_weekly_weights(px, ret)
    bt_mix = backtest.run_backtest(px, ret, mix["weekly_weights"])
    perf_mix_oos = robustness.slice_performance(bt_mix["portfolio_return"], bt_mix["turnover"], config.OOS_START, None)

    _log("Baseline (c): vol-z-score regime switching instead of irreversibility...")
    volz = variants.vol_z_weekly_weights(px, ret, threshold=thr)
    bt_volz = backtest.run_backtest(px, ret, volz["weekly_weights"])
    perf_volz_oos = robustness.slice_performance(bt_volz["portfolio_return"], bt_volz["turnover"], config.OOS_START, None)

    _log("Pure TSMOM sleeve (for the correlation hard-limit check)...")
    tsmom = variants.pure_tsmom_weekly_weights(px, ret)
    bt_tsmom = backtest.run_backtest(px, ret, tsmom["weekly_weights"])
    corr_tsmom_full = validation.tsmom_correlation(bt["portfolio_return"], bt_tsmom["portfolio_return"])
    corr_tsmom_oos = validation.tsmom_correlation(
        bt["portfolio_return"].loc[config.OOS_START:], bt_tsmom["portfolio_return"].loc[config.OOS_START:]
    )

    _log("Orthogonalizing Z against rolling vol / skew / |r|-autocorrelation...")
    resid_z, r2_by_col = validation.orthogonalize_z(res["z_weekly"], ret, res["anchors"])
    regime_orth = signal.classify_regime_panel(resid_z, upper=thr)
    trend_dir = signal.trend_direction(px).loc[res["anchors"]]
    meanrev_dir = signal.meanrev_direction(px).loc[res["anchors"]]
    direction_orth = signal.combine_direction(regime_orth, trend_dir, meanrev_dir)
    weekly_w_orth = strategy.build_weekly_weights(direction_orth, ret, res["anchors"])
    bt_orth = backtest.run_backtest(px, ret, weekly_w_orth)
    perf_orth_full = backtest.performance_summary(bt_orth["portfolio_return"], bt_orth["equity"], bt_orth["turnover"])
    perf_orth_oos = robustness.slice_performance(bt_orth["portfolio_return"], bt_orth["turnover"], config.OOS_START, None)
    _log(f"Orthogonalized-Z strategy -- full Sharpe={perf_orth_full['sharpe']:.3f}, "
         f"OOS Sharpe={perf_orth_oos['sharpe']:.3f} (R^2 of vol/skew/autocorr on Z: "
         f"{np.nanmean(list(r2_by_col.values())):.3f} avg across instruments)")

    _log(f"Stationary block-bootstrap null on the locked config's Z path "
         f"({n_bootstrap} runs, block~{config.BOOTSTRAP_BLOCK_LEN}d)... this is the slow step.")
    null_sharpes = validation.bootstrap_null_sharpe(
        px, ret, res["z_weekly"], res["anchors"], threshold=thr, n_runs=n_bootstrap
    )
    pval_full = validation.bootstrap_p_value(perf_full["sharpe"], null_sharpes)
    _log(f"Bootstrap null done: mean={np.nanmean(null_sharpes):.3f}, "
         f"std={np.nanstd(null_sharpes):.3f}, p-value(full Sharpe)={pval_full:.4f}")

    _log("Deflated Sharpe Ratio (N~80 configs, per user's explicit multiple-testing count)...")
    trial_sharpes = grid_df["is_sharpe"].tolist()
    dsr_oos = validation.deflated_sharpe(
        perf_oos["sharpe"], bt["portfolio_return"].loc[config.OOS_START:],
        trial_sharpes, n_trials_effective=config.N_CONFIGS_FOR_DSR
    )
    dsr_full = validation.deflated_sharpe(
        perf_full["sharpe"], bt["portfolio_return"], trial_sharpes, n_trials_effective=config.N_CONFIGS_FOR_DSR
    )
    _log(f"DSR (OOS): sr0={dsr_oos['sr0']}, dsr_excess={dsr_oos['dsr_excess']}, dsr_prob={dsr_oos['dsr_prob']}")

    # --- Pre-registered rejection criteria -----------------------------
    reasons = []
    if calibration_verdict.startswith("REJECT"):
        reasons.append("No (W, threshold) config keeps the primary HVG-KL estimator's Sharpe "
                        "sign consistent across all three pre-registered sub-periods (2000-07 / "
                        "2008-12 / 2013-18) using only in-sample (2000-2018) data.")
    if not (dsr_oos["dsr_excess"] is not None and not np.isnan(dsr_oos["dsr_excess"]) and dsr_oos["dsr_excess"] > 0):
        reasons.append(f"DSR (OOS) excess <= 0: observed OOS Sharpe {perf_oos['sharpe']:.3f} does not "
                        f"exceed the expected best-of-{config.N_CONFIGS_FOR_DSR} Sharpe under a "
                        f"skill-less null ({dsr_oos['sr0']:.3f}).")
    if perf_volz_oos["sharpe"] is not None and not np.isnan(perf_volz_oos["sharpe"]) and \
            perf_volz_oos["sharpe"] >= perf_oos["sharpe"]:
        reasons.append("Baseline (c) [vol-z-score regime switching] matches or beats the "
                        "irreversibility-driven strategy OOS -- irreversibility adds nothing "
                        "beyond a plain volatility-timing regime switch.")
    if corr_tsmom_full is not None and not np.isnan(corr_tsmom_full) and corr_tsmom_full > config.TSMOM_CORR_REJECT:
        reasons.append(f"Correlation vs pure TSMOM ({corr_tsmom_full:.2f}) exceeds the "
                        f"{config.TSMOM_CORR_REJECT} hard limit -- this is TSMOM in disguise.")
    sp_signs = [np.sign(v) for v in [primary_grid[(primary_grid.W == W) & (primary_grid.threshold == thr)]
                [f"sp_{name}_sharpe"].iloc[0] for name in config.SUBPERIODS]]
    if len(set(sp_signs)) > 1:
        reasons.append(f"Sign flips across sub-periods for the locked config: "
                        f"{dict(zip(config.SUBPERIODS.keys(), sp_signs))}.")

    verdict = "REJECT" if reasons else "DO NOT REJECT (survives pre-registered kill criteria)"
    _log(f"VERDICT: {verdict}")
    for r in reasons:
        _log(f"  - {r}")

    report = {
        "generated_at_days_elapsed": time.time() - t_start,
        "universe": config.UNIVERSE,
        "data_range": [str(px.index.min().date()), str(px.index.max().date())],
        "locked_config": locked,
        "calibration_verdict": calibration_verdict,
        "performance": {"full": perf_full, "is": perf_is, "oos": perf_oos},
        "baselines_oos": {
            "irreversibility_regime": perf_oos,
            "static_mix_b": perf_mix_oos,
            "vol_z_regime_c": perf_volz_oos,
        },
        "orthogonalization": {
            "avg_r2_vol_skew_autocorr_on_Z": float(np.nanmean(list(r2_by_col.values()))),
            "r2_by_instrument": r2_by_col,
            "orthogonalized_strategy_perf_full": perf_orth_full,
            "orthogonalized_strategy_perf_oos": perf_orth_oos,
        },
        "bootstrap_null": {
            "n_runs": n_bootstrap,
            "block_len_days": config.BOOTSTRAP_BLOCK_LEN,
            "null_mean_sharpe": float(np.nanmean(null_sharpes)),
            "null_std_sharpe": float(np.nanstd(null_sharpes)),
            "observed_full_sharpe": perf_full["sharpe"],
            "p_value": pval_full,
        },
        "deflated_sharpe": {"oos": dsr_oos, "full": dsr_full, "n_trials": config.N_CONFIGS_FOR_DSR},
        "tsmom_correlation": {"full": corr_tsmom_full, "oos": corr_tsmom_oos, "hard_limit": config.TSMOM_CORR_REJECT},
        "robustness_grid_summary": grid_df.to_dict(orient="records"),
        "sign_consistency": sign_df.drop(columns=["subperiod_consistent_by_estimator"]).to_dict(orient="records"),
        "verdict": verdict,
        "rejection_reasons": reasons,
        "log": log,
    }

    with open(out_path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, default=str)
    _log(f"Report written to {out_path}. Total elapsed: {time.time() - t_start:.1f}s")
    return report


if __name__ == "__main__":
    run()
