"""Fast-exit ladder, spec §8: Steg 0a -> Steg 7, exact numeric criteria,
executed strictly in order. A failed step aborts the ladder immediately
(rule 4); every step that DID run gets a full, honest record regardless of
its own verdict (assertions are never conditioned away, rule 5).

Design choices not fully pinned by the spec's prose (battery composition
for Steg 3a, the "neighbor cell" definition for Steg 5, the DSR pass
criterion for Steg 6, the additive oracle composition in oracle.py, the
standardized-forward-return definition for Steg 2/3b) are DECLARED inline
with a one-line rationale, house convention (docs/INSTRUKTION.md avsnitt 7),
and logged in full in AVVIKELSER.md -- none of them change what a number
MEANS (contrast: DSR formula family, permutation vs bootstrap, both of
which the spec resolves unambiguously elsewhere), so none rose to a rule-1
STOP; see the top-level session report for the one genuine STOP-adjacent
gap (the "repo-standard EWMA vol" claim in config.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lib.bootstrap import (
    circular_block_bootstrap_columns,
    stationary_block_bootstrap_indices,
)
from lib.metrics import deflated_sharpe_ratio
from lib.twins import twin_is_alive

from . import baseengine
from . import config
from . import metrics_ext
from . import opclock
from . import oracle
from . import oos_guard
from . import scheduling
from . import twins as twins_mod


def twin_liveness(score_frame: pd.DataFrame) -> dict:
    """Adapts a (weeks x tickers) score panel into lib.twins.twin_is_alive's
    expected long (score, direction) frame -- pooled across tickers/weeks.
    Unconditional per rule 5 / spec §1.2.7: computed and recorded for every
    twin that runs, never filtered based on outcome."""
    pooled_score = score_frame.stack()
    df = pooled_score.rename("score").to_frame()
    df["direction"] = np.sign(df["score"])
    return twin_is_alive(df)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def weekly_compound_panel(returns: pd.DataFrame) -> pd.DataFrame:
    week_period = returns.index.to_period(f"W-{config.REBALANCE_WEEKDAY}")
    return (1.0 + returns.fillna(0.0)).groupby(week_period).prod() - 1.0


def annualized_sharpe_weekly(weekly_returns: pd.Series) -> float:
    r = weekly_returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(config.WEEKS_YEAR))


def forward_standardized_return(returns: pd.DataFrame, vol_ewma: pd.DataFrame):
    """Next ISO-week's compounded return, standardized by the SAME
    sigma_hat_{i,20d,EWMA} used for position sizing, sampled at the
    predicting week's own Friday close (no look-ahead). DECLARED (spec
    doesn't give this formula): "standardized" is read as vol-standardized
    -- the common quant-research IC convention that prevents a few volatile
    names from dominating a pooled rank correlation -- using the SAME
    estimator already computed for sizing rather than inventing a second
    one. Applied identically to IC_op and every T2 draw, so the comparison
    stays paired/fair regardless of this choice."""
    fwd_ret = weekly_compound_panel(returns)
    vol_friday = scheduling.week_end_values(vol_ewma)
    fwd_ret.index = fwd_ret.index  # already weekly Period
    vol_friday_shift = vol_friday.copy()
    vol_friday_shift.index = vol_friday.index + 1  # vol known at week W predicts week W+1
    common_idx = fwd_ret.index.intersection(vol_friday_shift.index)
    return (fwd_ret.loc[common_idx] / vol_friday_shift.loc[common_idx]).sort_index()


def signal_at_friday_predicting_next_week(f_signal: pd.DataFrame) -> pd.DataFrame:
    """Signal sampled at week W's Friday close (unlagged), reindexed to
    align with week (W+1)'s forward return -- i.e. shifted forward by one
    weekly period so `.loc[W+1]` gives the signal that predicts week W+1."""
    at_friday = scheduling.week_end_values(f_signal)
    shifted = at_friday.copy()
    shifted.index = at_friday.index + 1
    return shifted.sort_index()


def pooled_ic(signal_friday_shifted: pd.DataFrame, fwd_ret_standardized: pd.DataFrame,
              start=None, end=None) -> dict:
    common_idx = signal_friday_shifted.index.intersection(fwd_ret_standardized.index)
    if start is not None:
        common_idx = common_idx[common_idx.start_time >= pd.Timestamp(start)]
    if end is not None:
        common_idx = common_idx[common_idx.start_time <= pd.Timestamp(end)]
    sig = signal_friday_shifted.loc[common_idx].stack()
    ret = fwd_ret_standardized.loc[common_idx].stack()
    common = pd.concat([sig, ret], axis=1, keys=["signal", "ret"]).dropna()
    ic = metrics_ext.pooled_rank_ic(common["signal"], common["ret"])
    return {"ic": ic, "n_obs": int(len(common))}


def cell_overlay_daily_returns(panel, returns, volume, cell, is_start, is_end):
    """Build the op book and T1 (calendar) book at `cell`, return
    (overlay_daily_returns, op_daily_returns, cal_daily_returns, op_book,
    cal_book) -- the shared per-cell computation reused by Steg 0c, 1, 5, 6."""
    op = twins_mod.build_primary(panel, returns, volume, cell, is_start, is_end)
    cal = twins_mod.build_t1(panel, returns, cell, is_start, is_end)
    overlay = (op["returns"] - cal["returns"]).reindex(op["returns"].index)
    return overlay, op["returns"], cal["returns"], op, cal


# ---------------------------------------------------------------------------
# Steg 0a -- data quality
# ---------------------------------------------------------------------------
def steg_0a(panel, tickers, is_end):
    pit_start = panel.pit_start()
    per_ticker = {}
    failed = []
    for t in tickers:
        start = pit_start.get(t, pd.NaT)
        if pd.isna(start):
            per_ticker[t] = {"passed": False, "reason": "no_data", "coverage": 0.0,
                              "max_year_zero_volume_frac": 1.0}
            failed.append(t)
            continue
        px = panel.adjusted_close.loc[start:is_end, t].dropna()
        trading_days = px.index
        vol = panel.volume_raw.reindex(trading_days)[t]
        coverage = float(vol.notna().mean()) if len(trading_days) else 0.0
        zero = (vol == 0.0) & vol.notna()
        year_frac = zero.groupby(zero.index.year).mean()
        max_year_frac = float(year_frac.max()) if len(year_frac) else 0.0
        ok = (coverage >= config.STEG0A_MIN_VOLUME_COVERAGE
              and max_year_frac <= config.STEG0A_MAX_ZERO_VOLUME_FRACTION_PER_YEAR)
        per_ticker[t] = {
            "pit_start": start.date().isoformat(), "n_trading_days": int(len(trading_days)),
            "coverage": coverage, "max_year_zero_volume_frac": max_year_frac, "passed": bool(ok),
        }
        if not ok:
            failed.append(t)
    n_failed = len(failed)
    passed = n_failed <= config.STEG0A_MAX_FAILED_TICKERS
    return {"per_ticker": per_ticker, "failed_tickers": failed, "n_failed": n_failed,
            "max_allowed_failed": config.STEG0A_MAX_FAILED_TICKERS, "passed": bool(passed)}


# ---------------------------------------------------------------------------
# Steg 0b -- clock sanity
# ---------------------------------------------------------------------------
def steg_0b(panel, tickers, is_end, cap=None):
    cap = config.PRIMARY_CELL.c if cap is None else cap
    tau = opclock.compute_tau(panel.volume, window=config.NORMALIZER_WINDOW, cap=cap)
    pit_start = panel.pit_start()
    per_ticker = {}
    for t in tickers:
        start = pit_start.get(t, pd.NaT)
        if pd.isna(start):
            per_ticker[t] = {"passed": False, "reason": "no_data", "frac_in_range": 0.0}
            continue
        tau_t = tau.loc[start:is_end, t]
        roll_mean = tau_t.rolling(config.NORMALIZER_WINDOW, min_periods=config.NORMALIZER_WINDOW).mean()
        valid = roll_mean.dropna()
        if len(valid) == 0:
            per_ticker[t] = {"passed": False, "reason": "insufficient_history", "frac_in_range": 0.0}
            continue
        in_range = valid.between(config.STEG0B_TAU_MEAN_LOW, config.STEG0B_TAU_MEAN_HIGH)
        frac = float(in_range.mean())
        per_ticker[t] = {"frac_in_range": frac, "n_days": int(len(valid)),
                          "passed": bool(frac >= config.STEG0B_MIN_DAYS_IN_RANGE_FRACTION)}

    tau_matrix = tau.loc[:, tickers]
    corr = tau_matrix.corr().to_numpy()
    n = len(tickers)
    off_diag = corr[np.triu_indices(n, k=1)] if n > 1 else np.array([])
    median_corr = float(np.nanmedian(off_diag)) if len(off_diag) else float("nan")

    n_failed = sum(1 for v in per_ticker.values() if not v["passed"])
    passed = n_failed == 0
    return {"per_ticker": per_ticker, "n_failed": n_failed,
            "median_pairwise_corr_tau_diagnostic": median_corr, "passed": bool(passed)}


# ---------------------------------------------------------------------------
# Steg 0c -- clock-choice oracle cap
# ---------------------------------------------------------------------------
def steg_0c(op_returns: pd.Series, cal_returns: pd.Series):
    result = oracle.clock_oracle_test(op_returns, cal_returns)
    result["passed"] = result["passes"]
    return result


# ---------------------------------------------------------------------------
# Steg 1 -- base-engine liveness
# ---------------------------------------------------------------------------
def steg_1(panel, t1_returns_full_is: pd.Series):
    weights, rets, k = baseengine.build_t0_book(panel, config.T0_IS_START, config.T0_IS_END)
    t0_sr = _daily_ann_sharpe(rets.loc[config.T0_IS_START:config.T0_IS_END])
    t0_ok = bool(np.isfinite(t0_sr)
                 and abs(t0_sr - config.T0_TARGET_IS_SHARPE) <= config.T0_TARGET_TOLERANCE)

    t1_sr = _daily_ann_sharpe(t1_returns_full_is.loc[config.IS_START:config.IS_END])
    t1_ok = bool(np.isfinite(t1_sr) and t1_sr >= config.STEG1_T1_MIN_NET_SHARPE)

    passed = t0_ok and t1_ok
    return {
        "t0_reproduced_sharpe": t0_sr, "t0_target_sharpe": config.T0_TARGET_IS_SHARPE,
        "t0_tolerance": config.T0_TARGET_TOLERANCE, "t0_passed": t0_ok,
        "t1_is_sharpe": t1_sr, "t1_min_required": config.STEG1_T1_MIN_NET_SHARPE, "t1_passed": t1_ok,
        "passed": bool(passed),
    }


def _daily_ann_sharpe(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(config.TRADING_DAYS_YEAR))


# ---------------------------------------------------------------------------
# T2 draws, computed once, reused by Steg 2 (IC null + own liveness) and
# Steg 4 (PC1/dispersion null).
# ---------------------------------------------------------------------------
def run_t2_draws(returns, volume, cell, z_cal_friday_full, vol_ewma, is_start, is_end):
    ic_draws, e_abs_z_draws, pc1_draws, dispersion_draws = [], [], [], []
    fwd_ret_std = forward_standardized_return(returns, vol_ewma)
    for seed, z, f_z in twins_mod.t2_draws(returns, volume, cell):
        z_valid = z.to_numpy()
        z_valid = z_valid[~np.isnan(z_valid)]
        e_abs_z_draws.append(float(np.mean(np.abs(z_valid))) if len(z_valid) else float("nan"))

        sig_shift = signal_at_friday_predicting_next_week(f_z)
        ic = pooled_ic(sig_shift, fwd_ret_std, start=is_start, end=is_end)["ic"]
        ic_draws.append(ic)

        z_friday = scheduling.week_end_values(z)
        delta_z = (z_friday - z_cal_friday_full).dropna(how="all")
        delta_z = delta_z.loc[(delta_z.index.start_time >= pd.Timestamp(is_start))
                               & (delta_z.index.start_time <= pd.Timestamp(is_end))]
        pc1_draws.append(metrics_ext.pc1_share(delta_z.to_numpy()))
        weekly_disp = delta_z.std(axis=1, skipna=True)
        dispersion_draws.append(float(weekly_disp.mean()) if len(weekly_disp) else float("nan"))

    return {
        "seeds": list(config.T2_SEEDS),
        "ic_draws": np.array(ic_draws, dtype=float),
        "e_abs_z_draws": np.array(e_abs_z_draws, dtype=float),
        "pc1_share_draws": np.array(pc1_draws, dtype=float),
        "dispersion_draws": np.array(dispersion_draws, dtype=float),
    }


def t2_liveness_assertion(e_abs_z_draws: np.ndarray, primary_e_abs_z: float) -> dict:
    """Spec §7 T2: "E|z| per dragning inom +-20% av primärens; >=95% av
    dragningar passerar, annars stopp." Unconditional (rule 5)."""
    valid = e_abs_z_draws[np.isfinite(e_abs_z_draws)]
    lo = primary_e_abs_z * (1 - config.T2_TOLERANCE_FRACTION)
    hi = primary_e_abs_z * (1 + config.T2_TOLERANCE_FRACTION)
    within = (valid >= lo) & (valid <= hi)
    frac_pass = float(within.mean()) if len(valid) else 0.0
    return {
        "primary_e_abs_z": float(primary_e_abs_z), "tolerance_lo": float(lo), "tolerance_hi": float(hi),
        "n_draws_valid": int(len(valid)), "frac_within_tolerance": frac_pass,
        "min_required_fraction": config.T2_MIN_PASS_FRACTION,
        "passed": bool(frac_pass >= config.T2_MIN_PASS_FRACTION),
    }


# ---------------------------------------------------------------------------
# Steg 2 -- estimator null + IC on traded signed signal
# ---------------------------------------------------------------------------
def steg_2(panel, returns, volume, cell, primary_book, is_start, is_end):
    vol_ewma = returns.ewm(span=config.SIGNAL_VOL_EWMA_SPAN,
                            min_periods=config.SIGNAL_VOL_EWMA_MIN_PERIODS).std()
    fwd_ret_std = forward_standardized_return(returns, vol_ewma)

    f_z_op = opclock.compute_signal(primary_book["M"], primary_book["V"], primary_book["T"],
                                     cell.hl_op, cell.f)
    sig_shift = signal_at_friday_predicting_next_week(f_z_op)
    ic_op_result = pooled_ic(sig_shift, fwd_ret_std, start=is_start, end=is_end)
    ic_op = ic_op_result["ic"]

    z_op = opclock.compute_raw_z(primary_book["M"], primary_book["V"], primary_book["T"], cell.hl_op)
    z_op_valid = z_op.to_numpy()
    z_op_valid = z_op_valid[~np.isnan(z_op_valid)]
    primary_e_abs_z = float(np.mean(np.abs(z_op_valid))) if len(z_op_valid) else float("nan")

    tau_cal = opclock.calendar_twin_tau(returns)
    M_cal, V_cal, T_cal = opclock.compute_op_ewma(returns, tau_cal, cell.hl_op)
    z_cal = opclock.compute_raw_z(M_cal, V_cal, T_cal, cell.hl_op)
    z_cal_friday_full = scheduling.week_end_values(z_cal)

    t2 = run_t2_draws(returns, volume, cell, z_cal_friday_full, vol_ewma, is_start, is_end)
    ic_t2 = t2["ic_draws"]
    ic_t2_valid = ic_t2[np.isfinite(ic_t2)]

    p95_t2 = float(np.percentile(ic_t2_valid, 95)) if len(ic_t2_valid) else float("nan")
    mean_t2 = float(np.mean(ic_t2_valid)) if len(ic_t2_valid) else float("nan")

    crit_abs = bool(np.isfinite(ic_op) and abs(ic_op) >= config.STEG2_MIN_ABS_IC)
    crit_p95 = bool(np.isfinite(ic_op) and np.isfinite(p95_t2) and ic_op > p95_t2)
    crit_gap = bool(np.isfinite(ic_op) and np.isfinite(mean_t2)
                     and (ic_op - mean_t2) >= config.STEG2_MIN_IC_MINUS_MEAN_T2)
    passed = crit_abs and crit_p95 and crit_gap

    t2_live = t2_liveness_assertion(t2["e_abs_z_draws"], primary_e_abs_z)

    return {
        "ic_op": ic_op, "ic_op_n_obs": ic_op_result["n_obs"],
        "ic_t2_p95": p95_t2, "ic_t2_mean": mean_t2, "ic_t2_n_draws_valid": int(len(ic_t2_valid)),
        "criterion_abs_ic": crit_abs, "criterion_ic_gt_p95": crit_p95, "criterion_ic_gap": crit_gap,
        "t2_liveness_assertion": t2_live,
        "_t2_full": t2,  # retained in-process for Steg 4; stripped before JSON serialization
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Steg 3 -- redundancy screen
# ---------------------------------------------------------------------------
def build_battery(panel, returns):
    """Battery {realized vol 20d, mean corr 60d, skew 60d, |r|-autocorr 60d,
    absorption ratio 60d (top-4 eigenvalues), volume-z 20d}, computed at
    DAILY frequency (each component keeps returns' own DatetimeIndex) --
    callers sample each component at Friday close themselves (via
    scheduling.week_end_values), matching how delta_z is sampled, rather
    than reindexing a daily-indexed frame against a weekly PeriodIndex
    directly (which silently matches nothing). Panel-level statistics
    (corr/skew/autocorr/absorption) are computed once and broadcast to
    every ticker; per-ticker ones (vol, volume-z) are not.
    Provenance: mean-corr/skew/autocorr/absorption pattern from
    research/vindkastet/run_gate_checks.py::redundancy_battery, branch
    claude/vindkastet-etf-transient-growth-lu7760, commit 036ca13 (adapted:
    that function used a fixed rolling covariance cache; here recomputed
    directly over `returns`)."""
    realized_vol_20d = returns.rolling(20, min_periods=20).std()

    ew_ret = returns.mean(axis=1)
    skew_60d = ew_ret.rolling(60, min_periods=60).skew()
    absr_autocorr_60d = ew_ret.abs().rolling(60, min_periods=60).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    n = returns.shape[1]
    mean_corr_60d = pd.Series(index=returns.index, dtype=float)
    absorption_60d = pd.Series(index=returns.index, dtype=float)
    r_arr = returns.to_numpy()
    for i in range(59, len(returns)):
        window = r_arr[i - 59:i + 1, :]
        if np.isnan(window).any():
            continue
        corr = np.corrcoef(window, rowvar=False)
        off = corr[~np.eye(n, dtype=bool)]
        mean_corr_60d.iloc[i] = np.nanmean(off)
        cov = np.cov(window, rowvar=False)
        w = np.sort(np.linalg.eigvalsh(cov))[::-1]
        absorption_60d.iloc[i] = w[:4].sum() / w.sum() if w.sum() > 0 else np.nan

    volume_z_20d = (panel.volume - panel.volume.rolling(20, min_periods=20).mean()) \
        / panel.volume.rolling(20, min_periods=20).std()

    cols = returns.columns

    def _broadcast(series):
        return pd.DataFrame({c: series for c in cols}, index=series.index)

    return {
        "realized_vol_20d": realized_vol_20d,
        "mean_corr_60d": _broadcast(mean_corr_60d),
        "skew_60d": _broadcast(skew_60d),
        "absr_autocorr_60d": _broadcast(absr_autocorr_60d),
        "absorption_60d": _broadcast(absorption_60d),
        "volume_z_20d": volume_z_20d,
    }


def steg_3(panel, returns, volume, cell, primary_book, is_start, is_end):
    z_op = opclock.compute_raw_z(primary_book["M"], primary_book["V"], primary_book["T"], cell.hl_op)
    tau_cal = opclock.calendar_twin_tau(returns)
    M_cal, V_cal, T_cal = opclock.compute_op_ewma(returns, tau_cal, cell.hl_op)
    z_cal = opclock.compute_raw_z(M_cal, V_cal, T_cal, cell.hl_op)

    m5 = opclock.variance_clock_input(returns, window=config.T3_VARIANCE_WINDOW)
    tau_var = opclock.compute_tau(m5, window=config.NORMALIZER_WINDOW, cap=cell.c)
    M_t3, V_t3, T_t3 = opclock.compute_op_ewma(returns, tau_var, cell.hl_op)

    z_op_f = scheduling.week_end_values(z_op)
    z_cal_f = scheduling.week_end_values(z_cal)
    idx = z_op_f.index.intersection(z_cal_f.index)
    idx = idx[(idx.start_time >= pd.Timestamp(is_start)) & (idx.start_time <= pd.Timestamp(is_end))]
    delta_z = z_op_f.loc[idx] - z_cal_f.loc[idx]
    delta_z_std = (delta_z - delta_z.mean()) / delta_z.std()  # standardized per asset

    battery = build_battery(panel, returns)
    battery_f = {k: scheduling.week_end_values(v).reindex(idx) for k, v in battery.items()}

    dz_pool = delta_z_std.stack()
    battery_pool = pd.concat({k: v.stack() for k, v in battery_f.items()}, axis=1)
    df_a = pd.concat([dz_pool.rename("delta_z"), battery_pool], axis=1).dropna()
    direct_r2 = float("nan")
    if len(df_a) > 30:
        X = sm.add_constant(df_a.drop(columns="delta_z"))
        model_a = sm.OLS(df_a["delta_z"], X).fit()
        direct_r2 = float(model_a.rsquared)
    direct_grind_kill = bool(np.isfinite(direct_r2) and direct_r2 >= config.STEG3_R2_KILL)

    vol_ewma = returns.ewm(span=config.SIGNAL_VOL_EWMA_SPAN,
                            min_periods=config.SIGNAL_VOL_EWMA_MIN_PERIODS).std()
    fwd_ret_std = forward_standardized_return(returns, vol_ewma)

    f_op = opclock.compute_signal(primary_book["M"], primary_book["V"], primary_book["T"], cell.hl_op, cell.f)
    f_cal = opclock.compute_signal(M_cal, V_cal, T_cal, cell.hl_op, cell.f)
    f_t3 = opclock.compute_signal(M_t3, V_t3, T_t3, cell.hl_op, cell.f)

    sig_op = signal_at_friday_predicting_next_week(f_op).stack()
    sig_cal = signal_at_friday_predicting_next_week(f_cal).stack()
    sig_t3 = signal_at_friday_predicting_next_week(f_t3).stack()
    ret_pool = fwd_ret_std.stack()

    df_b = pd.concat({"cal": sig_cal, "t3": sig_t3, "op": sig_op, "ret": ret_pool}, axis=1).dropna()
    df_b = df_b[(df_b.index.get_level_values(0).map(lambda p: p.start_time) >= pd.Timestamp(is_start))
                & (df_b.index.get_level_values(0).map(lambda p: p.start_time) <= pd.Timestamp(is_end))]

    delta_r2, nw_t_op = float("nan"), float("nan")
    incremental_kill = True
    if len(df_b) > 30:
        X_base = sm.add_constant(df_b[["cal", "t3"]])
        base_model = sm.OLS(df_b["ret"], X_base).fit()
        X_full = sm.add_constant(df_b[["cal", "t3", "op"]])
        full_model = sm.OLS(df_b["ret"], X_full).fit(cov_type="HAC", cov_kwds={"maxlags": config.HAC_LAGS})
        delta_r2 = float(full_model.rsquared - base_model.rsquared)
        nw_t_op = float(full_model.tvalues["op"])
        loadbearing_ok = (np.isfinite(delta_r2) and delta_r2 >= config.STEG3_MIN_DELTA_R2
                           and np.isfinite(nw_t_op) and nw_t_op >= config.STEG3_MIN_NW_T)
        incremental_kill = not loadbearing_ok

    killed = direct_grind_kill or incremental_kill
    passed = not killed
    return {
        "direct_r2": direct_r2, "direct_grind_kill": direct_grind_kill,
        "delta_r2_op_given_cal_t3": delta_r2, "nw_t_op": nw_t_op, "incremental_kill": incremental_kill,
        "killed": killed, "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Steg 4 -- PC1 / effective breadth on Delta-z panel
# ---------------------------------------------------------------------------
def steg_4(returns, cell, primary_book, is_start, is_end, t2_full):
    z_op = opclock.compute_raw_z(primary_book["M"], primary_book["V"], primary_book["T"], cell.hl_op)
    tau_cal = opclock.calendar_twin_tau(returns)
    M_cal, V_cal, T_cal = opclock.compute_op_ewma(returns, tau_cal, cell.hl_op)
    z_cal = opclock.compute_raw_z(M_cal, V_cal, T_cal, cell.hl_op)

    z_op_f = scheduling.week_end_values(z_op)
    z_cal_f = scheduling.week_end_values(z_cal)
    idx = z_op_f.index.intersection(z_cal_f.index)
    idx = idx[(idx.start_time >= pd.Timestamp(is_start)) & (idx.start_time <= pd.Timestamp(is_end))]
    delta_z = (z_op_f.loc[idx] - z_cal_f.loc[idx])

    pc1_primary = metrics_ext.pc1_share(delta_z.to_numpy())
    weekly_disp = delta_z.std(axis=1, skipna=True)
    dispersion_primary = float(weekly_disp.mean()) if len(weekly_disp) else float("nan")

    pc1_null = t2_full["pc1_share_draws"]
    disp_null = t2_full["dispersion_draws"]
    pc1_null_valid = pc1_null[np.isfinite(pc1_null)]
    disp_null_valid = disp_null[np.isfinite(disp_null)]

    pc1_ok = bool(np.isfinite(pc1_primary) and pc1_primary <= config.STEG4_MAX_PC1_SHARE)
    disp_p95 = float(np.percentile(disp_null_valid, config.STEG4_DISPERSION_NULL_PERCENTILE)) \
        if len(disp_null_valid) else float("nan")
    disp_ok = bool(np.isfinite(dispersion_primary) and np.isfinite(disp_p95)
                    and dispersion_primary > disp_p95)

    passed = pc1_ok and disp_ok
    return {
        "pc1_share_primary": float(pc1_primary) if np.isfinite(pc1_primary) else None,
        "pc1_share_max_allowed": config.STEG4_MAX_PC1_SHARE, "pc1_share_ok": pc1_ok,
        "dispersion_primary": dispersion_primary, "dispersion_null_p95": disp_p95,
        "dispersion_null_n_valid": int(len(disp_null_valid)), "dispersion_ok": disp_ok,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Steg 5 -- IS economics + robustness
# ---------------------------------------------------------------------------
def _neighbor_cells(primary: config.GridCell):
    neighbors = []
    for hl in config.HL_OP_GRID:
        if hl != primary.hl_op:
            neighbors.append(config.GridCell(hl_op=hl, c=primary.c, f=primary.f))
    for c in config.C_GRID:
        if c != primary.c:
            neighbors.append(config.GridCell(hl_op=primary.hl_op, c=c, f=primary.f))
    for f in config.F_GRID:
        if f != primary.f:
            neighbors.append(config.GridCell(hl_op=primary.hl_op, c=primary.c, f=f))
    return neighbors


def steg_5(panel, returns, volume, primary_overlay_daily, is_start, is_end):
    weekly_overlay = weekly_compound_panel(primary_overlay_daily.to_frame("overlay"))["overlay"]
    weekly_overlay_is = weekly_overlay.loc[
        (weekly_overlay.index.start_time >= pd.Timestamp(is_start))
        & (weekly_overlay.index.start_time <= pd.Timestamp(is_end))]

    is_overlay_sr = annualized_sharpe_weekly(weekly_overlay_is)
    sr_ok = bool(np.isfinite(is_overlay_sr) and is_overlay_sr >= config.STEG5_MIN_IS_OVERLAY_SHARPE)

    rng = np.random.default_rng(config.GLOBAL_SEED)
    r = weekly_overlay_is.dropna().to_numpy()
    n = len(r)
    draws = np.empty(config.STEG5_BOOTSTRAP_N_DRAWS)
    for i in range(config.STEG5_BOOTSTRAP_N_DRAWS):
        idx = stationary_block_bootstrap_indices(n, config.STEG5_BOOTSTRAP_BLOCK_WEEKS, rng)
        draws[i] = annualized_sharpe_weekly(pd.Series(r[idx]))
    draws = draws[np.isfinite(draws)]
    alpha = 1.0 - config.STEG5_BOOTSTRAP_CI
    ci_lo = float(np.quantile(draws, alpha / 2)) if len(draws) else float("nan")
    ci_hi = float(np.quantile(draws, 1 - alpha / 2)) if len(draws) else float("nan")
    bootstrap_ok = bool(np.isfinite(ci_lo) and ci_lo > 0)

    sub_signs = []
    for start, end in config.STEG5_SUBPERIODS:
        sub = weekly_overlay.loc[(weekly_overlay.index.start_time >= pd.Timestamp(start))
                                  & (weekly_overlay.index.start_time <= pd.Timestamp(end))]
        sub_signs.append(1 if sub.sum() > 0 else (-1 if sub.sum() < 0 else 0))
    n_positive = sum(1 for s in sub_signs if s > 0)
    subperiod_ok = bool(n_positive >= config.STEG5_MIN_POSITIVE_SUBPERIODS)

    neighbor_results = {}
    for nb in _neighbor_cells(config.PRIMARY_CELL):
        nb_overlay, _, _, _, _ = cell_overlay_daily_returns(panel, returns, volume, nb, is_start, is_end)
        nb_weekly = weekly_compound_panel(nb_overlay.to_frame("overlay"))["overlay"]
        nb_weekly_is = nb_weekly.loc[(nb_weekly.index.start_time >= pd.Timestamp(is_start))
                                      & (nb_weekly.index.start_time <= pd.Timestamp(is_end))]
        nb_sr = annualized_sharpe_weekly(nb_weekly_is)
        retention = (nb_sr / is_overlay_sr) if (np.isfinite(nb_sr) and np.isfinite(is_overlay_sr)
                                                  and is_overlay_sr != 0) else float("nan")
        neighbor_results[f"hl{nb.hl_op}_c{nb.c}_f{nb.f}"] = {
            "overlay_sr": nb_sr, "retention_of_primary": retention,
            "passed": bool(np.isfinite(retention) and retention >= config.STEG5_NEIGHBOR_MIN_SR_RETENTION),
        }
    neighbor_ok = all(v["passed"] for v in neighbor_results.values()) if neighbor_results else False

    passed = sr_ok and bootstrap_ok and subperiod_ok and neighbor_ok
    turnover_diagnostic = float(primary_overlay_daily.abs().mean())  # non-decision-bearing diagnostic
    return {
        "is_overlay_sharpe": is_overlay_sr, "sr_min_required": config.STEG5_MIN_IS_OVERLAY_SHARPE,
        "sr_ok": sr_ok, "bootstrap_ci_90pct": [ci_lo, ci_hi], "bootstrap_ok": bootstrap_ok,
        "subperiod_signs": sub_signs, "n_positive_subperiods": n_positive, "subperiod_ok": subperiod_ok,
        "neighbor_cells": neighbor_results, "neighbor_ok": neighbor_ok,
        "turnover_diagnostic_mean_abs_overlay_weight_change": turnover_diagnostic,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Steg 6 -- DSR with surface pool (tiling)
# ---------------------------------------------------------------------------
def steg_6(panel, returns, volume, is_start, is_end, primary_overlay_daily,
           neighbor_results_daily_overlays=None):
    weekly_overlay_primary = weekly_compound_panel(primary_overlay_daily.to_frame("o"))["o"]
    weekly_overlay_primary_is = weekly_overlay_primary.loc[
        (weekly_overlay_primary.index.start_time >= pd.Timestamp(is_start))
        & (weekly_overlay_primary.index.start_time <= pd.Timestamp(is_end))].dropna()

    grid_sharpes_per_period = []
    for cell in config.GRID:
        overlay, _, _, _, _ = cell_overlay_daily_returns(panel, returns, volume, cell, is_start, is_end)
        weekly = weekly_compound_panel(overlay.to_frame("o"))["o"]
        weekly_is = weekly.loc[(weekly.index.start_time >= pd.Timestamp(is_start))
                                & (weekly.index.start_time <= pd.Timestamp(is_end))].dropna()
        sr_ann = annualized_sharpe_weekly(weekly_is)
        sr_per_period = sr_ann / np.sqrt(config.WEEKS_YEAR) if np.isfinite(sr_ann) else np.nan
        grid_sharpes_per_period.append(sr_per_period)
    grid_sharpes_per_period = np.array(grid_sharpes_per_period, dtype=float)
    grid_valid = grid_sharpes_per_period[np.isfinite(grid_sharpes_per_period)]

    # Tiling mechanism, provenance: research/smittotalet/run_research.py:168
    # `np.tile(grid_sharpes, config.N_EFFECTIVE_SURFACE_READS)`, branch
    # claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b.
    tiled_trials = np.tile(grid_valid, config.N_EFFECTIVE_SURFACE_READS)

    sr_hat_ann = annualized_sharpe_weekly(weekly_overlay_primary_is)
    sr_hat_per_period = sr_hat_ann / np.sqrt(config.WEEKS_YEAR) if np.isfinite(sr_hat_ann) else np.nan
    n_obs = int(len(weekly_overlay_primary_is))

    dsr_result = deflated_sharpe_ratio(sr_hat_per_period, tiled_trials, n_obs) \
        if np.isfinite(sr_hat_per_period) and len(tiled_trials) >= 2 else \
        {"dsr": float("nan"), "z": float("nan"), "sr0": float("nan")}

    # DECLARED (spec §8 Steg 6 says "DSR > 0" without specifying probability
    # vs excess-over-null): a literal DSR-PROBABILITY > 0 bar is almost
    # always trivially true (dsr in (0,1) except at z=-inf), so the
    # economically meaningful, LESS-FAVORABLE-TO-THE-STRATEGY reading (rule
    # 1) is z > 0, i.e. dsr_prob > 0.5 -- observed SR must actually exceed
    # the expected-max-under-the-null benchmark, not merely have a
    # technically-nonzero probability of doing so.
    passed = bool(np.isfinite(dsr_result.get("z", float("nan"))) and dsr_result["z"] > 0)

    return {
        "n_grid_cells": config.STEG6_N_GRID_CELLS, "n_grid_valid": int(len(grid_valid)),
        "n_effective_surface_reads": config.N_EFFECTIVE_SURFACE_READS,
        "grid_sharpes_per_period": grid_valid.tolist(),
        "primary_sharpe_per_period": float(sr_hat_per_period) if np.isfinite(sr_hat_per_period) else None,
        "dsr_prob": dsr_result.get("dsr"), "dsr_z": dsr_result.get("z"), "dsr_sr0": dsr_result.get("sr0"),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Steg 7 -- OOS, one reading (UCITS). Gated; never runs unless unlock_oos.
# ---------------------------------------------------------------------------
def steg_7(config_dict, unlock_oos, is_overlay_sharpe, fetch_oos_panel_fn=None):
    try:
        oos_guard.enforce_universe_oos_gate(config_dict, unlock_oos)
    except oos_guard.OOSLockError as exc:
        return {"ran": False, "reason": str(exc), "passed": None}

    if fetch_oos_panel_fn is None:
        raise RuntimeError("Steg 7 unlocked but no OOS panel fetcher provided.")
    panel = fetch_oos_panel_fn()  # only ever called once unlock_oos=True

    tickers = config.oos_universe_flat()
    pit_start = panel.pit_start()
    per_ticker_cov = {}
    n_covered = 0
    for t in tickers:
        start = pit_start.get(t, pd.NaT)
        if pd.isna(start):
            per_ticker_cov[t] = {"coverage": 0.0, "max_year_zero_volume_frac": 1.0, "ok": False}
            continue
        px = panel.adjusted_close.loc[start:config.OOS_END, t].dropna()
        vol = panel.volume_raw.reindex(px.index)[t]
        coverage = float(vol.notna().mean()) if len(px) else 0.0
        zero = (vol == 0.0) & vol.notna()
        year_frac = zero.groupby(zero.index.year).mean()
        max_year_frac = float(year_frac.max()) if len(year_frac) else 0.0
        ok = (coverage >= config.STEG7_MIN_VOLUME_COVERAGE
              and max_year_frac <= config.STEG7_MAX_ZERO_VOLUME_FRACTION_PER_YEAR)
        per_ticker_cov[t] = {"coverage": coverage, "max_year_zero_volume_frac": max_year_frac, "ok": ok}
        if ok:
            n_covered += 1

    coverage_fraction = n_covered / len(tickers) if tickers else 0.0
    gate_pass = coverage_fraction >= config.STEG7_MIN_ISIN_COVERAGE_FRACTION
    if not gate_pass:
        return {"ran": True, "pre_data_gate_passed": False, "per_ticker_coverage": per_ticker_cov,
                "coverage_fraction": coverage_fraction, "passed": False}

    returns = panel.simple_returns()
    cell = config.PRIMARY_CELL
    overlay, op_ret, cal_ret, _, _ = cell_overlay_daily_returns(
        panel, returns, panel.volume, cell, config.OOS_START, config.OOS_END)
    weekly = weekly_compound_panel(overlay.to_frame("o"))["o"]
    oos_sr = annualized_sharpe_weekly(weekly)

    net_positive = bool(np.isfinite(oos_sr) and oos_sr > 0)
    frac_of_is = (oos_sr / is_overlay_sharpe) if (np.isfinite(oos_sr) and np.isfinite(is_overlay_sharpe)
                                                    and is_overlay_sharpe != 0) else float("nan")
    frac_ok = bool(np.isfinite(frac_of_is) and frac_of_is >= config.STEG7_MIN_IS_SHARPE_FRACTION)
    passed = net_positive and frac_ok
    return {
        "ran": True, "pre_data_gate_passed": True, "coverage_fraction": coverage_fraction,
        "oos_overlay_sharpe": oos_sr, "is_overlay_sharpe_reference": is_overlay_sharpe,
        "fraction_of_is": frac_of_is, "min_fraction_required": config.STEG7_MIN_IS_SHARPE_FRACTION,
        "net_positive": net_positive, "fraction_ok": frac_ok, "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator: run Steg 0a -> Steg 7 in strict order, stop at the
# first failure (rule 4). Returns the full per-step record plus which step
# (if any) the ladder stopped at.
# ---------------------------------------------------------------------------
STEP_ORDER = ["steg_0a", "steg_0b", "steg_0c", "steg_1", "steg_2", "steg_3",
              "steg_4", "steg_5", "steg_6", "steg_7"]


def run_ladder(panel, tickers, config_dict, unlock_oos: bool = False,
                fetch_oos_panel_fn=None) -> dict:
    is_start, is_end = config.IS_START, config.IS_END
    cell = config.PRIMARY_CELL
    returns = panel.simple_returns()
    volume = panel.volume

    steps = {}
    liveness = {}
    stopped_at = None

    steps["steg_0a"] = steg_0a(panel, tickers, is_end)
    if not steps["steg_0a"]["passed"]:
        stopped_at = "steg_0a"
        return _finish(steps, liveness, stopped_at)

    steps["steg_0b"] = steg_0b(panel, tickers, is_end)
    if not steps["steg_0b"]["passed"]:
        stopped_at = "steg_0b"
        return _finish(steps, liveness, stopped_at)

    primary_overlay, op_ret, cal_ret, primary_book, cal_book = cell_overlay_daily_returns(
        panel, returns, volume, cell, is_start, is_end)
    liveness["T1_calendar"] = twin_liveness(scheduling.week_end_values(
        opclock.compute_signal(cal_book["M"], cal_book["V"], cal_book["T"], cell.hl_op, cell.f)))

    steps["steg_0c"] = steg_0c(op_ret, cal_ret)
    if not steps["steg_0c"]["passed"]:
        stopped_at = "steg_0c"
        return _finish(steps, liveness, stopped_at)

    steps["steg_1"] = steg_1(panel, cal_ret)
    if not steps["steg_1"]["passed"]:
        stopped_at = "steg_1"
        return _finish(steps, liveness, stopped_at)

    steg2_result = steg_2(panel, returns, volume, cell, primary_book, is_start, is_end)
    t2_full = steg2_result.pop("_t2_full")
    liveness["T2_shuffle"] = {
        "e_abs_z_liveness": steg2_result["t2_liveness_assertion"],
        "n_draws": len(t2_full["seeds"]),
    }
    steps["steg_2"] = steg2_result
    if not steps["steg_2"]["passed"]:
        stopped_at = "steg_2"
        return _finish(steps, liveness, stopped_at)

    steps["steg_3"] = steg_3(panel, returns, volume, cell, primary_book, is_start, is_end)
    m5 = opclock.variance_clock_input(returns, window=config.T3_VARIANCE_WINDOW)
    tau_var = opclock.compute_tau(m5, window=config.NORMALIZER_WINDOW, cap=cell.c)
    M_t3, V_t3, T_t3 = opclock.compute_op_ewma(returns, tau_var, cell.hl_op)
    liveness["T3_variance"] = twin_liveness(scheduling.week_end_values(
        opclock.compute_signal(M_t3, V_t3, T_t3, cell.hl_op, cell.f)))
    liveness["T0_base_engine"] = twin_liveness(scheduling.week_end_values(baseengine.t0_raw_signal(panel)))
    if not steps["steg_3"]["passed"]:
        stopped_at = "steg_3"
        return _finish(steps, liveness, stopped_at)

    steps["steg_4"] = steg_4(returns, cell, primary_book, is_start, is_end, t2_full)
    if not steps["steg_4"]["passed"]:
        stopped_at = "steg_4"
        return _finish(steps, liveness, stopped_at)

    steps["steg_5"] = steg_5(panel, returns, volume, primary_overlay, is_start, is_end)
    if not steps["steg_5"]["passed"]:
        stopped_at = "steg_5"
        return _finish(steps, liveness, stopped_at)

    steps["steg_6"] = steg_6(panel, returns, volume, is_start, is_end, primary_overlay)
    if not steps["steg_6"]["passed"]:
        stopped_at = "steg_6"
        return _finish(steps, liveness, stopped_at)

    steps["steg_7"] = steg_7(config_dict, unlock_oos, steps["steg_5"]["is_overlay_sharpe"],
                              fetch_oos_panel_fn=fetch_oos_panel_fn)
    if steps["steg_7"]["passed"] is False:
        stopped_at = "steg_7"

    return _finish(steps, liveness, stopped_at)


def _finish(steps: dict, liveness: dict, stopped_at) -> dict:
    all_passed = stopped_at is None and all(
        s.get("passed") is not False for s in steps.values())
    return {"steps": steps, "liveness_assertions": liveness, "stopped_at": stopped_at,
            "all_steps_run_passed": bool(all_passed), "step_order": STEP_ORDER}
