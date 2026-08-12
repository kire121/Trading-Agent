"""Fast-exit-stege gate functions (spec Sec.10). Each function takes
whatever data it needs and returns a dict with every K-criterion's raw
value and PASS/FAIL bool, plus an overall `passed` bool -- never silently
dropping a criterion (mirrors lib/delivery.py's own "never filter out an
assertion" discipline, applied here to gate criteria).
"""
import numpy as np
import pandas as pd
from scipy import stats as sstats

from lib import bootstrap as lib_bootstrap
from lib import metrics as lib_metrics

from . import basbok
from . import config
from . import survival_trend as st


def steg1_hazard_structure(episodes: pd.DataFrame, bootstrap_b: int, seed: int,
                            k_bounds: tuple = config.K1_WEIBULL_K_BOUNDS) -> dict:
    """K1a (shape<=0.90 + cluster-bootstrap CI95 upper<1.00), K1b (frailty
    control, delta AIC>=6), K1c (direction lock -- k_hat>1 with CI95
    excluding 1 => kill, inverse flagged contaminated, never flipped)."""
    est = episodes[episodes["left_censored"] == 0]
    d, event, instr = est["d"].to_numpy(), est["event"].to_numpy(), est["instrument"].to_numpy()

    k_hat, lam_weib, ll_weib, aic_weib = st.weibull_stratified_mle(d, event, instr, k_bounds=k_bounds)
    _, ll_exp, aic_exp = st.exp_stratified_mle(d, event, instr)
    boot = st.cluster_bootstrap_k(est, B=bootstrap_b, seed=seed, k_bounds=k_bounds)

    k1a_shape_ok = k_hat <= config.K1A_MAX_K_HAT
    k1a_ci_ok = boot["ci_upper_95"] < config.K1A_MAX_CI95_UPPER
    k1a_pass = bool(k1a_shape_ok and k1a_ci_ok)

    delta_aic = aic_exp - aic_weib
    k1b_pass = bool(delta_aic >= config.K1B_MIN_DELTA_AIC)

    k1c_contaminated = bool(k_hat > 1.0 and boot["ci_lower_95"] > 1.0)

    return {
        "k_hat": k_hat,
        "ci_lower_95": boot["ci_lower_95"],
        "ci_upper_95": boot["ci_upper_95"],
        "aic_weibull": aic_weib,
        "aic_exp": aic_exp,
        "delta_aic": float(delta_aic),
        "loglik_weibull": ll_weib,
        "loglik_exp": ll_exp,
        "n_episodes_est": int(len(est)),
        "n_instruments_est": int(len(np.unique(instr))) if len(instr) else 0,
        "K1a_shape": bool(k1a_shape_ok),
        "K1a_ci": bool(k1a_ci_ok),
        "K1a": k1a_pass,
        "K1b": k1b_pass,
        "K1c_contaminated": k1c_contaminated,
        "passed_1a_1b": bool(k1a_pass and k1b_pass),
        "passed": bool(k1a_pass and k1b_pass and not k1c_contaminated),
    }


# ---------------------------------------------------------------------------
# Steg 2 -- redundancy screen (spec Sec.10)
# ---------------------------------------------------------------------------

def _stack_panel_battery(x_panel: pd.DataFrame, battery: dict) -> pd.DataFrame:
    """Long-format (instrument, week, X, control...) frame, market-wide
    (Series-valued) controls broadcast across instruments."""
    long = x_panel.stack(future_stack=True).rename("X").reset_index()
    long.columns = ["week", "instrument", "X"]
    for name, ctrl in battery.items():
        if isinstance(ctrl, pd.Series):
            long[name] = long["week"].map(ctrl)
        else:
            stacked = ctrl.stack(future_stack=True).rename(name).reset_index()
            stacked.columns = ["week", "instrument", name]
            long = long.merge(stacked, on=["week", "instrument"], how="left")
    return long.dropna()


def steg2_redundancy_screen(x_panel: pd.DataFrame, battery: dict,
                             hac_lags: int = config.K2_HAC_LAGS) -> dict:
    """Pooled panel regression of X on the control battery, instrument fixed
    effects via within-transformation (demean by instrument), R^2-based kill
    (K2: kill if R^2>=0.5). HAC(lag=8) t-stats reported per coefficient as a
    diagnostic (statsmodels HAC on a single pooled, instrument-sorted series
    approximates within-instrument serial correlation; it does not gate the
    R^2-based kill decision, which is SE-methodology-independent -- see
    module note in gates.py / AVVIKELSER.md for why the exact panel-HAC
    estimator choice does not change K2's pass/fail)."""
    import statsmodels.api as sm

    long = _stack_panel_battery(x_panel, battery)
    control_cols = [c for c in battery.keys()]
    demeaned = long.copy()
    for col in ["X"] + control_cols:
        demeaned[col] = demeaned[col] - demeaned.groupby("instrument")[col].transform("mean")
    demeaned = demeaned.sort_values(["instrument", "week"])

    y = demeaned["X"].to_numpy()
    X = demeaned[control_cols].to_numpy()
    if len(y) < len(control_cols) + 2:
        return {"r2": float("nan"), "n_obs": len(y), "K2_passed": False, "coefs": {}}

    model = sm.OLS(y, X).fit()
    r2 = float(model.rsquared)
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(hac_lags, 1)})

    return {
        "r2": r2,
        "n_obs": int(len(y)),
        "coefs": {name: float(c) for name, c in zip(control_cols, model.params)},
        "hac_tstats": {name: float(t) for name, t in zip(control_cols, hac.tvalues)},
        "K2_kill": bool(r2 >= config.K2_MAX_R2),
        "passed": bool(r2 < config.K2_MAX_R2),
    }


# ---------------------------------------------------------------------------
# Steg 3 -- IC on the traded, signed construction (spec Sec.10)
# ---------------------------------------------------------------------------

def weekly_rank_ic(x_panel: pd.DataFrame, y_panel: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank-IC(X_t, Y_t) per week."""
    ic = {}
    for week in x_panel.index:
        x = x_panel.loc[week]
        y = y_panel.loc[week] if week in y_panel.index else None
        if y is None:
            continue
        common = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
        if len(common) < 3:
            continue
        ic[week] = sstats.spearmanr(common["x"], common["y"]).correlation
    return pd.Series(ic)


def residualize_cross_sectionally(x_panel: pd.DataFrame, controls: dict) -> pd.DataFrame:
    """Per-week cross-sectional OLS residual of X on the given controls
    (spec K3.2: "X residualiserad tvarsnittligt per vecka pa {u, sigma_ann,
    z1v}")."""
    out = pd.DataFrame(np.nan, index=x_panel.index, columns=x_panel.columns)
    for week in x_panel.index:
        x = x_panel.loc[week]
        ctrl_frame = pd.DataFrame({name: c.loc[week] if week in c.index else np.nan
                                    for name, c in controls.items()})
        frame = pd.concat([x.rename("x"), ctrl_frame], axis=1).dropna()
        if len(frame) < len(controls) + 2:
            continue
        X = np.column_stack([np.ones(len(frame)), frame[list(controls.keys())].to_numpy()])
        y = frame["x"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out.loc[week, frame.index] = resid
    return out


def steg3_ic(x_panel: pd.DataFrame, y_panel: pd.DataFrame, controls: dict,
             tb_null_ics: np.ndarray, hac_lags: int = config.NW_HAC_LAGS) -> dict:
    ic_series = weekly_rank_ic(x_panel, y_panel)
    mean_ic = float(ic_series.mean())
    nw_t = lib_metrics.newey_west_tstat(ic_series, lags=hac_lags)
    k3_1 = bool(np.isfinite(mean_ic) and mean_ic >= config.K3_1_MIN_IC and
                np.isfinite(nw_t) and nw_t >= config.K3_1_MIN_NW_T)

    x_res = residualize_cross_sectionally(x_panel, controls)
    ic_res_series = weekly_rank_ic(x_res, y_panel)
    mean_ic_res = float(ic_res_series.mean())
    nw_t_res = lib_metrics.newey_west_tstat(ic_res_series, lags=hac_lags)
    k3_2 = bool(np.isfinite(mean_ic_res) and mean_ic_res >= config.K3_2_MIN_IC_RES and
                np.isfinite(nw_t_res) and nw_t_res >= config.K3_2_MIN_NW_T)

    p95_null = float(np.nanpercentile(tb_null_ics, 95))
    k3_3 = bool(np.isfinite(mean_ic) and mean_ic > p95_null)

    return {
        "mean_ic": mean_ic, "nw_t": float(nw_t), "K3_1": k3_1,
        "mean_ic_res": mean_ic_res, "nw_t_res": float(nw_t_res), "K3_2": k3_2,
        "tb_null_p95": p95_null, "n_tb_draws": int(len(tb_null_ics)), "K3_3": k3_3,
        "passed": bool(k3_1 and k3_2 and k3_3),
    }


# ---------------------------------------------------------------------------
# Steg 4 -- effective breadth / common factor (spec Sec.10)
# ---------------------------------------------------------------------------

def pc1_share(x_panel: pd.DataFrame) -> float:
    """K4.1: PC1's share of total variance of the weekly X panel."""
    clean = x_panel.dropna(axis=0, how="all").fillna(0.0)
    if clean.shape[1] < 2 or clean.shape[0] < 2:
        return float("nan")
    cov = np.cov(clean.to_numpy(), rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    total = eigvals.sum()
    return float(eigvals.max() / total) if total > 0 else float("nan")


def compression_share(ages: pd.DataFrame, spread_weeks: int = config.K4_2_COMPRESSION_AGE_SPREAD_WEEKS) -> dict:
    """K4.2: share of weeks where (a_q90 - a_q10) among KNOWN ages < spread_weeks."""
    spreads = []
    for week in ages.index:
        known = ages.loc[week].dropna()
        if len(known) < 5:
            continue
        q10, q90 = np.percentile(known, [10, 90])
        spreads.append(q90 - q10)
    spreads = np.array(spreads)
    if len(spreads) == 0:
        return {"share": float("nan"), "n_weeks": 0}
    share = float((spreads < spread_weeks).mean())
    return {"share": share, "n_weeks": int(len(spreads))}


def rank_persistence(p_panel: pd.DataFrame, lag_weeks: int = config.K4_3_LAG_WEEKS) -> float:
    """K4.3: mean Spearman(P_t, P_{t-lag}) across weeks."""
    corrs = []
    idx = p_panel.index
    for i in range(lag_weeks, len(idx)):
        cur = p_panel.iloc[i]
        prev = p_panel.iloc[i - lag_weeks]
        common = pd.concat([cur.rename("c"), prev.rename("p")], axis=1).dropna()
        if len(common) < 3:
            continue
        corrs.append(sstats.spearmanr(common["c"], common["p"]).correlation)
    return float(np.nanmean(corrs)) if corrs else float("nan")


def turnover_ratio(weights_tilt: pd.DataFrame, weights_t0: pd.DataFrame) -> float:
    """K4.4: mean weekly turnover(tilted) / mean weekly turnover(T0)."""
    to_tilt = weights_tilt.diff().abs().sum(axis=1).mean()
    to_t0 = weights_t0.diff().abs().sum(axis=1).mean()
    return float(to_tilt / to_t0) if to_t0 else float("nan")


def steg4_effective_breadth(x_panel: pd.DataFrame, ages: pd.DataFrame, p_panel: pd.DataFrame,
                             weights_tilt: pd.DataFrame, weights_t0: pd.DataFrame) -> dict:
    pc1 = pc1_share(x_panel)
    k4_1 = bool(np.isfinite(pc1) and pc1 <= config.K4_1_MAX_PC1_SHARE)

    comp = compression_share(ages)
    k4_2 = bool(np.isfinite(comp["share"]) and comp["share"] <= config.K4_2_MAX_COMPRESSION_SHARE)

    persist = rank_persistence(p_panel)
    k4_3 = bool(np.isfinite(persist) and persist >= config.K4_3_MIN_RANK_PERSISTENCE)

    turnover = turnover_ratio(weights_tilt, weights_t0)
    k4_4 = bool(np.isfinite(turnover) and turnover <= config.K4_4_MAX_TURNOVER_RATIO)

    return {
        "pc1_share": pc1, "K4_1": k4_1,
        "compression_share": comp["share"], "compression_n_weeks": comp["n_weeks"], "K4_2": k4_2,
        "rank_persistence": persist, "K4_3": k4_3,
        "turnover_ratio": turnover, "K4_4": k4_4,
        "passed": bool(k4_1 and k4_2 and k4_3 and k4_4),
    }


# ---------------------------------------------------------------------------
# Steg 5a -- IS-A backtest + 6-cell grid (spec Sec.10)
# ---------------------------------------------------------------------------

def sr_net_uplift(returns_main: pd.Series, returns_t0: pd.Series, window=None) -> float:
    """ΔSR_net = weekly-compounded Sharpe(main) - Sharpe(T0). Both inputs
    are DAILY return series (basbok.portfolio_returns' native frequency);
    compounded to weekly (basbok.weekly_return, ISO W-FRI buckets) before
    Sharpe so lib.metrics.sharpe_ratio's periods_per_year=52 default is
    correct -- see basbok.weekly_return's docstring."""
    r_main = returns_main.loc[window[0]:window[1]] if window else returns_main
    r_t0 = returns_t0.loc[window[0]:window[1]] if window else returns_t0
    return float(lib_metrics.sharpe_ratio(basbok.weekly_return(r_main))
                 - lib_metrics.sharpe_ratio(basbok.weekly_return(r_t0)))


def is_a_thirds(returns_main: pd.Series, returns_t0: pd.Series, is_a_start, is_a_end,
                 n_parts: int = config.N_IS_A_SUBPERIODS) -> dict:
    idx = pd.date_range(is_a_start, is_a_end, freq="D")
    bounds = pd.date_range(is_a_start, is_a_end, periods=n_parts + 1)
    uplifts = []
    for i in range(n_parts):
        uplifts.append(sr_net_uplift(returns_main, returns_t0, (bounds[i], bounds[i + 1])))
    n_positive = sum(1 for u in uplifts if np.isfinite(u) and u > 0)
    worst = min((u for u in uplifts if np.isfinite(u)), default=float("nan"))
    return {
        "uplifts": uplifts,
        "n_positive": n_positive,
        "worst": worst,
        "K5_3": bool(n_positive >= config.K5_3_MIN_THIRDS_POSITIVE and
                     (np.isfinite(worst) and worst >= config.K5_3_MIN_THIRD_UPLIFT)),
    }


def pooled_surface_dsr(observed_returns: pd.Series, grid_returns: dict, is_start, is_end,
                        n_effective_reads: int = config.N_EFFECTIVE_SURFACE_READS) -> dict:
    """Smittotalet's tiling technique (research/smittotalet/run_research.py,
    branch claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b,
    reused per spec Sec.10 Steg 5a: "Smittotalets tiling-implementation"):
    tile the grid's own M per-period Sharpes n_effective_reads times
    (preserves std(sr_trials) exactly, inflates n for the DSR order-
    statistics approximation) as the trial pool for
    lib.metrics.deflated_sharpe_ratio, evaluated against `observed_returns`'
    own (not a T0-relative uplift) weekly Sharpe -- DSR is about the raw
    strategy Sharpe's plausibility against a null of N_trials draws, not
    about the uplift metric."""
    grid_sharpes = np.array([lib_metrics.sharpe_ratio(basbok.weekly_return(r.loc[is_start:is_end]),
                                                        annualize=False)
                              for r in grid_returns.values()])
    grid_sharpes = grid_sharpes[np.isfinite(grid_sharpes)]
    pooled_trials = np.tile(grid_sharpes, n_effective_reads) if len(grid_sharpes) else grid_sharpes

    obs_weekly = basbok.weekly_return(observed_returns.loc[is_start:is_end])
    sr_period = lib_metrics.sharpe_ratio(obs_weekly, annualize=False)
    summary = lib_metrics.summary_stats(obs_weekly)
    if len(pooled_trials) < 2:
        return {"dsr": float("nan"), "z": float("nan"), "n_trials": int(len(pooled_trials))}
    dsr = lib_metrics.deflated_sharpe_ratio(sr_period, pooled_trials, n_obs=summary["n_obs"],
                                             skewness=summary["skew"], kurtosis=summary["kurtosis"])
    return {"dsr": dsr["dsr"], "z": dsr["z"], "n_trials": int(len(pooled_trials)),
            "n_grid_cells": int(len(grid_sharpes))}


def steg5a_backtest_grid(grid_books: dict, t0_book: dict, twin_a_book: dict, twin_c_book: dict,
                          tb_portfolio_uplifts: np.ndarray, is_a_start, is_a_end) -> dict:
    """`grid_books`: {(kappa, exec_lag): book_dict} for all 6 cells, each
    book_dict having a "returns" Series (full-history daily; windowed to
    IS-A here). Primary cell is config.PRIMARY_CELL."""
    primary_key = (config.PRIMARY_CELL.kappa, config.PRIMARY_CELL.exec_lag)
    primary_book = grid_books[primary_key]
    win = (is_a_start, is_a_end)

    primary_uplift = sr_net_uplift(primary_book["returns"], t0_book["returns"], win)
    k5_1 = bool(np.isfinite(primary_uplift) and primary_uplift >= config.K5_1_MIN_PRIMARY_UPLIFT)

    cell_uplifts = {k: sr_net_uplift(b["returns"], t0_book["returns"], win) for k, b in grid_books.items()}
    k5_2 = bool(all(np.isfinite(u) and u > 0 for u in cell_uplifts.values()))

    thirds = is_a_thirds(primary_book["returns"], t0_book["returns"], is_a_start, is_a_end)

    uplift_ta = sr_net_uplift(twin_a_book["returns"], t0_book["returns"], win)
    uplift_tc = sr_net_uplift(twin_c_book["returns"], t0_book["returns"], win)
    k5_4 = bool(np.isfinite(primary_uplift) and np.isfinite(uplift_ta) and np.isfinite(uplift_tc)
                and primary_uplift > uplift_ta and primary_uplift > uplift_tc)

    p95_tb = float(np.nanpercentile(tb_portfolio_uplifts, 95))
    k5_5 = bool(np.isfinite(primary_uplift) and primary_uplift > p95_tb)

    dsr = pooled_surface_dsr(primary_book["returns"], {k: b["returns"] for k, b in grid_books.items()},
                              is_a_start, is_a_end)

    return {
        "primary_uplift": primary_uplift, "K5_1": k5_1,
        "cell_uplifts": {f"kappa={k[0]}_lag={k[1]}": u for k, u in cell_uplifts.items()}, "K5_2": k5_2,
        "thirds": thirds, "K5_3": thirds["K5_3"],
        "uplift_ta": uplift_ta, "uplift_tc": uplift_tc, "K5_4": k5_4,
        "tb_portfolio_p95": p95_tb, "n_tb_portfolio_draws": int(len(tb_portfolio_uplifts)), "K5_5": k5_5,
        "is_dsr_pooled_surface": dsr,
        "passed": bool(k5_1 and k5_2 and thirds["K5_3"] and k5_4 and k5_5),
    }


# ---------------------------------------------------------------------------
# Steg 5b -- IS-B temporal confirmation (spec Sec.10)
# ---------------------------------------------------------------------------

def steg5b_temporal_confirmation(primary_returns: pd.Series, t0_returns: pd.Series,
                                  x_panel: pd.DataFrame, y_panel: pd.DataFrame,
                                  is_b_start, is_b_end) -> dict:
    uplift = sr_net_uplift(primary_returns, t0_returns, (is_b_start, is_b_end))
    k5b_1 = bool(np.isfinite(uplift) and uplift > 0)

    ic_series = weekly_rank_ic(x_panel.loc[is_b_start:is_b_end], y_panel.loc[is_b_start:is_b_end])
    mean_ic = float(ic_series.mean()) if len(ic_series) else float("nan")
    k5b_2 = bool(np.isfinite(mean_ic) and mean_ic > 0)

    return {
        "uplift": uplift, "K5b_1": k5b_1,
        "mean_ic": mean_ic, "K5b_2": k5b_2,
        "passed": bool(k5b_1 and k5b_2),
    }


# ---------------------------------------------------------------------------
# Steg 6 -- OOS (spec Sec.10, Sec.11)
# ---------------------------------------------------------------------------

def steg6_oos(main_returns: pd.Series, t0_returns: pd.Series, twin_a_returns: pd.Series,
              tb_portfolio_uplifts: np.ndarray, is_a_grid_returns: dict, is_a_start, is_a_end,
              oos_start, oos_end, n_trials: int = config.N_EFFECTIVE_SURFACE_READS) -> dict:
    """K6.4's DSR reuses the SAME pooled_surface_dsr construction as Steg 5a
    (Smittotalet tiling, N_trials=9): the trial pool is anchored to the
    IS-A 6-cell grid (where the actual specification search happened -- OOS
    itself reruns no grid, only T0/T-A/T-B per spec Sec.10), evaluated
    against the OOS primary cell's own OOS Sharpe/n_obs."""
    win = (oos_start, oos_end)
    uplift_main = sr_net_uplift(main_returns, t0_returns, win)
    k6_1 = bool(np.isfinite(uplift_main) and uplift_main > 0)

    uplift_ta = sr_net_uplift(twin_a_returns, t0_returns, win)
    k6_2 = bool(np.isfinite(uplift_main) and np.isfinite(uplift_ta) and uplift_main > uplift_ta)

    p95_tb = float(np.nanpercentile(tb_portfolio_uplifts, 95))
    k6_3 = bool(np.isfinite(uplift_main) and uplift_main > p95_tb)

    dsr = pooled_surface_dsr(main_returns.loc[oos_start:oos_end],
                              {k: v.loc[is_a_start:is_a_end] for k, v in is_a_grid_returns.items()},
                              oos_start, oos_end, n_effective_reads=n_trials)
    # "DSR > 0" (spec K6.4): DSR is a probability in [0,1] (lib.metrics.
    # deflated_sharpe_ratio via norm.cdf(z)), so a literal ">0" bound is
    # trivially satisfied by almost any nonzero-variance input and would not
    # gate anything. Read as the standard decision threshold DSR=0.5 <=> z=0
    # (observed Sharpe exceeds the expected max of N_trials null draws) --
    # the least-favorable-to-the-strategy reading that still makes K6.4 a
    # real criterion; see AVVIKELSER.md.
    k6_4 = bool(np.isfinite(dsr["dsr"]) and dsr["dsr"] > 0.5)

    return {
        "uplift_main": uplift_main, "K6_1": k6_1,
        "uplift_ta": uplift_ta, "K6_2": k6_2,
        "tb_portfolio_p95": p95_tb, "K6_3": k6_3,
        "dsr": dsr["dsr"], "dsr_z": dsr["z"], "n_trials": dsr["n_trials"], "K6_4": k6_4,
        "passed": bool(k6_1 and k6_2 and k6_3 and k6_4),
    }
