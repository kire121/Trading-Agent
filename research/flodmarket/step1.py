"""Steg 1 -- estimator null + IC on the handlade, signerade primary signal
(spec SS9, Ekolodet-mallkravet). IS only, primary cell (K=40, z*=2.0,
demean=252d)."""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from lib.metrics import newey_west_tstat

from . import config
from . import intrabar
from . import nulls
from . import signal


def pooled_weekly_rank_ic(g_wide: pd.DataFrame, vol_scaled_ret_wide: pd.DataFrame) -> float:
    g_flat = g_wide.stack()
    r_flat = vol_scaled_ret_wide.stack()
    common = pd.concat([g_flat, r_flat], axis=1, keys=["g", "r"]).dropna()
    if len(common) < 20:
        return float("nan")
    return float(scipy_stats.spearmanr(common["g"], common["r"]).correlation)


def weekly_ic_series(g_wide: pd.DataFrame, vol_scaled_ret_wide: pd.DataFrame) -> pd.Series:
    """Per-week cross-sectional Spearman IC (Fama-MacBeth convention), used
    for the NW-t significance test (K1.2)."""
    out = {}
    for date in g_wide.index:
        if date not in vol_scaled_ret_wide.index:
            continue
        g_row = g_wide.loc[date].dropna()
        r_row = vol_scaled_ret_wide.loc[date].dropna()
        common = g_row.index.intersection(r_row.index)
        if len(common) < 5:
            continue
        ic = scipy_stats.spearmanr(g_row.loc[common], r_row.loc[common]).correlation
        out[date] = ic
    return pd.Series(out).sort_index()


def run_steg1(shadow: pd.DataFrame, remaining_tickers: list, adjC_wide: pd.DataFrame,
              seed: int = config.GLOBAL_SEED, n_null_draws: int = config.STEG1_N_NULL_DRAWS) -> dict:
    s = shadow["s"].loc[shadow.index.get_level_values("ticker").isin(remaining_tickers)]
    s_tilde = intrabar.fe_demean(s, config.K_PRIMARY, window=config.FE_DEMEAN_WINDOW)
    S = intrabar.rolling_tstat(s_tilde, config.K_PRIMARY, min_valid=config.K_EFF_MIN_FRACTION)
    g = signal.compute_g(S, config.Z_STAR_PRIMARY)
    g_wide = g.unstack("ticker")

    decision_dates = signal.weekly_decision_dates(adjC_wide.index)
    from . import returns as returns_mod
    sigma_hat = signal.annualized_log_return_vol(adjC_wide[remaining_tickers])
    fwd = returns_mod.weekly_execution_returns(adjC_wide[remaining_tickers], decision_dates)
    vol_scaled = returns_mod.vol_scaled_returns(fwd, sigma_hat)

    g_at_decisions = g_wide.reindex(vol_scaled.index)

    observed_ic = pooled_weekly_rank_ic(g_at_decisions, vol_scaled)
    ic_series = weekly_ic_series(g_at_decisions, vol_scaled)
    nw = newey_west_tstat(ic_series, lags=config.STEG1_NW_MAXLAGS)

    rng = np.random.default_rng(seed + 20_000_003)
    null_ics = np.empty(n_null_draws)
    for i in range(n_null_draws):
        s_perm = nulls.block_permute_within_ticker(s, config.STEG1_NULL_BLOCK_DAYS, rng)
        s_tilde_perm = intrabar.fe_demean(s_perm, config.K_PRIMARY, window=config.FE_DEMEAN_WINDOW)
        S_perm = intrabar.rolling_tstat(s_tilde_perm, config.K_PRIMARY, min_valid=config.K_EFF_MIN_FRACTION)
        g_perm = signal.compute_g(S_perm, config.Z_STAR_PRIMARY)
        g_perm_at_decisions = g_perm.unstack("ticker").reindex(vol_scaled.index)
        null_ics[i] = pooled_weekly_rank_ic(g_perm_at_decisions, vol_scaled)
    null_ics_valid = null_ics[np.isfinite(null_ics)]
    null_p95 = float(np.percentile(null_ics_valid, config.STEG1_NULL_PERCENTILE)) if len(null_ics_valid) else float("nan")
    null_mean = float(np.mean(null_ics_valid)) if len(null_ics_valid) else float("nan")

    k1_1 = bool(np.isfinite(observed_ic) and observed_ic >= config.STEG1_MIN_POOLED_RANK_IC)
    k1_2 = bool(np.isfinite(nw) and nw >= config.STEG1_MIN_NW_T)
    k1_3 = bool(np.isfinite(observed_ic) and np.isfinite(null_p95) and observed_ic > null_p95)

    passes = bool(k1_1 and k1_2 and k1_3)

    return {
        "observed_pooled_ic": observed_ic,
        "n_weekly_ic_obs": int(ic_series.notna().sum()),
        "nw_t": nw,
        "null_p95": null_p95,
        "null_mean": null_mean,
        "n_null_draws": n_null_draws,
        "K1.1_pooled_ic_ge_0.015": k1_1,
        "K1.2_nw_t_ge_2.5": k1_2,
        "K1.3_ic_gt_null_p95": k1_3,
        "passes": passes,
    }
