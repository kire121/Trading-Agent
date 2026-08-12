"""Pre-registered null/robustness/validation machinery:

(a) stationary block-bootstrap null on the weekly Z path -> p-value for Sharpe
(b) static 50/50 trend+meanrev mix baseline (no switching)
(c) vol-z-score regime baseline (switching, but not on irreversibility)
    -- both (b)/(c) built in variants.py, compared here
orthogonalization of Z against rolling vol / skew / |r|-autocorrelation
Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014 / Mertens 2002 SE)
correlation vs a pure TSMOM sleeve (hard reject if > 0.7)
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config, signal, strategy, backtest, variants


# --- (a) Stationary block bootstrap null ---------------------------------

def _stationary_bootstrap_index(n, mean_block_len, rng):
    """Politis & Romano (1994) stationary bootstrap resampling index, with
    circular wrap-around, expected block length `mean_block_len`."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block_len
    i = 0
    while i < n:
        start = rng.integers(0, n)
        L = rng.geometric(p)
        take = min(L, n - i)
        idx[i:i + take] = (start + np.arange(take)) % n
        i += take
    return idx


def bootstrap_null_sharpe(px_df, returns_df, z_weekly, anchors, threshold, n_runs=config.BOOTSTRAP_RUNS,
                           block_len_days=config.BOOTSTRAP_BLOCK_LEN, seed=0):
    """Block-bootstrap the WEEKLY Z path (same index applied jointly across
    all instruments, to preserve cross-sectional structure), re-run the
    regime classifier + direction rule + sizing + backtest on each draw, and
    build a null distribution of full-sample Sharpe ratios. The real price
    series (and hence the trend/meanrev direction rules) are left untouched
    -- only the *timing* of which weeks are TREND vs MEANREV is randomized,
    with block persistence similar to the real Z path's own autocorrelation.
    """
    rng = np.random.default_rng(seed)
    block_len_weeks = max(1, round(block_len_days / 5))
    n = len(z_weekly)

    trend_dir = signal.trend_direction(px_df).loc[anchors]
    meanrev_dir = signal.meanrev_direction(px_df).loc[anchors]

    null_sharpes = np.full(n_runs, np.nan)
    for r in range(n_runs):
        idx = _stationary_bootstrap_index(n, block_len_weeks, rng)
        z_shuffled = pd.DataFrame(z_weekly.values[idx], index=z_weekly.index, columns=z_weekly.columns)
        regime = signal.classify_regime_panel(z_shuffled, upper=threshold)
        direction = signal.combine_direction(regime, trend_dir, meanrev_dir)
        weekly_w = strategy.build_weekly_weights(direction, returns_df, anchors)
        bt = backtest.run_backtest(px_df, returns_df, weekly_w)
        null_sharpes[r] = backtest.sharpe_ratio(bt["portfolio_return"])

    return null_sharpes


def bootstrap_p_value(observed_sharpe, null_sharpes):
    null_sharpes = null_sharpes[~np.isnan(null_sharpes)]
    if len(null_sharpes) == 0:
        return np.nan
    return float((null_sharpes >= observed_sharpe).sum() / len(null_sharpes))


# --- Orthogonalization -----------------------------------------------------

def rolling_skew(returns_df, lookback=63):
    return returns_df.rolling(lookback, min_periods=lookback).skew()


def rolling_abs_return_autocorr(returns_df, lookback=63, lag=1):
    abs_r = returns_df.abs()

    def _ac(x):
        if x.std(ddof=0) == 0 or len(x) < lag + 2:
            return np.nan
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]

    return abs_r.rolling(lookback, min_periods=lookback).apply(_ac, raw=True)


def orthogonalize_z(z_weekly, returns_df, anchors, vol_lookback=config.VOL_LOOKBACK):
    """Regress the weekly Z panel (pooled across instruments) on rolling vol
    z-score, rolling skew, and rolling |r| autocorrelation; return the
    re-standardized residual panel (same shape as z_weekly). If no
    incremental signal survives in the residual, the hypothesis is
    considered dead per the lab's pre-registered kill criterion.
    """
    vol_daily = strategy.instrument_vol(returns_df, lookback=vol_lookback)
    vol_z_daily = signal.rolling_zscore(vol_daily, history=config.Z_HISTORY)
    skew_daily = rolling_skew(returns_df)
    autocorr_daily = rolling_abs_return_autocorr(returns_df)

    vol_z_w = vol_z_daily.loc[anchors]
    skew_w = skew_daily.loc[anchors]
    autocorr_w = autocorr_daily.loc[anchors]

    resid = pd.DataFrame(index=z_weekly.index, columns=z_weekly.columns, dtype=float)
    r2_by_col = {}
    for col in z_weekly.columns:
        y = z_weekly[col]
        X = pd.DataFrame({
            "vol_z": vol_z_w[col],
            "skew": skew_w[col],
            "abs_r_autocorr": autocorr_w[col],
        })
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(df) < 30:
            resid[col] = np.nan
            r2_by_col[col] = np.nan
            continue
        Xm = np.column_stack([np.ones(len(df)), df[["vol_z", "skew", "abs_r_autocorr"]].values])
        beta, *_ = np.linalg.lstsq(Xm, df["y"].values, rcond=None)
        fitted = Xm @ beta
        res = df["y"].values - fitted
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((df["y"].values - df["y"].values.mean()) ** 2)
        r2_by_col[col] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        res_series = pd.Series(res, index=df.index)
        std = res_series.std(ddof=0)
        resid.loc[df.index, col] = (res_series / std) if std > 0 else res_series

    return resid, r2_by_col


# --- Deflated Sharpe Ratio --------------------------------------------------

def _euler_mascheroni():
    return 0.5772156649015329


def expected_max_sharpe_under_null(trial_sharpes, n_trials_effective):
    """SR0: expected maximum Sharpe ratio across `n_trials_effective`
    independent, zero-skill trials, using the variance of Sharpes actually
    observed across the robustness grid as the estimate of V[SR_n]
    (Bailey & Lopez de Prado 2014, eq. for E[max SR] under the null).
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    trial_sharpes = trial_sharpes[~np.isnan(trial_sharpes)]
    if len(trial_sharpes) < 2:
        return np.nan
    var_sr = trial_sharpes.var(ddof=1)
    gamma = _euler_mascheroni()
    n = max(n_trials_effective, 2)
    z1 = stats.norm.ppf(1 - 1.0 / n)
    z2 = stats.norm.ppf(1 - 1.0 / (n * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * z1 + gamma * z2)
    return float(sr0)


def deflated_sharpe(observed_sharpe, periodic_returns, trial_sharpes,
                     n_trials_effective=config.N_CONFIGS_FOR_DSR, periods_per_year=52):
    """Returns dict with:
      sr0            : expected max Sharpe under a skill-less null given N trials
      dsr_excess     : observed annualized Sharpe minus sr0 (annualized) -- the
                       quantity the lab's pre-registered rule rejects on <= 0
      psr            : probabilistic Sharpe ratio, i.e. P(true SR > 0) using the
                        Mertens/Bailey-Lopez de Prado standard error, informative
                        but not itself the reject/accept criterion
      dsr_prob       : P(true SR > sr0) -- the fully "deflated" probability
    """
    r = periodic_returns.dropna()
    T = len(r)
    if T < 10:
        return {"sr0": np.nan, "dsr_excess": np.nan, "psr": np.nan, "dsr_prob": np.nan}

    sr_period = r.mean() / r.std(ddof=0) if r.std(ddof=0) > 0 else np.nan
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # non-excess (normal = 3)

    se_sr = np.sqrt(max(1e-12, (1 - skew * sr_period + (kurt - 1) / 4 * sr_period ** 2) / (T - 1)))

    sr0_period = expected_max_sharpe_under_null(trial_sharpes, n_trials_effective) / np.sqrt(periods_per_year) \
        if not np.isnan(expected_max_sharpe_under_null(trial_sharpes, n_trials_effective)) else np.nan

    psr = float(stats.norm.cdf(sr_period / se_sr)) if se_sr > 0 else np.nan
    dsr_prob = float(stats.norm.cdf((sr_period - (sr0_period if not np.isnan(sr0_period) else 0)) / se_sr)) \
        if se_sr > 0 else np.nan

    sr0_annual = sr0_period * np.sqrt(periods_per_year) if not np.isnan(sr0_period) else np.nan
    dsr_excess = observed_sharpe - sr0_annual if not np.isnan(sr0_annual) else np.nan

    return {"sr0": sr0_annual, "dsr_excess": dsr_excess, "psr": psr, "dsr_prob": dsr_prob}


# --- TSMOM correlation hard limit ------------------------------------------

def tsmom_correlation(strategy_returns, tsmom_returns):
    df = pd.concat([strategy_returns.rename("strat"), tsmom_returns.rename("tsmom")], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    return float(df["strat"].corr(df["tsmom"]))
