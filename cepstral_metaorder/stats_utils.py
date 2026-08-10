"""
Small numpy-only statistics helpers (Newey-West HAC t-stats, deflated Sharpe
ratio) so the validation battery doesn't need statsmodels or any dependency
beyond numpy/scipy/pandas, per the spec's explicit "no exotic dependencies"
constraint.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def newey_west_mean_tstat(x: np.ndarray, lags: int) -> dict:
    """HAC (Bartlett-kernel) standard error of a sample mean -- equivalent to
    Newey-West SE from a regression of x_t on a constant only. Used to test
    whether the time series of daily Fama-MacBeth coefficients is
    significantly different from zero once their own serial correlation
    (induced by e.g. overlapping 5-day forward returns) is accounted for."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    t = len(x)
    if t < max(3, lags + 2):
        return {"mean": np.nan, "se": np.nan, "t_stat": np.nan, "n": t}
    xbar = x.mean()
    dev = x - xbar
    gamma0 = np.mean(dev * dev)
    s = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma_l = np.mean(dev[lag:] * dev[:-lag])
        s += 2 * weight * gamma_l
    s = max(s, 1e-18)
    se = np.sqrt(s / t)
    tstat = xbar / se if se > 0 else np.nan
    return {"mean": float(xbar), "se": float(se), "t_stat": float(tstat), "n": int(t)}


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, trial_sharpes: np.ndarray,
                           periods_per_year: int) -> dict:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio: probability that
    the observed (per-period) Sharpe exceeds the Sharpe expected from the best
    of `n_trials` independent zero-skill strategies, given the actual
    return series' skew/kurtosis and the empirical spread of Sharpe ratios
    observed ACROSS the tau-window x holding-period grid (trial_sharpes) as
    the estimate of the null's cross-trial Sharpe standard deviation."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    t = len(returns)
    if t < 10:
        return {"sr": np.nan, "sr_annual": np.nan, "sr_benchmark": np.nan, "dsr": np.nan, "n": t}

    sr = returns.mean() / returns.std(ddof=1) if returns.std(ddof=1) > 0 else np.nan
    sr_annual = sr * np.sqrt(periods_per_year)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=False)  # regular (non-excess) kurtosis

    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    trial_sharpes = trial_sharpes[~np.isnan(trial_sharpes)]
    sigma_sr = trial_sharpes.std(ddof=1) if len(trial_sharpes) >= 2 else abs(sr) * 0.5
    sigma_sr = max(sigma_sr, 1e-6)

    euler_mascheroni = 0.5772156649
    n = max(int(n_trials), 2)
    sr_benchmark = sigma_sr * (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n)
        + euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n * np.e))
    )

    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-9))
    z = (sr - sr_benchmark) * np.sqrt(t - 1) / denom
    dsr = float(stats.norm.cdf(z))

    return {
        "sr": float(sr), "sr_annual": float(sr_annual), "sr_benchmark_per_period": float(sr_benchmark),
        "skew": float(skew), "kurtosis": float(kurt), "n_trials": n, "sigma_sr_across_trials": float(sigma_sr),
        "dsr": dsr, "n": int(t),
    }
