"""Performance/risk statistics.

DSR/PSR follow the Bailey & Lopez de Prado (2014) formulation, copy-
structurally identical across every sibling research branch in this repo
(dammluckan/metrics.py, oglegrinden/stats.py, omori/metrics.py) --
reproduced here in the same form. Newey-West t-stats use statsmodels HAC.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm

from . import config

TRADING_DAYS = config.TRADING_DAYS_YEAR
EULER_MASCHERONI = 0.5772156649015329


def ann_return(returns: pd.Series) -> float:
    r = returns.dropna()
    return r.mean() * TRADING_DAYS if len(r) else np.nan


def ann_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if not len(r):
        return np.nan
    curve = (1.0 + r).cumprod()
    dd = curve / curve.cummax() - 1.0
    return dd.min()


def newey_west_tstat(returns: pd.Series, lags=None) -> float:
    """HAC-robust t-stat that the mean (per-period) return is 0."""
    r = returns.dropna()
    if len(r) < 5:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (len(r) / 100) ** (2 / 9)))
    ones = np.ones(len(r))
    model = sm.OLS(r.to_numpy(), ones).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    return float(model.tvalues[0])


def probabilistic_sharpe_ratio(sr_hat: float, benchmark_sr: float, n_obs: int,
                                skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012). sr_hat/benchmark_sr must be on the
    SAME (per-period, unannualized) scale."""
    if n_obs < 2:
        return np.nan
    excess_kurt = kurtosis - 3.0
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + (excess_kurt / 4.0) * sr_hat ** 2))
    z = (sr_hat - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(scipy_stats.norm.cdf(z))


def expected_max_sharpe(sr_trials) -> float:
    """E[max Sharpe] under N independent noise trials (extreme-value
    approximation), on the per-period scale of sr_trials."""
    sr_trials = np.asarray(sr_trials, dtype=float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    n = len(sr_trials)
    if n < 2:
        raise ValueError("expected_max_sharpe requires at least 2 trials")
    sigma_sr = np.std(sr_trials, ddof=1)
    return sigma_sr * ((1 - EULER_MASCHERONI) * scipy_stats.norm.ppf(1 - 1.0 / n)
                        + EULER_MASCHERONI * scipy_stats.norm.ppf(1 - 1.0 / (n * np.e)))


def deflated_sharpe_ratio(sr_hat: float, n_obs: int, sr_trials, skew: float = 0.0,
                           kurtosis: float = 3.0) -> dict:
    """DSR probability (PSR against the expected-max-under-the-null
    benchmark) plus the Sharpe-scaled excess gap. sr_hat and sr_trials must
    both be per-period Sharpes."""
    sr0 = expected_max_sharpe(sr_trials)
    psr = probabilistic_sharpe_ratio(sr_hat, sr0, n_obs, skew, kurtosis)
    return {"dsr_prob": psr, "dsr_excess": sr_hat - sr0, "expected_max_sr": sr0}


def information_coefficient(signal_vals, forward_returns) -> float:
    """Spearman rank correlation between a signal and forward returns."""
    df = pd.concat([pd.Series(signal_vals), pd.Series(forward_returns)], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    rho, _ = scipy_stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return rho


def sub_period_boundaries(start, end, n_periods=config.N_SUBPERIODS):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    edges = pd.date_range(start, end, periods=n_periods + 1)
    return [(edges[i], edges[i + 1]) for i in range(n_periods)]


def sub_period_sign_consistency(returns: pd.Series, start, end, n_periods=config.N_SUBPERIODS) -> dict:
    """Sign of annualized return in each of n_periods equal calendar
    sub-periods vs. the full-sample sign."""
    full_sign = np.sign(ann_return(returns))
    boundaries = sub_period_boundaries(start, end, n_periods)
    signs, ann_rets = [], []
    for lo, hi in boundaries:
        sub = returns.loc[lo:hi]
        ar = ann_return(sub)
        ann_rets.append(ar)
        signs.append(np.sign(ar) if np.isfinite(ar) else 0.0)
    n_consistent = sum(1 for s in signs if s == full_sign and s != 0)
    consistency = n_consistent / n_periods if n_periods else np.nan
    return {"full_sign": full_sign, "period_signs": signs, "period_ann_returns": ann_rets,
            "consistency": consistency, "boundaries": boundaries}
