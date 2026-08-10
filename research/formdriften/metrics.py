"""Performance / statistical-significance metrics shared across the pipeline."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649


def ann_return(returns):
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    return r.mean() * TRADING_DAYS


def ann_vol(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sortino(returns, target=0.0):
    r = pd.Series(returns).dropna()
    downside = r[r < target]
    if len(downside) == 0:
        return np.nan
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return np.nan
    return (r.mean() - target) / dd * np.sqrt(TRADING_DAYS)


def max_drawdown(returns):
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return np.nan
    curve = (1 + r).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1
    return dd.min()


def newey_west_tstat(returns, lags=None):
    """NW/HAC t-stat that the mean daily return is 0, via OLS on a constant."""
    r = pd.Series(returns).dropna()
    if len(r) < 10:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (len(r) / 100) ** (2 / 9)))  # Newey-West (1994) rule of thumb
    X = np.ones((len(r), 1))
    model = sm.OLS(r.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    return float(model.tvalues[0])


def pnl_quarter_concentration(returns):
    """Max fraction of total PnL (sum of returns) attributable to a single calendar quarter."""
    r = pd.Series(returns).dropna()
    if len(r) == 0 or r.sum() == 0:
        return np.nan
    q = r.groupby([r.index.year, r.index.quarter]).sum()
    total = r.sum()
    if total == 0:
        return np.nan
    # use abs-normalized share so both concentrated gains and concentrated
    # losses trigger the kill check symmetrically
    return (q.abs().max()) / q.abs().sum() if q.abs().sum() != 0 else np.nan


def probabilistic_sharpe_ratio(sr_hat, benchmark_sr, n_obs, skew=0.0, kurtosis=3.0):
    """PSR(SR*): probability the true Sharpe exceeds benchmark_sr (Bailey & Lopez de Prado 2012)."""
    excess_kurt = kurtosis - 3.0
    denom = np.sqrt(max(1 - skew * sr_hat + (excess_kurt / 4.0) * sr_hat ** 2, 1e-12))
    z = (sr_hat - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return stats.norm.cdf(z)


def expected_max_sharpe(sr_trials):
    """E[max SR] under N independent trials, from the variance of the trial Sharpes
    (Bailey & Lopez de Prado 2014 deflated-Sharpe benchmark)."""
    sr_trials = np.asarray([s for s in sr_trials if np.isfinite(s)])
    n = len(sr_trials)
    if n < 2:
        return 0.0
    var_sr = np.var(sr_trials, ddof=1)
    if var_sr <= 0:
        return float(np.mean(sr_trials))
    sigma_sr = np.sqrt(var_sr)
    term1 = (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1.0 / n)
    term2 = EULER_MASCHERONI * stats.norm.ppf(1 - 1.0 / (n * np.e))
    return sigma_sr * (term1 + term2)


def deflated_sharpe_ratio(sr_hat, n_obs, sr_trials, skew=0.0, kurtosis=3.0):
    """DSR: probability the strategy's true (daily, per-period) Sharpe exceeds the
    Sharpe you'd expect from the best of `len(sr_trials)` random/noise variants,
    given the annualization-consistent per-period sr_hat and n_obs periods.

    sr_hat and sr_trials must be on the SAME (unannualized, per-period) scale.
    """
    sr0 = expected_max_sharpe(sr_trials)
    return probabilistic_sharpe_ratio(sr_hat, sr0, n_obs, skew, kurtosis)


def turnover_annualized(turnover_series, periods_per_year=12):
    t = pd.Series(turnover_series).dropna()
    if len(t) == 0:
        return np.nan
    return t.mean() * periods_per_year


def information_coefficient(signal, fwd_return):
    """Cross-sectional Spearman IC between a signal and forward returns (aligned Series)."""
    df = pd.concat([signal, fwd_return], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    return df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
