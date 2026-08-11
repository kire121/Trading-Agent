"""Shared statistics: IC, portfolio Sharpe, PC1 variance share, Deflated Sharpe Ratio."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52.0


def pooled_ic(pred: pd.Series, target: pd.Series) -> float:
    df = pd.DataFrame({"pred": pred, "target": target}).dropna()
    if len(df) < 10 or df["pred"].std() == 0 or df["target"].std() == 0:
        return np.nan
    return float(df["pred"].corr(df["target"]))


def oof_r2(pred: pd.Series, target: pd.Series) -> float:
    df = pd.DataFrame({"pred": pred, "target": target}).dropna()
    if df.empty:
        return np.nan
    sse = float(np.sum((df["target"] - df["pred"]) ** 2))
    sst = float(np.sum((df["target"] - df["target"].mean()) ** 2))
    if sst == 0:
        return np.nan
    return 1.0 - sse / sst


def portfolio_weekly_returns(positions: pd.DataFrame, cost_bp: float = 0.0) -> pd.Series:
    """Aggregate per-(asset,week) `weight` * `next_week_return` into a single
    portfolio weekly return series, net of a simple one-way turnover cost in
    basis points.

    Grouped by the ISO calendar week of `t_signal` (not the raw
    `execution_date`) so that an asset whose own execution_date drifts by a
    day or two from its peers (e.g. one ETF missing a single trading day)
    still lands in the same weekly portfolio bucket as everyone else, rather
    than silently forming its own single-name "week". See positions.py's
    module docstring for the matching rationale in position sizing itself.
    """
    df = positions.copy()
    iso = df["t_signal"].dt.isocalendar()
    df["week_bucket"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    bucket_date = df.groupby("week_bucket")["execution_date"].min()

    df["contrib"] = df["weight"] * df["next_week_return"]
    gross_by_week = df.groupby("week_bucket")["contrib"].sum()

    df_sorted = df.sort_values(["asset", "execution_date"])
    df_sorted["prev_weight"] = df_sorted.groupby("asset")["weight"].shift(1).fillna(0.0)
    df_sorted["turnover"] = (df_sorted["weight"] - df_sorted["prev_weight"]).abs()
    cost_by_week = df_sorted.groupby("week_bucket")["turnover"].sum() * (cost_bp / 10000.0)

    net = gross_by_week.sub(cost_by_week, fill_value=0.0)
    net.index = bucket_date.reindex(net.index)
    return net.sort_index()


def sharpe_ratio(weekly_returns: pd.Series, periods_per_year: float = WEEKS_PER_YEAR) -> float:
    r = weekly_returns.dropna()
    if len(r) < 10 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(weekly_returns: pd.Series) -> float:
    r = weekly_returns.dropna()
    if r.empty:
        return np.nan
    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def pc1_share(matrix: np.ndarray) -> float:
    """Share of total variance explained by the first principal component of
    a (T, N) matrix (columns mean-centred; NaNs treated as 0 after
    centring, i.e. "no signal this week" contributes no deviation)."""
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    X = matrix.copy()
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    X = X - X.mean(axis=0, keepdims=True)
    col_var = X.var(axis=0)
    if np.allclose(col_var, 0):
        return np.nan
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total <= 0:
        return np.nan
    return float(eigvals.max() / total)


def _skew(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return 3.0
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 3.0
    return float(np.mean(((x - m) / s) ** 4))


def deflated_sharpe_ratio(weekly_returns: pd.Series, n_trials: int,
                           periods_per_year: float = WEEKS_PER_YEAR,
                           var_sr_across_trials: float | None = None) -> dict:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Returns a dict with the annualised Sharpe, its non-normality-adjusted
    standard error, the implied SR* benchmark (expected max Sharpe under
    n_trials independent trials) and the resulting DSR (a probability in
    [0, 1], P(true SR > 0 | observed, accounting for selection over
    n_trials)).
    """
    r = weekly_returns.dropna().to_numpy()
    n = len(r)
    out = {"sharpe_annual": np.nan, "n_obs": n, "n_trials": n_trials,
           "sr_benchmark_annual": np.nan, "dsr": np.nan}
    if n < 20:
        return out

    sr_weekly = r.mean() / r.std() if r.std() > 0 else np.nan
    sr_annual = sr_weekly * np.sqrt(periods_per_year) if not np.isnan(sr_weekly) else np.nan
    out["sharpe_annual"] = sr_annual
    if np.isnan(sr_annual):
        return out

    gamma3 = _skew(r)
    gamma4 = _kurtosis(r)

    # Expected maximum Sharpe (weekly units) across n_trials independent trials
    # with per-trial SR variance `var_sr_across_trials` (defaults to the single
    # observed trial's own SR variance -- a conservative, standard fallback).
    if var_sr_across_trials is None:
        var_sr_across_trials = (1 - gamma3 * sr_weekly + (gamma4 - 1) / 4.0 * sr_weekly ** 2) / (n - 1)
    var_sr_across_trials = max(var_sr_across_trials, 1e-12)
    sr_std = np.sqrt(var_sr_across_trials)

    euler_gamma = 0.5772156649015329
    if n_trials <= 1:
        sr_benchmark_weekly = 0.0
    else:
        sr_benchmark_weekly = sr_std * (
            (1 - euler_gamma) * _inv_norm_cdf(1 - 1.0 / n_trials)
            + euler_gamma * _inv_norm_cdf(1 - 1.0 / (n_trials * np.e))
        )
    out["sr_benchmark_annual"] = sr_benchmark_weekly * np.sqrt(periods_per_year)

    se_sr = np.sqrt(max((1 - gamma3 * sr_weekly + (gamma4 - 1) / 4.0 * sr_weekly ** 2) / (n - 1), 1e-12))
    z = (sr_weekly - sr_benchmark_weekly) / se_sr
    out["dsr"] = float(_norm_cdf(z))
    return out


def _norm_cdf(x: float) -> float:
    from scipy.stats import norm
    return float(norm.cdf(x))


def _inv_norm_cdf(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))
