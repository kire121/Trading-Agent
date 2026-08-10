"""Performance metrics: Sharpe, drawdown, and the Bailey & Lopez de Prado
probabilistic/deflated Sharpe ratio -- copy-structurally identical to
research/dammluckan/metrics.py (itself matching fasflocken/oglegrinden/
irreversibility_lab/formdriften), the one canonical implementation used
throughout this repo's research-branch series.
"""
import numpy as np
from scipy import stats as scipy_stats

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS_YEAR = 252


def annualized_sharpe(daily_returns, trading_days=TRADING_DAYS_YEAR):
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(trading_days))


def max_drawdown(daily_returns):
    r = np.asarray(daily_returns, dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def probabilistic_sharpe_ratio(sr_hat, benchmark_sr, n_obs, skew=0.0, kurtosis=3.0):
    excess_kurt = kurtosis - 3.0
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + (excess_kurt / 4.0) * sr_hat ** 2))
    z = (sr_hat - benchmark_sr) * np.sqrt(max(n_obs - 1, 1)) / denom
    return float(scipy_stats.norm.cdf(z))


def expected_max_sharpe(sr_trials):
    sr_trials = np.asarray(sr_trials, dtype=float)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    n = len(sr_trials)
    if n < 2:
        return 0.0
    sigma_sr = np.std(sr_trials, ddof=1)
    if sigma_sr == 0:
        return float(np.mean(sr_trials))
    return float(sigma_sr * ((1 - EULER_MASCHERONI) * scipy_stats.norm.ppf(1 - 1.0 / n)
                              + EULER_MASCHERONI * scipy_stats.norm.ppf(1 - 1.0 / (n * np.e))))


def deflated_sharpe_ratio(sr_hat, n_obs, sr_trials, skew=0.0, kurtosis=3.0):
    sr0 = expected_max_sharpe(sr_trials)
    return {
        "dsr_prob": probabilistic_sharpe_ratio(sr_hat, sr0, n_obs, skew, kurtosis),
        "dsr_excess": sr_hat - sr0,
        "expected_max_sr": sr0,
    }


def hit_rate(net_returns):
    r = np.asarray(net_returns, dtype=float)
    r = r[np.isfinite(r)]
    return float((r > 0).mean()) if len(r) else float("nan")


def summarize(daily_returns, closed_events_df=None, n_prior_trials=0):
    sr = annualized_sharpe(daily_returns)
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    out = {
        "sharpe": sr,
        "annualized_return": float(r.mean() * TRADING_DAYS_YEAR),
        "annualized_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR)) if len(r) > 1 else 0.0,
        "max_drawdown": max_drawdown(daily_returns),
        "n_days": int(len(r)),
        "skew": float(scipy_stats.skew(r)) if len(r) > 2 else 0.0,
        "kurtosis": float(scipy_stats.kurtosis(r, fisher=False)) if len(r) > 2 else 3.0,
    }
    if closed_events_df is not None and len(closed_events_df):
        out["n_events"] = int(len(closed_events_df))
        out["hit_rate"] = hit_rate(closed_events_df["net_return"])
        out["avg_holding_days"] = float(closed_events_df["holding_days"].mean())
        out["exit_reason_counts"] = closed_events_df["exit_reason"].value_counts().to_dict()
    return out
