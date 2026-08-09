"""Performance statistics and null-hypothesis tests.

Implements the three declared null-hypothesis baselines from the strategy
brief:
  (a) stationary block bootstrap of the gate series (does gate *placement*
      matter beyond an equally-persistent random ON/OFF pattern?)
  (b) always-on baseline comparison (must beat both Sharpe and drawdown)
  (c) twin gates: same rules, different raw signal (rho_bar, absorption
      ratio, index vol) -- the topology signal must add explanatory power
      beyond these in a regression that controls for them.

Also implements the deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)
for the declared ~30-variant parameter grid.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Basic performance statistics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 52, annualize: bool = True) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    return float(sr * np.sqrt(periods_per_year)) if annualize else float(sr)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 52, target: float = 0.0) -> float:
    r = returns.dropna()
    downside = r[r < target]
    if len(downside) == 0:
        return float("inf") if r.mean() > target else 0.0
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return 0.0
    sortino = (r.mean() - target) / dd
    return float(sortino * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.fillna(0.0)
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 52) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    wealth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or wealth <= 0:
        return float("nan")
    return float(wealth ** (1.0 / years) - 1.0)


def summary_stats(returns: pd.Series, periods_per_year: int = 52) -> dict:
    r = returns.dropna()
    return {
        "n_obs": int(len(r)),
        "mean_period_return": float(r.mean()) if len(r) else 0.0,
        "annualized_return": annualized_return(r, periods_per_year),
        "annualized_vol": float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0,
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "skew": float(sstats.skew(r)) if len(r) > 2 else 0.0,
        "kurtosis": float(sstats.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0,
        "hit_rate": float((r > 0).mean()) if len(r) else 0.0,
    }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
# ---------------------------------------------------------------------------

def expected_max_sharpe_under_null(trial_sharpes: Sequence[float]) -> dict:
    """SR0: the expected maximum Sharpe ratio across N independent trials
    under the null of zero true skill, given the empirical variance of the
    N observed (per-period) trial Sharpe ratios (extreme-value
    approximation used by Bailey & Lopez de Prado).
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[~np.isnan(trials)]
    n = len(trials)
    if n < 2:
        raise ValueError("need >=2 trials to estimate the null distribution of Sharpe ratios")
    var_sr = float(trials.var(ddof=1))
    if var_sr <= 0:
        sr0 = 0.0
    else:
        sr0 = float(
            np.sqrt(var_sr)
            * (
                (1 - EULER_GAMMA) * sstats.norm.ppf(1 - 1.0 / n)
                + EULER_GAMMA * sstats.norm.ppf(1 - 1.0 / (n * np.e))
            )
        )
    return {"sr0": sr0, "var_sr": var_sr, "n_trials": n}


def deflated_sharpe_ratio(
    observed_sharpe_per_period: float,
    trial_sharpes_per_period: Sequence[float],
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """DSR = P(true SR > 0 | observed SR is the max of N trials), i.e. the
    probability the observed Sharpe ratio exceeds the Sharpe ratio expected
    from pure luck given N independent trials, sample length `n_obs`, and
    the return distribution's skew/(non-excess) kurtosis.

    All Sharpe ratios here are *per-period* (not annualized) -- the
    annualization factor cancels inside the PSR z-statistic, so mixing
    annualized and per-period values would silently bias the result.
    """
    null = expected_max_sharpe_under_null(trial_sharpes_per_period)
    sr0 = null["sr0"]
    denom = np.sqrt(
        max(1e-12, 1 - skewness * observed_sharpe_per_period + ((kurtosis - 1) / 4.0) * observed_sharpe_per_period ** 2)
    )
    z = (observed_sharpe_per_period - sr0) * np.sqrt(max(n_obs - 1, 1)) / denom
    dsr = float(sstats.norm.cdf(z))
    return {"dsr": dsr, "z": float(z), "sr0": sr0, "n_obs": n_obs, **{k: v for k, v in null.items() if k != "sr0"}}


# ---------------------------------------------------------------------------
# (a) Stationary block bootstrap null test on the gate series
# ---------------------------------------------------------------------------

def _stationary_bootstrap_indices(n: int, block_size: float, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index sequence: random
    starting points, geometric block lengths with mean `block_size`,
    circular wrap-around, concatenated to length n.
    """
    p = 1.0 / block_size
    idx = np.empty(n, dtype=int)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        length = min(length, n - pos)
        for k in range(length):
            idx[pos + k] = (start + k) % n
        pos += length
    return idx


def block_bootstrap_gate_test(
    gate: pd.Series,
    underlying_returns: pd.Series,
    n_boot: int = 1000,
    block_size: float = 13.0,
    periods_per_year: int = 52,
    seed: int = 0,
) -> dict:
    """Null test (a): is the *placement* of the topology gate on top of the
    reversal engine better than a random ON/OFF pattern with the same
    marginal ON-fraction and the same block/run-length persistence?

    `underlying_returns` should be the always-on baseline's net weekly
    returns (same universe/formation/cost mechanics every week); `gate`
    the primary hysteresis gate state aligned to the same dates.
    """
    idx = gate.index.intersection(underlying_returns.index)
    gate_bool = gate.reindex(idx).fillna(False).values.astype(bool)
    rets = underlying_returns.reindex(idx).fillna(0.0).values
    n = len(idx)

    actual_returns = np.where(gate_bool, rets, 0.0)
    actual_sharpe = sharpe_ratio(pd.Series(actual_returns), periods_per_year=periods_per_year)

    rng = np.random.default_rng(seed)
    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        bidx = _stationary_bootstrap_indices(n, block_size, rng)
        boot_gate = gate_bool[bidx]
        boot_returns = np.where(boot_gate, rets, 0.0)
        boot_sharpes[b] = sharpe_ratio(pd.Series(boot_returns), periods_per_year=periods_per_year)

    p_value = float(np.mean(boot_sharpes >= actual_sharpe))
    return {
        "actual_sharpe": actual_sharpe,
        "boot_sharpe_mean": float(boot_sharpes.mean()),
        "boot_sharpe_std": float(boot_sharpes.std(ddof=1)),
        "p_value": p_value,
        "n_boot": n_boot,
        "block_size": block_size,
        "on_fraction": float(gate_bool.mean()),
        "boot_sharpes": boot_sharpes,
    }


# ---------------------------------------------------------------------------
# (b) Always-on comparison
# ---------------------------------------------------------------------------

def gated_vs_always_on(gated_returns: pd.Series, always_on_returns: pd.Series, periods_per_year: int = 52) -> dict:
    gated = summary_stats(gated_returns, periods_per_year)
    baseline = summary_stats(always_on_returns, periods_per_year)
    return {
        "gated": gated,
        "always_on": baseline,
        "beats_sharpe": gated["sharpe"] > baseline["sharpe"],
        "beats_drawdown": gated["max_drawdown"] > baseline["max_drawdown"],  # less negative = better
        "beats_both": (gated["sharpe"] > baseline["sharpe"]) and (gated["max_drawdown"] > baseline["max_drawdown"]),
    }


# ---------------------------------------------------------------------------
# (c) Twin-gate regression: does L survive controlling for rho_bar / AR?
# ---------------------------------------------------------------------------

def twin_gate_regression(
    forward_pnl: pd.Series,
    l_smoothed: pd.Series,
    rho_bar_smoothed: pd.Series,
    absorption_ratio_smoothed: pd.Series,
    hac_lags: int = 4,
) -> dict:
    """OLS of forward (always-on) reversal PnL on the smoothed H1 signal,
    controlling for smoothed rho_bar and absorption ratio, with
    Newey-West HAC standard errors (returns are weekly and the regressors
    are themselves trailing-smoothed, so residual autocorrelation is
    expected).

    Regressors are standardized (z-scored) first so the coefficient on L
    is directly comparable in magnitude to the control coefficients.
    """
    import statsmodels.api as sm

    df = pd.DataFrame(
        {
            "y": forward_pnl,
            "L": l_smoothed,
            "rho_bar": rho_bar_smoothed,
            "absorption_ratio": absorption_ratio_smoothed,
        }
    ).dropna()

    if len(df) < 30:
        raise ValueError(f"twin_gate_regression: too few overlapping observations ({len(df)})")

    for col in ["L", "rho_bar", "absorption_ratio"]:
        std = df[col].std()
        df[col] = (df[col] - df[col].mean()) / std if std > 0 else 0.0

    X = sm.add_constant(df[["L", "rho_bar", "absorption_ratio"]])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    return {
        "n_obs": int(len(df)),
        "b_L": float(model.params["L"]),
        "se_L": float(model.bse["L"]),
        "t_L": float(model.tvalues["L"]),
        "p_L": float(model.pvalues["L"]),
        "b_rho_bar": float(model.params["rho_bar"]),
        "p_rho_bar": float(model.pvalues["rho_bar"]),
        "b_absorption_ratio": float(model.params["absorption_ratio"]),
        "p_absorption_ratio": float(model.pvalues["absorption_ratio"]),
        "r_squared": float(model.rsquared),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Rejection-criteria helpers
# ---------------------------------------------------------------------------

def max_pnl_concentration(returns: pd.Series, window: int = 8) -> dict:
    """Largest share of total PnL contributed by any single rolling window
    of `window` consecutive weeks. Flags "PnL dominated by one episode".
    """
    r = returns.fillna(0.0)
    total = r.sum()
    if total == 0:
        return {"share": float("nan"), "window": window, "total_pnl": 0.0}
    rolling_sum = r.rolling(window).sum()
    max_window_pnl = rolling_sum.max()
    share = float(max_window_pnl / total) if total != 0 else float("nan")
    return {
        "share": share,
        "window": window,
        "total_pnl": float(total),
        "max_window_pnl": float(max_window_pnl),
        "max_window_end_date": rolling_sum.idxmax() if not rolling_sum.dropna().empty else None,
    }


def subperiod_sign_check(
    forward_pnl: pd.Series,
    l_smoothed: pd.Series,
    rho_bar_smoothed: pd.Series,
    absorption_ratio_smoothed: pd.Series,
    boundaries: Sequence[tuple],
    hac_lags: int = 4,
) -> list:
    """Re-run `twin_gate_regression` on each (start, end) subperiod and
    report the sign/significance of b_L in each -- rejection criterion:
    sign of b flips across 2000-07 / 2008-12 / 2013-17-style subperiods.
    """
    results = []
    for start, end in boundaries:
        # Slice each series independently by its own date range (the
        # series can have different underlying indices -- forward_pnl
        # spans every decision Friday, l_smoothed/etc. only the weeks
        # where the topology signal was computable -- so a boolean mask
        # built from one series' index cannot be applied to another).
        y = forward_pnl.loc[start:end]
        n_obs = int(y.shape[0])
        if n_obs < 30:
            results.append({"start": start, "end": end, "n_obs": n_obs, "insufficient_data": True})
            continue
        try:
            reg = twin_gate_regression(
                y,
                l_smoothed.loc[start:end],
                rho_bar_smoothed.loc[start:end],
                absorption_ratio_smoothed.loc[start:end],
                hac_lags=hac_lags,
            )
            reg.pop("model")
            reg.update({"start": start, "end": end, "insufficient_data": False})
            results.append(reg)
        except ValueError as exc:
            results.append({"start": start, "end": end, "n_obs": n_obs, "insufficient_data": True, "error": str(exc)})
    return results
