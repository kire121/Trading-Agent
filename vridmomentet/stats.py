"""Performance statistics and the falsification suite.

Generic performance/DSR/bootstrap formulas are the same textbook
(Bailey & Lopez de Prado 2014; Politis & Romano 1994) constructions
Oglegrinden's stats.py already implements and unit-tests in this
repository; reproduced here (not imported cross-package, since
Oglegrinden lives on a separate, unmerged branch) with the same
per-period-Sharpe convention and the same documented gotcha: every DSR
call must be fed *per-period* (weekly), not annualized, Sharpe ratios.

Two things in this file are specific to Vridmomentet and have no
Oglegrinden precedent:

1. `shuffle_null_check` -- the brief's own pre-registered "Huvudnull":
   shuffle day-order *within each window* and recompute the Levy area.
   See signal.py's module docstring for the exact (and exactly-provable)
   sense in which E[A]=0 under this null, and why that's an approximation
   (not an identity) for the actual z-scored estimator.
2. `pead_exclusion_mask` -- a declared proxy control for "PEAD in
   disguise" (earnings-day volume dominating the area). No point-in-time
   earnings-calendar data source was available to build a literal
   earnings-date control, so this flags and excludes windows containing an
   abnormal single-day volume spike (relative to *that name's own* trailing
   volume distribution) as an earnings-announcement proxy, and the pipeline
   reports whether the strategy's edge survives with those weeks excluded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sstats

from vridmomentet.config import RejectionThresholds, WEEKS_PER_YEAR
from vridmomentet.data import Panel

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# Basic performance statistics
# --------------------------------------------------------------------------

_ZERO_STD_TOL = 1e-12  # a "constant" series can still have ~1e-18-scale float noise, not exact 0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR, annualize: bool = True) -> float:
    r = returns.dropna()
    std = r.std(ddof=1) if len(r) >= 2 else 0.0
    if len(r) < 2 or std < _ZERO_STD_TOL:
        return 0.0
    sr = r.mean() / std
    return float(sr * np.sqrt(periods_per_year)) if annualize else float(sr)


def sortino_ratio(returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR, target: float = 0.0) -> float:
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
    return float(drawdown.min()) if len(drawdown) else 0.0


def annualized_return(returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    wealth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or wealth <= 0:
        return float("nan")
    return float(wealth ** (1.0 / years) - 1.0)


def summary_stats(returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> dict:
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


# --------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# --------------------------------------------------------------------------

def expected_max_sharpe_under_null(trial_sharpes: list[float]) -> dict:
    """SR0: expected max Sharpe across len(trial_sharpes) independent,
    zero-skill trials, using the empirical variance of the observed
    (per-period) trial Sharpes as the estimate of V[SR_n].
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
            * ((1 - EULER_GAMMA) * sstats.norm.ppf(1 - 1.0 / n) + EULER_GAMMA * sstats.norm.ppf(1 - 1.0 / (n * np.e)))
        )
    return {"sr0": sr0, "var_sr": var_sr, "n_trials": n}


def deflated_sharpe_ratio(
    observed_sharpe_per_period: float,
    trial_sharpes_per_period: list[float],
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """DSR = P(true SR > 0 | observed SR is the max of N trials). All Sharpe
    ratios here are PER-PERIOD (weekly), not annualized -- mixing scales
    silently biases the z-statistic.
    """
    null = expected_max_sharpe_under_null(trial_sharpes_per_period)
    sr0 = null["sr0"]
    denom = np.sqrt(max(1e-12, 1 - skewness * observed_sharpe_per_period + ((kurtosis - 1) / 4.0) * observed_sharpe_per_period ** 2))
    z = (observed_sharpe_per_period - sr0) * np.sqrt(max(n_obs - 1, 1)) / denom
    dsr = float(sstats.norm.cdf(z))
    return {"dsr": dsr, "z": float(z), "sr0": sr0, "n_obs": n_obs, "n_trials": null["n_trials"], "var_sr": null["var_sr"]}


# --------------------------------------------------------------------------
# Stationary block bootstrap significance of the Sharpe ratio
# (Politis & Romano 1994)
# --------------------------------------------------------------------------

def _stationary_bootstrap_indices(n: int, block_size: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / block_size
    idx = np.empty(n, dtype=int)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        length = min(length, n - pos)
        idx[pos : pos + length] = (start + np.arange(length)) % n
        pos += length
    return idx


def block_bootstrap_sharpe_pvalue(
    returns: pd.Series, n_boot: int = 1000, block_size: float = 8.0,
    periods_per_year: int = WEEKS_PER_YEAR, seed: int = 0,
) -> dict:
    """Demean the weekly return series (a true zero-mean null generator),
    stationary-block-bootstrap resample it n_boot times preserving the
    series' own autocorrelation/heteroskedasticity structure, and ask: how
    often does a resampled *zero-mean* path produce a Sharpe at least as
    large as the one actually observed (raw, non-demeaned)?
    """
    r = returns.dropna()
    observed_sharpe = sharpe_ratio(r, periods_per_year)
    demeaned = (r - r.mean()).values
    n = len(demeaned)
    if n < 10:
        return {"observed_sharpe": observed_sharpe, "p_value": float("nan"), "n_boot": n_boot,
                "block_size": block_size, "boot_sharpe_mean": float("nan"), "boot_sharpe_std": float("nan")}

    rng = np.random.default_rng(seed)
    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, block_size, rng)
        boot_sharpes[b] = sharpe_ratio(pd.Series(demeaned[idx]), periods_per_year)

    p_value = float(np.mean(boot_sharpes >= observed_sharpe))
    return {
        "observed_sharpe": observed_sharpe, "p_value": p_value, "n_boot": n_boot, "block_size": block_size,
        "boot_sharpe_mean": float(boot_sharpes.mean()), "boot_sharpe_std": float(boot_sharpes.std(ddof=1)),
    }


# --------------------------------------------------------------------------
# Primary pre-registered null: shuffle day-order within each window
# --------------------------------------------------------------------------

def shuffle_null_check(
    panel: Panel, window: int, n_windows_sample: int = 300, n_reps: int = 1000,
    block_sizes: tuple[int, ...] = (1, 4), seed: int = 0,
) -> dict:
    """Brief: "shuffla dagordningen inom varje fonster. E[A]=0 exakt under
    utbytbarhet ... 1000 rep, aven blockvis." For a random sample of actual
    (ticker, date) windows, shuffle day-order (both a full iid permutation,
    block_size=1, and block-preserving shuffles for larger block sizes),
    recompute the Levy area on each shuffled draw, and ask: for what
    fraction of sampled windows does the REAL |A| exceed the 95th
    percentile of its own window's shuffle-null distribution? Under the
    null (no genuine order information), this fraction should be close to
    the nominal 5%; a much higher fraction is evidence the estimator is
    picking up real order information the brief's null is designed to kill.
    """
    from vridmomentet.signal import levy_area_of_windows, signed_dollar_volume

    u = signed_dollar_volume(panel)
    r_vals = panel.log_returns.values
    u_vals = u.values
    n_dates, n_names = r_vals.shape

    rng = np.random.default_rng(seed)
    valid_positions = []
    for _ in range(n_windows_sample * 20):  # oversample, since most draws will be invalid (NaN window)
        if len(valid_positions) >= n_windows_sample:
            break
        t = rng.integers(window - 1, n_dates)
        j = rng.integers(0, n_names)
        r_win = r_vals[t - window + 1 : t + 1, j]
        u_win = u_vals[t - window + 1 : t + 1, j]
        if np.any(np.isnan(r_win)) or np.any(np.isnan(u_win)):
            continue
        valid_positions.append((r_win, u_win))

    results = {}
    for block_size in block_sizes:
        exceed_count = 0
        n_tested = 0
        for r_win, u_win in valid_positions:
            real_a = float(levy_area_of_windows(r_win[None, :], u_win[None, :])[0])
            if np.isnan(real_a):
                continue
            null_a = np.empty(n_reps)
            for rep in range(n_reps):
                perm = _block_shuffle_indices(window, block_size, rng)
                null_a[rep] = levy_area_of_windows(r_win[perm][None, :], u_win[perm][None, :])[0]
            null_a = null_a[~np.isnan(null_a)]
            if len(null_a) < 10:
                continue
            threshold = np.percentile(np.abs(null_a), 95)
            exceed_count += int(abs(real_a) > threshold)
            n_tested += 1
        results[block_size] = {
            "n_windows_tested": n_tested,
            "fraction_exceeding_95th_pct_null": exceed_count / n_tested if n_tested else float("nan"),
        }
    return {"n_reps": n_reps, "block_sizes": list(block_sizes), "by_block_size": results}


def _block_shuffle_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 1:
        return rng.permutation(n)
    n_blocks = int(np.ceil(n / block_size))
    blocks = [np.arange(i * block_size, min((i + 1) * block_size, n)) for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    return np.concatenate([blocks[i] for i in order])


def ic_stage1_check(s_panel: pd.DataFrame, forward_return_panel: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> dict:
    """The brief's own headline falsifiable prediction: "da ar IC ~ 0 och
    ideen dor billigt i steg 1." Cross-sectional Spearman IC between the
    signal and the forward 5-day return, at each decision date, aggregated
    with a simple mean/t-stat across weeks (each week's IC is treated as
    one observation; no attempt to correct for cross-week autocorrelation
    here -- block_bootstrap_sharpe_pvalue-style resampling of the *strategy
    return* series, done separately in stage 2/3, is the more careful test
    of statistical significance; this is the cheap, fast, stage-1 screen).
    """
    ics = {}
    for d in decision_dates:
        if d not in s_panel.index or d not in forward_return_panel.index:
            continue
        s_row = s_panel.loc[d].dropna()
        fwd_row = forward_return_panel.loc[d].reindex(s_row.index).dropna()
        common = s_row.index.intersection(fwd_row.index)
        if len(common) < 20:
            continue
        rho, _ = sstats.spearmanr(s_row[common], fwd_row[common])
        if not np.isnan(rho):
            ics[d] = rho

    ic_series = pd.Series(ics)
    if len(ic_series) < 2:
        return {"mean_ic": float("nan"), "t_stat": float("nan"), "n_weeks": len(ic_series), "ic_series": ic_series}
    mean_ic = float(ic_series.mean())
    se_ic = float(ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
    t_stat = mean_ic / se_ic if se_ic > 0 else float("nan")
    return {"mean_ic": mean_ic, "t_stat": t_stat, "n_weeks": len(ic_series), "ic_series": ic_series}


# --------------------------------------------------------------------------
# Twin comparison
# --------------------------------------------------------------------------

def beats_all_twins(primary_returns: pd.Series, twin_returns: dict[str, pd.Series], periods_per_year: int = WEEKS_PER_YEAR) -> dict:
    primary_sharpe = sharpe_ratio(primary_returns, periods_per_year)
    per_twin = {}
    for name, returns in twin_returns.items():
        twin_sharpe = sharpe_ratio(returns, periods_per_year)
        per_twin[name] = {"twin_sharpe": twin_sharpe, "primary_beats_twin": primary_sharpe > twin_sharpe}
    return {
        "primary_sharpe": primary_sharpe, "per_twin": per_twin,
        "beats_all_twins": all(v["primary_beats_twin"] for v in per_twin.values()) if per_twin else None,
    }


# --------------------------------------------------------------------------
# PEAD-in-disguise proxy control
# --------------------------------------------------------------------------

def pead_exclusion_mask(panel: Panel, window: int, spike_lookback: int = 120, spike_z_threshold: float = 4.0) -> pd.DataFrame:
    """True where a name's trailing `window`-day formation window contains a
    single-day dollar-volume spike (z-score, vs. that name's own trailing
    `spike_lookback`-day distribution, > spike_z_threshold) -- a proxy for
    "an earnings announcement (or comparable news event) happened inside
    this window", since no point-in-time earnings-calendar data source was
    available (see module docstring).
    """
    # Compare each day's volume to the *prior* trailing distribution
    # (shifted, excluding the day itself) -- an outlier included in its own
    # rolling mean/std dilutes its own z-score, which would silently make
    # spike detection weaker exactly on the days it matters most.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        roll_mean = panel.dollar_volume.rolling(spike_lookback, min_periods=60).mean().shift(1)
        roll_std = panel.dollar_volume.rolling(spike_lookback, min_periods=60).std(ddof=1).shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_z = (panel.dollar_volume - roll_mean) / roll_std
    daily_spike = vol_z > spike_z_threshold
    return daily_spike.rolling(window, min_periods=1).max().astype(bool)


def apply_exclusion_mask(s_panel: pd.DataFrame, exclusion_mask: pd.DataFrame) -> pd.DataFrame:
    mask = exclusion_mask.reindex_like(s_panel).fillna(False)
    return s_panel.where(~mask)


# --------------------------------------------------------------------------
# PnL concentration
# --------------------------------------------------------------------------

def max_pnl_concentration(returns: pd.Series, window: int = 8) -> dict:
    r = returns.fillna(0.0)
    total = r.sum()
    if total == 0 or len(r) < window:
        return {"share": float("nan"), "window": window, "total_pnl": float(total)}
    rolling_sum = r.rolling(window).sum()
    max_window_pnl = rolling_sum.max()
    share = float(max_window_pnl / total) if total != 0 else float("nan")
    return {
        "share": share, "window": window, "total_pnl": float(total), "max_window_pnl": float(max_window_pnl),
        "max_window_end_date": rolling_sum.idxmax() if not rolling_sum.dropna().empty else None,
    }


# --------------------------------------------------------------------------
# Sub-period sign stability
# --------------------------------------------------------------------------

def sign_stability(returns: pd.Series, subperiods: dict[str, tuple]) -> dict:
    signs = {}
    for name, (start, end) in subperiods.items():
        window = returns.loc[str(start):str(end)]
        if len(window.dropna()) < 10:
            signs[name] = {"n_obs": int(len(window.dropna())), "mean_return": float("nan"), "insufficient_data": True}
            continue
        signs[name] = {"n_obs": int(len(window.dropna())), "mean_return": float(window.mean()), "insufficient_data": False}
    valid_signs = [np.sign(v["mean_return"]) for v in signs.values() if not v["insufficient_data"]]
    stable = bool(len(valid_signs) >= 2 and all(s == valid_signs[0] for s in valid_signs))
    return {"by_subperiod": signs, "sign_stable": stable, "n_subperiods_with_data": len(valid_signs)}


# --------------------------------------------------------------------------
# Diversification
# --------------------------------------------------------------------------

def diversification_stats(strategy_returns: pd.Series, benchmark_returns: pd.Series, trend_proxy_returns: pd.Series | None) -> dict:
    common = strategy_returns.index.intersection(benchmark_returns.index)
    x = benchmark_returns.reindex(common).fillna(0.0).values
    y = strategy_returns.reindex(common).fillna(0.0).values
    if len(common) > 5 and x.std() > 0:
        beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
        corr_beta = float(np.corrcoef(x, y)[0, 1])
    else:
        beta, corr_beta = float("nan"), float("nan")

    trend_corr = float("nan")
    if trend_proxy_returns is not None:
        common2 = strategy_returns.index.intersection(trend_proxy_returns.index)
        if len(common2) > 5:
            trend_corr = float(np.corrcoef(
                strategy_returns.reindex(common2).fillna(0.0), trend_proxy_returns.reindex(common2).fillna(0.0)
            )[0, 1])

    return {
        "beta_to_benchmark": beta,
        "corr_to_benchmark": corr_beta,
        "corr_to_trend_proxy": trend_corr,
        "corr_to_tidspilen": None,  # N/A: no "Tidspilen" strategy exists anywhere in this codebase (verified
                                     # across all sibling branches) -- reported as N/A, not fabricated, per house
                                     # convention (see Oglegrinden's identical N/A disclosure for the same reason).
    }


# --------------------------------------------------------------------------
# Rejection verdict
# --------------------------------------------------------------------------

def evaluate_rejection(
    dsr_oos: dict, bootstrap: dict, shuffle: dict, twins: dict, concentration: dict, sign: dict,
    pead_delta_sharpe: float, thresholds: RejectionThresholds = RejectionThresholds(),
) -> dict:
    # The brief's own primary pre-registered null (block_size=1: a full,
    # unrestricted within-window day-order permutation). Under the null, a
    # 95th-percentile threshold test should flag a real window ~5% of the
    # time by construction; the observed fraction needs to sit meaningfully
    # above that (i.e. above shuffle_p_max) to count as evidence of genuine
    # order information, not just the test's own nominal false-positive rate.
    shuffle_fraction = shuffle.get("by_block_size", {}).get(1, {}).get("fraction_exceeding_95th_pct_null", float("nan"))

    reasons = {
        "dsr_oos_z_leq_threshold": bool(np.isnan(dsr_oos.get("z", np.nan)) or dsr_oos["z"] <= thresholds.dsr_z_min),
        "bootstrap_p_geq_threshold": bool(np.isnan(bootstrap.get("p_value", np.nan)) or bootstrap["p_value"] >= thresholds.bootstrap_p_max),
        "shuffle_null_shows_no_excess_order_information": bool(np.isnan(shuffle_fraction) or shuffle_fraction <= thresholds.shuffle_p_max),
        "twin_beats_primary": bool(twins.get("beats_all_twins") is not True),
        "pnl_concentration_gt_threshold": bool((concentration.get("share") or 0) > thresholds.pnl_concentration_max),
        "subperiod_sign_unstable": bool(not sign.get("sign_stable", False)),
        "pead_delta_sharpe_below_threshold": bool(np.isnan(pead_delta_sharpe) or pead_delta_sharpe < thresholds.pead_delta_sharpe_min),
    }
    return {"reject": bool(any(reasons.values())), "reasons": reasons}
