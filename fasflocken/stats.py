"""
Fasflocken (PH-1) -- null-hypothesis baseline & validation suite.

Implements the four checks from the spec's "Nollhypotes-baslinje" section
plus the Deflated Sharpe Ratio used to correct the declared grid, and the
sign-instability check used in the final rejection rule:

  1. circular_block_bootstrap_pvalue -- resample the Z_s(t) signal panel in
     13-week circular blocks, replay the SAME portfolio construction/cost
     model against the REAL (unshuffled) forward returns, get a null
     Sharpe distribution.
  2. run_twin_backtest / delta_sharpe_vs_twin -- boring-twin comparison
     (mean pairwise correlation in place of Kuramoto R).
  3. random_sector_baseline -- random-selection Sharpe distribution with
     matched gross/turnover, for a sanity floor under "pick any 3-vs-3".
  4. overlap_control -- generic Spearman/return-correlation hook against
     an external strategy's signal/returns (e.g. "Oglegrinden"; that
     strategy isn't implemented in this repository, so this is a pluggable
     comparison, not a canned result).

  oracle_backtest -- perfect-foresight sector-rotation ceiling (same L/S-N,
     dollar-neutral, vol-targeted, costed construction, but legs chosen by
     realized forward return instead of Z_s).
  deflated_sharpe_ratio -- Bailey & Lopez de Prado PSR/DSR, correcting for
     the size of the declared grid.
  sign_stability -- checks the strategy's average return keeps the same
     sign across the three declared sub-periods.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from arch.bootstrap import CircularBlockBootstrap

from fasflocken.config import (
    BOOTSTRAP_BLOCK_WEEKS,
    BOOTSTRAP_N_DRAWS,
    OVERLAP_CORR_THRESHOLD,
    SAMPLE_WINDOW,
    SignalParams,
    CostModel,
    DEFAULT_COSTS,
    VOL_TARGET_ANN,
    MAX_GROSS,
    WEEKS_PER_YEAR,
)
from fasflocken.backtest import (
    BacktestResult,
    run_backtest,
    annualized_sharpe,
    _week_windows,
)
from fasflocken.pipeline import compute_sector_signal, build_z_panel
from fasflocken.portfolio import (
    eligible_sectors,
    equal_dollar_direction,
    vol_target_scale,
    trailing_cov_matrix,
    compute_turnover,
)
from fasflocken.config import SECTOR_TO_ETF, GICS_SECTORS
from fasflocken.universe import UniverseProvider

_EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# 1. Circular block bootstrap of the signal.
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    observed_sharpe: float
    null_sharpes: np.ndarray
    p_value: float
    n_draws: int
    block_weeks: int


def _resample_panel(panel: pd.DataFrame, block_len: int, rng: np.random.Generator) -> pd.DataFrame:
    """One circular-block-bootstrap draw of `panel`'s rows (wraparound at the
    edges preserves each week's own autocorrelation while breaking the link
    between a given calendar date's signal and its forward return).

    Uses `arch.bootstrap.CircularBlockBootstrap` (the library named in the
    spec's toolchain) rather than a hand-rolled resampler. `rng` is reused
    (and advanced) across repeated calls so consecutive draws differ.
    """
    block_len = max(1, min(block_len, len(panel)))
    bs = CircularBlockBootstrap(block_len, panel, seed=rng)
    (resampled,), _kwdata = next(bs.bootstrap(1))
    resampled = resampled.copy()
    resampled.index = panel.index  # keep the real calendar; only the content is shuffled
    return resampled


def circular_block_bootstrap_pvalue(
    provider: UniverseProvider,
    start: _dt.date,
    end: _dt.date,
    params: SignalParams = SignalParams(),
    n_draws: int = BOOTSTRAP_N_DRAWS,
    block_weeks: int = BOOTSTRAP_BLOCK_WEEKS,
    seed: int | None = 0,
    target_vol_ann: float = VOL_TARGET_ANN,
    max_gross: float = MAX_GROSS,
    cost_model: CostModel = DEFAULT_COSTS,
    sectors: tuple[str, ...] = GICS_SECTORS,
    z_panel: pd.DataFrame | None = None,
) -> BootstrapResult:
    """H0: the observed net Sharpe is explainable by the signal's own
    autocorrelation alone, with no genuine timing relationship to forward
    returns. Reject H0 (i.e. the strategy has real timing information) when
    p_value is small.

    p_value = P(null Sharpe >= observed Sharpe), estimated as
    (#draws with null_sharpe >= observed + 1) / (n_draws + 1).
    """
    if z_panel is None:
        sector_signals = {s: compute_sector_signal(provider, s, start, end, params) for s in sectors}
        z_panel = build_z_panel(sector_signals)

    observed = run_backtest(provider, start, end, params, target_vol_ann, max_gross, cost_model=cost_model,
                             sectors=sectors, z_override=z_panel)
    observed_sharpe = annualized_sharpe(observed.weekly_returns)

    rng = np.random.default_rng(seed)
    null_sharpes = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        z_resampled = _resample_panel(z_panel, block_weeks, rng)
        bt = run_backtest(provider, start, end, params, target_vol_ann, max_gross, cost_model=cost_model,
                           sectors=sectors, z_override=z_resampled)
        null_sharpes[i] = annualized_sharpe(bt.weekly_returns)

    valid = null_sharpes[~np.isnan(null_sharpes)]
    if len(valid) == 0 or np.isnan(observed_sharpe):
        p_value = float("nan")
    else:
        p_value = float((np.sum(valid >= observed_sharpe) + 1) / (len(valid) + 1))

    return BootstrapResult(observed_sharpe, null_sharpes, p_value, n_draws, block_weeks)


# ---------------------------------------------------------------------------
# 2. Boring twin.
# ---------------------------------------------------------------------------

def run_twin_backtest(
    provider: UniverseProvider,
    start: _dt.date,
    end: _dt.date,
    params: SignalParams = SignalParams(),
    target_vol_ann: float = VOL_TARGET_ANN,
    max_gross: float = MAX_GROSS,
    cost_model: CostModel = DEFAULT_COSTS,
    sectors: tuple[str, ...] = GICS_SECTORS,
    sector_signals: dict | None = None,
) -> BacktestResult:
    """Identical construction, R_s replaced by mean pairwise correlation Corr_s."""
    if sector_signals is None:
        sector_signals = {s: compute_sector_signal(provider, s, start, end, params) for s in sectors}
    twin_z_panel = build_z_panel(sector_signals, twin=True)
    return run_backtest(provider, start, end, params, target_vol_ann, max_gross, cost_model=cost_model,
                         sectors=sectors, z_override=twin_z_panel)


def delta_sharpe_vs_twin(main_result: BacktestResult, twin_result: BacktestResult) -> float:
    """Net-Sharpe(phase) - net-Sharpe(amplitude-weighted twin). Must exceed
    0.15 (spec's rejection threshold) or the phase machinery is decoration.
    """
    return annualized_sharpe(main_result.weekly_returns) - annualized_sharpe(twin_result.weekly_returns)


# ---------------------------------------------------------------------------
# 3. Randomized sector selection baseline.
# ---------------------------------------------------------------------------

def random_sector_baseline(
    provider: UniverseProvider,
    start: _dt.date,
    end: _dt.date,
    n_legs: int = 3,
    n_draws: int = 200,
    seed: int | None = 0,
    target_vol_ann: float = VOL_TARGET_ANN,
    max_gross: float = MAX_GROSS,
    cost_model: CostModel = DEFAULT_COSTS,
    sectors: tuple[str, ...] = GICS_SECTORS,
) -> np.ndarray:
    """Sharpe distribution from picking `n_legs` random longs / `n_legs`
    random shorts each week (same gross target, no hysteresis -- a fresh
    random draw every week gives an upper-bound turnover comparison; this
    is deliberately the noisiest, most turnover-heavy baseline).
    """
    daily_dates = provider.trading_calendar(start, end)
    etfs = [SECTOR_TO_ETF[s] for s in sectors]
    etf_prices = provider.sector_etf_prices(start, end, etfs).reindex(index=daily_dates)
    etf_log_returns = np.log(etf_prices).diff()

    dummy = pd.Series(0.0, index=daily_dates)
    from fasflocken.signals import resample_weekly_last

    signal_dates = resample_weekly_last(dummy).index
    signal_dates = signal_dates[(signal_dates >= pd.Timestamp(start)) & (signal_dates <= pd.Timestamp(end))]
    windows = _week_windows(signal_dates, daily_dates)
    cost_rate = cost_model.total_bps_per_side() / 10_000.0

    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_draws, dtype=float)

    for d in range(n_draws):
        prev_weights = pd.Series(dtype=float)
        rets = []
        for f_i, window in zip(signal_dates, windows):
            elig = eligible_sectors(f_i.date(), sectors)
            if len(elig) < 2 * n_legs:
                rets.append(0.0)
                continue
            choice = rng.permutation(elig)
            longs, shorts = list(choice[:n_legs]), list(choice[n_legs : 2 * n_legs])
            base_weights = equal_dollar_direction(longs, shorts)

            cov_matrix = trailing_cov_matrix(etf_log_returns, f_i, 60)
            k = vol_target_scale(base_weights, cov_matrix, target_vol_ann, max_gross)
            final_weights = base_weights * k

            turnover = compute_turnover(prev_weights, final_weights)
            if len(window) == 0:
                gross_ret = 0.0
            else:
                r = etf_log_returns.loc[window, final_weights.index.intersection(etf_log_returns.columns)]
                w_aligned = final_weights.reindex(r.columns, fill_value=0.0)
                gross_ret = float(r.mul(w_aligned, axis=1).sum(axis=1).sum())
            rets.append(gross_ret - turnover * cost_rate)
            prev_weights = final_weights

        series = pd.Series(rets, index=signal_dates)
        sharpes[d] = annualized_sharpe(series)

    return sharpes


# ---------------------------------------------------------------------------
# Oracle cap.
# ---------------------------------------------------------------------------

def oracle_backtest(
    provider: UniverseProvider,
    start: _dt.date,
    end: _dt.date,
    n_legs: int = 3,
    target_vol_ann: float = VOL_TARGET_ANN,
    max_gross: float = MAX_GROSS,
    cost_model: CostModel = DEFAULT_COSTS,
    sectors: tuple[str, ...] = GICS_SECTORS,
) -> pd.Series:
    """Perfect-foresight ceiling: each week, go long the n_legs sectors with
    the HIGHEST realized forward return and short the n_legs with the
    LOWEST, using the same dollar-neutral / vol-targeted / costed
    construction as the real strategy but no hysteresis (foresight makes
    hysteresis moot). Returns the net weekly return series.

    If this ceiling's Sharpe is low, the trade structure itself can't
    support the hypothesis regardless of signal quality ("Tidspilen
    lesson" cited in the spec).
    """
    daily_dates = provider.trading_calendar(start, end)
    etfs = [SECTOR_TO_ETF[s] for s in sectors]
    etf_prices = provider.sector_etf_prices(start, end, etfs).reindex(index=daily_dates)
    etf_log_returns = np.log(etf_prices).diff()

    dummy = pd.Series(0.0, index=daily_dates)
    from fasflocken.signals import resample_weekly_last

    signal_dates = resample_weekly_last(dummy).index
    signal_dates = signal_dates[(signal_dates >= pd.Timestamp(start)) & (signal_dates <= pd.Timestamp(end))]
    windows = _week_windows(signal_dates, daily_dates)
    cost_rate = cost_model.total_bps_per_side() / 10_000.0

    prev_weights = pd.Series(dtype=float)
    rets = []
    for f_i, window in zip(signal_dates, windows):
        elig = eligible_sectors(f_i.date(), sectors)
        if len(elig) < 2 * n_legs or len(window) == 0:
            rets.append(0.0)
            continue

        etf_of = {s: SECTOR_TO_ETF[s] for s in elig}
        fwd_ret = pd.Series({s: etf_log_returns.loc[window, etf_of[s]].sum() for s in elig}).sort_values()
        shorts = list(fwd_ret.index[:n_legs])   # worst forward performers
        longs = list(fwd_ret.index[-n_legs:])   # best forward performers
        base_weights = equal_dollar_direction(longs, shorts)

        cov_matrix = trailing_cov_matrix(etf_log_returns, f_i, 60)
        k = vol_target_scale(base_weights, cov_matrix, target_vol_ann, max_gross)
        final_weights = base_weights * k

        turnover = compute_turnover(prev_weights, final_weights)
        r = etf_log_returns.loc[window, final_weights.index.intersection(etf_log_returns.columns)]
        w_aligned = final_weights.reindex(r.columns, fill_value=0.0)
        gross_ret = float(r.mul(w_aligned, axis=1).sum(axis=1).sum())
        rets.append(gross_ret - turnover * cost_rate)
        prev_weights = final_weights

    return pd.Series(rets, index=signal_dates, name="oracle_net_return")


# ---------------------------------------------------------------------------
# 4. Overlap control (generic hook -- external strategy not in this repo).
# ---------------------------------------------------------------------------

def overlap_control(
    signal_a: pd.Series,
    signal_b: pd.Series,
    returns_a: pd.Series,
    returns_b: pd.Series,
    threshold: float = OVERLAP_CORR_THRESHOLD,
) -> dict:
    """Spearman-rho between two market-level signals, and Pearson corr
    between two books' returns. Intended usage: signal_a = market average
    of R_s across the 11 sectors, signal_b = the comparison strategy's
    own signal (e.g. "Oglegrinden"'s L_t); returns_a/b = each book's net
    weekly returns. That strategy is not implemented in this repository/
    session, so signal_b/returns_b must be supplied externally -- this
    function only evaluates the |rho| < threshold pass/fail rule.
    """
    aligned_sig = pd.concat([signal_a.rename("a"), signal_b.rename("b")], axis=1).dropna()
    if len(aligned_sig) >= 3:
        rho, p = spearmanr(aligned_sig["a"], aligned_sig["b"])
    else:
        rho, p = float("nan"), float("nan")

    aligned_ret = pd.concat([returns_a.rename("a"), returns_b.rename("b")], axis=1).dropna()
    ret_corr = float(aligned_ret["a"].corr(aligned_ret["b"])) if len(aligned_ret) >= 3 else float("nan")

    passes = bool(
        (not np.isnan(rho) and abs(rho) < threshold)
        and (not np.isnan(ret_corr) and abs(ret_corr) < threshold)
    )
    return {"spearman_rho": rho, "spearman_p": p, "return_corr": ret_corr, "threshold": threshold, "passes": passes}


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio.
# ---------------------------------------------------------------------------

def expected_max_sharpe(sharpe_std_across_trials: float, n_trials: int) -> float:
    """E[max Sharpe] under N independent trials with Sharpe std `sharpe_std_across_trials`
    (Bailey & Lopez de Prado 2014, eq. for the expected maximum of N Gaussians)."""
    if n_trials <= 1 or sharpe_std_across_trials <= 0:
        return 0.0
    return float(
        sharpe_std_across_trials
        * (
            (1 - _EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials)
            + _EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(
    weekly_returns: pd.Series, trial_sharpes: np.ndarray, periods_per_year: int = WEEKS_PER_YEAR
) -> dict:
    """Bailey & Lopez de Prado (2014) PSR/DSR, using this trial's own return
    moments and the empirical spread of Sharpe ratios across the declared
    grid (`trial_sharpes`) to estimate the expected maximum Sharpe you'd
    see from that many trials under a true-zero-skill null.

    The PSR/DSR formula (the skew/kurtosis correction terms and the
    sqrt(n_obs - 1) scaling in particular) is derived for the *periodic*
    Sharpe ratio -- i.e. the same frequency as the return observations
    feeding it, weekly here -- not an annualized one. `trial_sharpes` is,
    by convention everywhere else in this codebase (grid_search.py,
    run.py), a set of *annualized* Sharpes, so it's converted back to the
    weekly scale (divide by sqrt(periods_per_year)) before use; this is
    exactly the inverse of how those trial Sharpes were annualized in the
    first place, so it's lossless. All internal math (sr, sr0, psr) runs
    on the periodic scale; `sharpe`, `sharpe0_expected_max` and
    `deflated_sharpe_gap` are then reported back in annualized units
    (a fixed sqrt(periods_per_year) rescaling, consistent with
    backtest.annualized_sharpe) so they're directly comparable to every
    other Sharpe figure this package prints.

    Returns both:
      * `psr` -- the standard [0,1] probabilistic DSR (P(true SR > 0) after
        deflation); and
      * `deflated_sharpe_gap` -- observed_sharpe - expected_max_sharpe (both
        annualized), a Sharpe-ratio-scaled quantity that is negative when
        the observed Sharpe fails to clear what pure multiple-testing luck
        would produce. The spec's rejection rule "DSR <= 0" is evaluated
        on this gap (see grid_search.evaluate_rejection), since a
        probability can never be <= 0 in a meaningful way.
    """
    r = weekly_returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3 or np.std(r, ddof=1) == 0:
        return {
            "psr": float("nan"), "deflated_sharpe_gap": float("nan"), "sharpe": float("nan"),
            "sharpe0_expected_max": float("nan"), "skew": float("nan"), "kurtosis": float("nan"),
            "n_obs": n, "n_trials": len(trial_sharpes),
        }

    ann_factor = np.sqrt(periods_per_year)
    sr = float(r.mean() / r.std(ddof=1))  # periodic (weekly) Sharpe -- what the PSR formula expects
    skew = float(pd.Series(r).skew())
    kurtosis = float(pd.Series(r).kurtosis()) + 3.0  # pandas reports excess kurtosis; formula wants raw kurtosis

    valid_trials = trial_sharpes[~np.isnan(trial_sharpes)] if len(trial_sharpes) else np.array([])
    valid_trials_periodic = valid_trials / ann_factor  # de-annualize to match `sr`'s scale
    sharpe_std = float(np.std(valid_trials_periodic, ddof=1)) if len(valid_trials_periodic) > 1 else 0.0
    n_trials = max(len(valid_trials_periodic), 1)
    sr0 = expected_max_sharpe(sharpe_std, n_trials)  # periodic scale

    denom = np.sqrt(max(1e-12, 1 - skew * sr + ((kurtosis - 1) / 4) * sr**2))
    psr = float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))

    return {
        "psr": psr,
        "deflated_sharpe_gap": float((sr - sr0) * ann_factor),
        "sharpe": sr * ann_factor,
        "sharpe0_expected_max": sr0 * ann_factor,
        "skew": skew,
        "kurtosis": kurtosis,
        "n_obs": n,
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Sign instability.
# ---------------------------------------------------------------------------

def sign_stability(weekly_returns: pd.Series, periods=None) -> dict:
    periods = periods or SAMPLE_WINDOW.sign_check_periods
    period_means = {}
    for s, e in periods:
        seg = weekly_returns.loc[str(s) : str(e)]
        period_means[f"{s}:{e}"] = float(seg.mean()) if len(seg) > 0 else float("nan")

    vals = [v for v in period_means.values() if not np.isnan(v)]
    stable = len(vals) > 0 and (all(v > 0 for v in vals) or all(v < 0 for v in vals))
    return {"period_means": period_means, "sign_stable": stable}
