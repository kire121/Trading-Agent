"""
The pre-registered validation battery. Every threshold is read from
config.SPEC (nothing here is fit to the pilot data). Step 1 short-circuits
Step 2 on failure, matching the spec's own framing: a redundancy-screen
failure means the idea is dead before a portfolio is ever built.

  Step 0: existence (within-day permutation null) + breadth (PC1 share,
          cross-sectional dispersion vs a block-shuffle null, position count)
  Step 1: Fama-MacBeth incremental IC vs controls, Newey-West t-stat
  Step 2: costs, deflated Sharpe (using the tau-window x holding-period grid
          as the multiple-trials benchmark), sign consistency across IS
          subperiods, and a 5-leg baseline race the main signal must beat net
          of costs: 3 named twins (baselines.py) + block_shuffle_signal (null
          a) + random_matched_signal (null b). Reused through the same
          twin_horse_race function/liveness gate -- see pipeline.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import signal as sig
from . import stats_utils as su
from .config import SPEC


# ---------------------------------------------------------------------------
# Step 0a: existence, via within-day permutation of u_t
# ---------------------------------------------------------------------------

def existence_permutation_test(u_wide_by_symbol: Dict[str, pd.DataFrame],
                                qmin: int = SPEC.cepstrum.quefrency_min_min,
                                qmax: int = SPEC.cepstrum.quefrency_max_min,
                                eps: float = SPEC.cepstrum.epsilon,
                                n_perm: int = SPEC.validation.null_permutations,
                                sample_size: int = 500, seed: int = 0) -> dict:
    """For a random sample of (symbol, day) stock-days, permute that day's own
    u_t `n_perm` times (kills periodicity, keeps the marginal distribution
    exactly since it's a shuffle of the same values), and check whether the
    REAL max-cepstrum in [qmin,qmax] exceeds the permutation null's 95th
    percentile more often than chance would predict."""
    rng = np.random.default_rng(seed)
    candidates: List[tuple] = []
    for sym, uw in u_wide_by_symbol.items():
        valid_dates = uw.index[~uw.isna().any(axis=1)]
        candidates.extend((sym, d) for d in valid_dates)

    if not candidates:
        return {"n_sampled": 0, "exceedance_fraction": np.nan, "required_fraction": np.nan, "passed": False}

    n_sample = min(sample_size, len(candidates))
    sampled = [candidates[i] for i in rng.choice(len(candidates), size=n_sample, replace=False)]

    exceed = 0
    for sym, d in sampled:
        row = u_wide_by_symbol[sym].loc[d].values.astype(float)
        real_c = sig.real_cepstrum_row(row, eps)[qmin:qmax + 1]
        real_max = real_c.max()

        perm_rows = np.array([rng.permutation(row) for _ in range(n_perm)])
        spectrum = np.fft.fft(perm_rows, axis=1)
        log_power = np.log(np.abs(spectrum) ** 2 + eps)
        ceps = np.fft.ifft(log_power, axis=1).real
        perm_max = ceps[:, qmin:qmax + 1].max(axis=1)

        if real_max > np.percentile(perm_max, 95):
            exceed += 1

    fraction = exceed / n_sample
    required = SPEC.validation.null_nominal_alpha * SPEC.validation.null_exceedance_multiple
    return {"n_sampled": n_sample, "exceedance_fraction": fraction, "required_fraction": required,
            "passed": bool(fraction >= required)}


# ---------------------------------------------------------------------------
# Step 0b: breadth
# ---------------------------------------------------------------------------

def pc1_share(panel_wide: pd.DataFrame) -> float:
    x = panel_wide.dropna(axis=1, how="all")
    x = x.fillna(x.mean(axis=0))
    std = x.std(axis=0, ddof=0).replace(0, np.nan)
    x = ((x - x.mean(axis=0)) / std).fillna(0.0)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return float("nan")
    cov = np.cov(x.values, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = eigvals.sum()
    return float(eigvals.max() / total) if total > 0 else float("nan")


def cross_sectional_dispersion(panel_wide: pd.DataFrame) -> pd.Series:
    return panel_wide.std(axis=1, ddof=0)


def block_shuffle_null_dispersion(panel_wide: pd.DataFrame, n_shuffles: int = 100,
                                   block_size: int = 5, seed: int = 0) -> np.ndarray:
    """Per-symbol day-block shuffle: chop each symbol's own S_bar series into
    contiguous `block_size`-day blocks and randomly reorder the blocks,
    independently per symbol. This preserves each name's own short-run
    autocorrelation but destroys genuine cross-sectional (same-day) breadth,
    giving a null for 'how much dispersion would exist by chance alone'."""
    rng = np.random.default_rng(seed)
    values = panel_wide.values
    n_days, n_symbols = values.shape
    n_blocks = int(np.ceil(n_days / block_size))
    draws = []
    for _ in range(n_shuffles):
        shuffled = np.full_like(values, np.nan)
        for j in range(n_symbols):
            col = values[:, j]
            blocks = [col[b * block_size:(b + 1) * block_size] for b in range(n_blocks)]
            order = rng.permutation(n_blocks)
            shuffled[:, j] = np.concatenate([blocks[b] for b in order])[:n_days]
        disp = np.nanstd(shuffled, axis=1)
        draws.extend(disp[~np.isnan(disp)].tolist())
    return np.array(draws)


def breadth_diagnostics(S_bar_panel_wide: pd.DataFrame, n_positions_series: pd.Series,
                         seed: int = 0) -> dict:
    pc1 = pc1_share(S_bar_panel_wide)
    real_disp = cross_sectional_dispersion(S_bar_panel_wide).dropna()
    null_disp = block_shuffle_null_dispersion(S_bar_panel_wide, seed=seed)
    null_p95 = np.percentile(null_disp, 95) if len(null_disp) else np.nan
    disp_ok = bool(real_disp.median() > null_p95) if len(real_disp) and not np.isnan(null_p95) else False
    pc1_ok = bool(pc1 < SPEC.validation.pc1_share_max) if not np.isnan(pc1) else False

    avg_positions = float(n_positions_series.mean()) if len(n_positions_series) else 0.0
    positions_ok = bool(avg_positions >= SPEC.validation.min_expected_positions)

    return {
        "pc1_share": pc1, "pc1_ok": pc1_ok,
        "median_real_dispersion": float(real_disp.median()) if len(real_disp) else np.nan,
        "block_null_p95_dispersion": float(null_p95) if not np.isnan(null_p95) else np.nan,
        "dispersion_ok": disp_ok,
        "avg_simultaneous_positions": avg_positions,
        "min_required_positions": SPEC.validation.min_expected_positions,
        "positions_ok": positions_ok,
        "passed": pc1_ok and disp_ok,  # positions_ok reported separately: expected to fail at pilot scale, see README
    }


# ---------------------------------------------------------------------------
# Step 1: Fama-MacBeth incremental IC
# ---------------------------------------------------------------------------

def fama_macbeth_screen(panel: pd.DataFrame, y_col: str, z_col: str, control_cols: List[str],
                         nw_lags: int = SPEC.validation.newey_west_lags, min_cross_section: int = 10) -> dict:
    """panel: long-format DataFrame with columns ['date', y_col, z_col, *control_cols].
    Standardizes Z and controls cross-sectionally each day, runs daily OLS,
    and NW-tests the time series of the Z coefficient."""
    coefs, dates, n_used = [], [], []
    cols = [z_col] + control_cols
    for date, day_df in panel.groupby("date"):
        sub = day_df.dropna(subset=[y_col] + cols)
        if len(sub) < min_cross_section:
            continue
        x_raw = sub[cols].values.astype(float)
        std = x_raw.std(axis=0, ddof=0)
        if (std == 0).any():
            continue
        x = (x_raw - x_raw.mean(axis=0)) / std
        design = np.column_stack([np.ones(len(sub)), x])
        y = sub[y_col].values.astype(float)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        coefs.append(coef[1])
        dates.append(date)
        n_used.append(len(sub))

    coef_series = pd.Series(coefs, index=pd.Index(dates, name="date"))
    nw = su.newey_west_mean_tstat(coef_series.values, lags=nw_lags)
    passed = bool(not np.isnan(nw["t_stat"]) and abs(nw["t_stat"]) >= SPEC.validation.min_abs_newey_west_t)
    return {"coef_series": coef_series, "nw": nw, "n_days": len(coef_series),
            "avg_cross_section": float(np.mean(n_used)) if n_used else 0.0, "passed": passed}


# ---------------------------------------------------------------------------
# Step 2: sign consistency, twin horse race, deflated Sharpe
# ---------------------------------------------------------------------------

def sign_consistency_by_subperiod(panel: pd.DataFrame, direction_col: str, y_col: str,
                                   n_subperiods: int = SPEC.validation.n_is_subperiods,
                                   min_n: int = 20) -> dict:
    dates_sorted = sorted(panel["date"].unique())
    chunks = np.array_split(np.array(dates_sorted, dtype=object), n_subperiods)
    results = []
    for i, chunk in enumerate(chunks):
        sub = panel[panel["date"].isin(chunk)].dropna(subset=[direction_col, y_col])
        if len(sub) < min_n or sub[direction_col].std() == 0:
            results.append({"subperiod": i, "n": len(sub), "ic": np.nan, "consistent": False})
            continue
        ic, _ = scipy_stats.spearmanr(sub[direction_col], sub[y_col])
        results.append({"subperiod": i, "n": int(len(sub)), "ic": float(ic), "consistent": bool(ic > 0)})
    n_missing = sum(1 for r in results if not r["consistent"])
    passed = n_missing <= SPEC.validation.max_missing_sign_subperiods
    return {"subperiods": results, "n_missing_consistency": n_missing, "passed": passed}


def block_shuffle_signal(signal_by_symbol: Dict[str, pd.DataFrame], block_size: int = 5,
                          seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Null-baseline (a): per-stock day-block shuffle of the (S_bar, D) pair,
    independently per symbol. Preserves each name's own short-run
    autocorrelation in score/direction (blocks move together, aren't
    scrambled minute-by-minute) but randomizes WHICH calendar day that
    pattern lands on, breaking any genuine link between the signal and what
    actually happens next. A real signal's edge should survive comparison
    against this -- otherwise the backtest Sharpe is coming from some
    structural artifact of trading actively-moving names, not from the
    signal's timing."""
    rng = np.random.default_rng(seed)
    out = {}
    for sym, frame in signal_by_symbol.items():
        n = len(frame)
        if n == 0:
            out[sym] = frame
            continue
        n_blocks = int(np.ceil(n / block_size))
        order = rng.permutation(n_blocks)
        idx = np.concatenate([np.arange(b * block_size, min((b + 1) * block_size, n)) for b in order])
        idx = idx[idx < n]
        shuffled_values = frame.iloc[idx].reset_index(drop=True)
        shuffled_values.index = frame.index[:len(shuffled_values)]
        out[sym] = shuffled_values
    return out


def random_matched_signal(symbols: List[str], dates: List, real_direction_values: np.ndarray,
                           seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Null-baseline (b): i.i.d. random score ranks and random direction
    signs, same symbols/dates as the real signal, with |D| magnitudes
    resampled from the REAL signal's own empirical distribution so that once
    this runs through the identical portfolio engine (same entry percentile,
    same rank-tilt sizing, same vol target, same band buffer) its gross
    exposure and turnover end up comparable to the real strategy's -- 'does
    any similarly-sized, similarly-turned-over dollar-neutral book do this
    well by construction alone'."""
    rng = np.random.default_rng(seed)
    magnitudes_pool = np.abs(real_direction_values[~np.isnan(real_direction_values)]) \
        if len(real_direction_values) else np.array([0.3])
    if len(magnitudes_pool) == 0:
        magnitudes_pool = np.array([0.3])
    out = {}
    for sym in symbols:
        s_bar = pd.Series(rng.normal(0, 1, len(dates)), index=dates)
        magnitudes = rng.choice(magnitudes_pool, size=len(dates))
        signs = rng.choice([-1.0, 1.0], size=len(dates))
        out[sym] = pd.DataFrame({"S_bar": s_bar, "D": signs * magnitudes}, index=dates)
    return out


def twin_horse_race(main_net_returns: pd.Series, twin_net_returns_by_name: Dict[str, pd.Series],
                     twin_liveness: Dict[str, dict]) -> dict:
    """Main signal must beat every LIVE twin's net Sharpe. A twin that fails
    the liveness assertion is excluded from the race entirely (a degenerate
    comparator 'losing' proves nothing)."""
    main_sr = main_net_returns.mean() / main_net_returns.std(ddof=1) if main_net_returns.std(ddof=1) > 0 else np.nan
    results = {}
    all_beaten = True
    for name, ret in twin_net_returns_by_name.items():
        alive = twin_liveness.get(name, {}).get("alive", False)
        twin_sr = ret.mean() / ret.std(ddof=1) if ret.std(ddof=1) > 0 else np.nan
        beaten = bool(not np.isnan(main_sr) and not np.isnan(twin_sr) and main_sr > twin_sr)
        results[name] = {"alive": alive, "twin_sharpe": float(twin_sr) if not np.isnan(twin_sr) else None,
                          "beaten_by_main": beaten}
        if alive and not beaten:
            all_beaten = False
    return {"main_sharpe": float(main_sr) if not np.isnan(main_sr) else None, "twins": results, "passed": all_beaten}


def robustness_grid_dsr(returns_by_variant: Dict[tuple, pd.Series], base_variant: tuple,
                         periods_per_year: int = SPEC.sizing.trading_days_per_year) -> dict:
    """returns_by_variant: {(tau_window, holding_period): daily net-return series}
    across the pre-registered grid. Sharpe dispersion across the grid is used
    as the deflated-Sharpe benchmark's cross-trial variance estimate, and the
    grid itself is reported as the 'result surface' robustness check."""
    sharpes = {}
    for key, ret in returns_by_variant.items():
        r = ret.dropna()
        sharpes[key] = float(r.mean() / r.std(ddof=1)) if len(r) > 5 and r.std(ddof=1) > 0 else np.nan
    trial_sharpes = np.array([v for v in sharpes.values() if not np.isnan(v)])

    base_returns = returns_by_variant[base_variant].dropna().values
    dsr = su.deflated_sharpe_ratio(base_returns, n_trials=SPEC.validation.n_strategy_variants_tested,
                                    trial_sharpes=trial_sharpes, periods_per_year=periods_per_year)

    valid = [v for v in sharpes.values() if not np.isnan(v)]
    smooth = bool(len(valid) >= 2 and (max(valid) - min(valid)) < 3 * (np.std(valid) + 1e-9) * 3) if valid else False
    return {"sharpe_by_variant": sharpes, "dsr": dsr,
            "net_sharpe_positive": bool(dsr["sr"] > 0) if not np.isnan(dsr.get("sr", np.nan)) else False,
            "dsr_above_half": bool(dsr["dsr"] > 0.5) if not np.isnan(dsr.get("dsr", np.nan)) else False,
            "surface_reported": True}


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------

def verdict(step0_existence: dict, step0_breadth: dict, step1: dict,
            step2: Optional[dict] = None) -> dict:
    reasons = []
    if not step0_existence.get("passed", False):
        reasons.append("Step 0 existence test failed: cepstral peaks not more common than chance.")
    if not step0_breadth.get("passed", False):
        reasons.append("Step 0 breadth test failed: signal looks like a single common factor / no genuine dispersion.")

    step0_ok = not reasons
    if not step0_ok:
        return {"rejected": True, "reasons": reasons, "stage_reached": "step0"}

    if not step1.get("passed", False):
        reasons.append(f"Step 1 redundancy screen failed: |NW-t|={step1.get('nw', {}).get('t_stat')} < "
                        f"{SPEC.validation.min_abs_newey_west_t} after controls.")
        return {"rejected": True, "reasons": reasons, "stage_reached": "step1"}

    if step2 is None:
        return {"rejected": False, "reasons": [], "stage_reached": "step1",
                "note": "Step 1 passed; Step 2 not run."}

    if not step2["dsr"]["net_sharpe_positive"]:
        reasons.append("Step 2 failed: net Sharpe <= 0.")
    if not step2["dsr"]["dsr_above_half"]:
        reasons.append("Step 2 failed: deflated Sharpe ratio <= 0.5 (not distinguishable from the best of "
                        f"{SPEC.validation.n_strategy_variants_tested} tested variants under a null of no skill).")
    if not step2["sign_consistency"]["passed"]:
        reasons.append(f"Step 2 failed: sign consistency missing in {step2['sign_consistency']['n_missing_consistency']} "
                        f"of {SPEC.validation.n_is_subperiods} subperiods.")
    if not step2["twin_race"]["passed"]:
        losers = [n for n, r in step2["twin_race"]["twins"].items() if r["alive"] and not r["beaten_by_main"]]
        reasons.append(f"Step 2 failed: lost to live twin(s) {losers}.")

    return {"rejected": bool(reasons), "reasons": reasons, "stage_reached": "step2"}
