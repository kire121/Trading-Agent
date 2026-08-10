"""
The core hypothesis, as math:

  u_t       = log(1+v_t) - u_hat_t              detrend (21d rolling per-minute-of-day profile)
  C(tau)    = IFFT{ log(|FFT{u}|^2 + eps) }      real cepstrum, quefrency tau in [2,45] minutes
  Chat_i(tau) = (C_i(tau) - median_j C_j(tau)) / MAD_j(tau)   cross-sectional standardization per quefrency/day
  s_i,d     = max_tau Chat_i,d(tau)              daily slicing score
  Sbar_i    = 5-day mean of s_i,d
  tau*      = argmax_tau of the 5-day-mean Chat curve (see note below)
  m_t       = comb filter at tau*: phases of (t mod tau*) with elevated detrended
              volume, folded/averaged over the same 5-day window
  D_i       = sum(sign(r_t) * v_t * m_t) / sum(v_t * m_t) over the 5-day window

Interpretation note on tau*: the spec gives "s_id = max_tau Chat_id(tau); Sbar_i
= 5-day mean; tau* = argmax" without pinning down whether tau* is a per-day
argmax or the argmax of the smoothed curve. This module uses the argmax of the
*5-day-averaged* Chat(tau) curve (rolling_tau_star_window), because tau* feeds
a comb filter whose mask is applied across that same 5-day window in the D
computation -- a mask built from a single noisy day's argmax would jump
target-period every day and produce an incoherent mask over the window. The
per-day argmax (tau_star_day) is still computed and kept as a diagnostic.

Quefrency-as-minutes: the trimmed RTH grid (bars.session_minute_grid) is
sampled at 1 sample/minute, so an N-point FFT's quefrency index k already
equals a lag of k minutes -- no unit conversion needed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import SPEC


# ---------------------------------------------------------------------------
# Per-symbol: dense grid -> wide matrices -> detrended log-volume -> cepstrum
# ---------------------------------------------------------------------------

def volume_wide(dense: pd.DataFrame) -> pd.DataFrame:
    return dense.pivot(index="session_date", columns="minute_of_day", values="volume").sort_index()


def ret_wide(dense: pd.DataFrame) -> pd.DataFrame:
    return dense.pivot(index="session_date", columns="minute_of_day", values="ret").sort_index()


def detrend(vol_wide: pd.DataFrame, window: int = SPEC.cepstrum.detrend_window_days) -> pd.DataFrame:
    """u_t = log(1+v_t) - rolling 21-session profile at that minute-of-day,
    profile computed from the *preceding* `window` sessions only (causal)."""
    log1p = np.log1p(vol_wide)
    profile = log1p.shift(1).rolling(window, min_periods=window).mean()
    return log1p - profile


def real_cepstrum_row(u_row: np.ndarray, eps: float = SPEC.cepstrum.epsilon) -> np.ndarray:
    spectrum = np.fft.fft(u_row)
    log_power = np.log(np.abs(spectrum) ** 2 + eps)
    return np.fft.ifft(log_power).real


def cepstrum_wide(u_wide: pd.DataFrame, qmin: int = SPEC.cepstrum.quefrency_min_min,
                   qmax: int = SPEC.cepstrum.quefrency_max_min, eps: float = SPEC.cepstrum.epsilon) -> pd.DataFrame:
    """Raw (pre cross-sectional standardization) cepstrum restricted to the
    quefrency band of interest. Rows with an incomplete detrend warm-up are NaN."""
    cols = list(range(qmin, qmax + 1))
    out = np.full((len(u_wide), len(cols)), np.nan)
    values = u_wide.values
    complete = ~np.isnan(values).any(axis=1)
    for i in np.nonzero(complete)[0]:
        out[i, :] = real_cepstrum_row(values[i, :], eps)[qmin:qmax + 1]
    return pd.DataFrame(out, index=u_wide.index, columns=cols)


# ---------------------------------------------------------------------------
# Cross-sectional standardization (needs all symbols' raw cepstra per day)
# ---------------------------------------------------------------------------

def cross_sectional_standardize_panel(raw_by_symbol: Dict[str, pd.DataFrame],
                                       min_cross_section: int = 10) -> Dict[str, pd.DataFrame]:
    """raw_by_symbol[symbol] is a cepstrum_wide() frame (index=session_date,
    columns=quefrency). Standardizes each (date, quefrency) cell against the
    cross-section of symbols that have a valid (non-NaN) value that day, using
    median/MAD -- robust to the handful of names with fat-tailed cepstral
    outliers, and this is what strips out market-wide periodicity (the same
    on-the-hour hedging rhythm affects every name's raw C(tau) equally, so it
    cancels in the median subtraction)."""
    all_dates = sorted(set().union(*[df.index for df in raw_by_symbol.values()]))
    symbols = list(raw_by_symbol.keys())
    cols = next(iter(raw_by_symbol.values())).columns

    aligned = {s: raw_by_symbol[s].reindex(all_dates) for s in symbols}
    standardized = {s: pd.DataFrame(np.nan, index=all_dates, columns=cols) for s in symbols}

    for date in all_dates:
        day_mat = pd.DataFrame({s: aligned[s].loc[date] for s in symbols})  # index=quefrency, columns=symbol
        valid_cols = day_mat.columns[day_mat.notna().all(axis=0)]
        if len(valid_cols) < min_cross_section:
            continue
        sub = day_mat[valid_cols]
        med = sub.median(axis=1)
        mad = sub.sub(med, axis=0).abs().median(axis=1).replace(0, np.nan)
        z = sub.sub(med, axis=0).div(mad, axis=0)
        for s in valid_cols:
            standardized[s].loc[date] = z[s].values

    return standardized


# ---------------------------------------------------------------------------
# Score, tau*, direction
# ---------------------------------------------------------------------------

def _argmax_prefer_fundamental(vals: np.ndarray, cols: np.ndarray, tol: float = 0.75) -> int:
    """argmax with an octave-error correction borrowed from cepstral pitch
    detection: a period-tau comb aliases onto every integer multiple of tau
    (2*tau, 3*tau, ...), and the harmonic sometimes outranks the fundamental
    itself, so the raw argmax is biased toward higher multiples. If a peak at
    an integer sub-multiple of the raw argmax is within `tol` of the top
    peak's height, prefer the smallest such sub-multiple."""
    idx_max = int(np.argmax(vals))
    peak_val = vals[idx_max]
    peak_tau = cols[idx_max]
    best_idx = idx_max
    for divisor in range(2, 6):
        if peak_tau % divisor != 0:
            continue
        candidate_tau = peak_tau // divisor
        matches = np.nonzero(cols == candidate_tau)[0]
        if len(matches) == 0:
            continue
        cand_idx = matches[0]
        if vals[cand_idx] >= tol * peak_val:
            best_idx = cand_idx
    return best_idx


def _row_argmax_tau(vals: np.ndarray, cols: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(vals).all(axis=1)
    filled = np.where(np.isnan(vals), -np.inf, vals)
    tau = np.full(len(vals), np.nan)
    for i in np.nonzero(valid)[0]:
        tau[i] = cols[_argmax_prefer_fundamental(filled[i], cols)]
    return tau, valid


def score_and_tau_day(Chat_wide: pd.DataFrame) -> pd.DataFrame:
    """Per-day s_i,d = max_tau Chat(tau) and the day's own argmax (diagnostic).
    s_i,d itself always uses the true (unadjusted) peak height; only the
    reported tau location goes through the fundamental-preference correction."""
    cols = np.asarray(Chat_wide.columns)
    vals = Chat_wide.values
    valid = ~np.isnan(vals).all(axis=1)
    filled = np.where(np.isnan(vals), -np.inf, vals)
    idx_max = np.argmax(filled, axis=1)
    s = np.where(valid, vals[np.arange(len(vals)), idx_max], np.nan)
    tau, _ = _row_argmax_tau(vals, cols)
    return pd.DataFrame({"s": s, "tau_star_day": tau}, index=Chat_wide.index)


def rolling_S_bar(s: pd.Series, window: int = SPEC.cepstrum.slicing_score_avg_days) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def rolling_tau_star_window(Chat_wide: pd.DataFrame, window: int = SPEC.cepstrum.direction_window_days) -> pd.Series:
    """tau* used for the comb filter: argmax (fundamental-preferred) of the
    window-averaged Chat(tau) curve."""
    smoothed = Chat_wide.rolling(window, min_periods=window).mean()
    cols = np.asarray(smoothed.columns)
    tau, _ = _row_argmax_tau(smoothed.values, cols)
    return pd.Series(tau, index=Chat_wide.index)


def comb_burst_phases(u_wide: pd.DataFrame, dates: Iterable, tau_star: int,
                       burst_quantile: float = SPEC.cepstrum.burst_quantile) -> np.ndarray:
    """Comb filter at period tau_star, implemented as phase-synchronous
    averaging (fold the detrended series at period tau_star, average across
    the window) -- the time-domain dual of spectral-comb liftering, and the
    same 'fold and average at the candidate period' operation Bogert-Healy-
    Tukey used to pull an echo out of a seismogram. Returns the set of phases
    (0..tau_star-1) whose folded-average detrended log-volume sits in the top
    (1 - burst_quantile) fraction -- i.e. where the recurring child-order
    burst actually lands within its tau_star-minute cycle."""
    rows = u_wide.loc[list(dates)].values
    n_minutes = rows.shape[1]
    phase_of_t = np.arange(n_minutes) % tau_star
    profile = np.array([np.nanmean(rows[:, phase_of_t == p]) for p in range(tau_star)])
    order = np.argsort(profile)[::-1]
    n_select = int(np.clip(round((1 - burst_quantile) * tau_star), 1, tau_star - 1))
    return np.sort(order[:n_select])


def direction_for_window(vol_wide: pd.DataFrame, ret_w: pd.DataFrame, dates: Iterable,
                          tau_star: int, burst_phases: np.ndarray) -> float:
    """D_i = sum(sign(r_t) * v_t * m_t) / sum(v_t * m_t) over the given dates,
    restricted to minutes whose phase (t mod tau_star) is a burst phase."""
    dates = list(dates)
    n_minutes = vol_wide.shape[1]
    phase_of_t = np.arange(n_minutes) % tau_star
    mask = np.isin(phase_of_t, burst_phases)
    v = vol_wide.loc[dates].values[:, mask]
    r = ret_w.loc[dates].values[:, mask]
    weight = np.nan_to_num(v, nan=0.0)
    denom = weight.sum()
    if denom <= 0:
        return np.nan
    numer = np.nansum(np.sign(r) * weight)
    return float(numer / denom)


def signal_frame_for_symbol(dense: pd.DataFrame, Chat_wide: pd.DataFrame) -> pd.DataFrame:
    """Combines the cross-sectionally standardized cepstrum with the raw
    volume/return grid to produce the full per-day signal panel for one
    symbol: s, tau_star_day, S_bar, tau_star_window, D."""
    vw = volume_wide(dense)
    rw = ret_wide(dense)
    uw = detrend(vw)

    day_stats = score_and_tau_day(Chat_wide)
    day_stats["S_bar"] = rolling_S_bar(day_stats["s"])
    day_stats["tau_star_window"] = rolling_tau_star_window(Chat_wide)

    window = SPEC.cepstrum.direction_window_days
    dates = list(Chat_wide.index)
    D = pd.Series(np.nan, index=dates)
    for i in range(window - 1, len(dates)):
        d = dates[i]
        tau = day_stats["tau_star_window"].loc[d]
        if pd.isna(tau) or d not in uw.index:
            continue
        tau = int(tau)
        window_dates = [dt for dt in dates[i - window + 1: i + 1] if dt in uw.index]
        if len(window_dates) < window:
            continue
        phases = comb_burst_phases(uw, window_dates, tau)
        D.loc[d] = direction_for_window(vw, rw, window_dates, tau, phases)
    day_stats["D"] = D
    return day_stats
