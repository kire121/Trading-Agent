"""
Fasflocken (PH-1) -- signal primitives.

Pure math, no data-provider or portfolio concerns:

  1. causal_bandpass       -- Butterworth order-2 bandpass via sosfilt (never filtfilt).
  2. rolling_analytic_phase -- rolling-window Hilbert transform, phase of the last point.
  3. kuramoto_order_parameter -- amplitude-free cross-sectional coherence R_s(t).
  4. resample_weekly_last / rolling_zscore -- weekly Z_s(t) vs trailing history.

Causality note (also see backtest.py): every function here computes its
output at time t using only inputs at times <= t (sosfilt is a one-pass
IIR filter with no forward look; the rolling Hilbert window ends at t and
only the phase of its *last* sample is kept). Precomputing the full series
in one vectorized call is mathematically identical to computing it
"online" day by day during a backtest loop -- the causality property is a
statement about each output's input set, not about when the code happens
to execute. Point-in-time *universe membership* (who counts as a sector
constituent on a given day) is a separate concern handled in universe.py /
pipeline.py, not here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.signal import hilbert as _hilbert
from numpy.lib.stride_tricks import sliding_window_view


def _design_bandpass_sos(low_period_days: float, high_period_days: float, order: int, fs: float = 1.0):
    if low_period_days <= 0 or high_period_days <= 0:
        raise ValueError("cycle periods must be positive")
    if high_period_days <= low_period_days:
        raise ValueError("high_period_days must exceed low_period_days")
    nyquist = fs / 2.0
    f_low = 1.0 / high_period_days   # slower cycle -> lower frequency edge
    f_high = 1.0 / low_period_days   # faster cycle -> higher frequency edge
    if not (0 < f_low < f_high < nyquist):
        raise ValueError(
            f"band [{f_low}, {f_high}] cycles/sample is not within (0, Nyquist={nyquist}); "
            "low_period_days is too short relative to the sampling interval"
        )
    return butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")


def _bridge_short_gaps(segment: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill each maximal run of consecutive NaNs with 0.0 if (and only if)
    that run's own length is <= max_gap; longer runs are left untouched.

    Deliberately NOT linear interpolation: interpolating between the
    values before and after a gap requires the value *after* it, which for
    a gap sitting near the end of a rolling-window computation would be
    tomorrow's data feeding into today's causal filter output. A constant
    0.0 fill (read: "assume no price change on a halted trading day")
    uses zero information from either side, so it can never leak a future
    observation backward -- unlike pandas' `Series.interpolate(...,
    limit=N)`, which still anchors on the next valid value even when
    `limit_direction='forward'` is set, and whose `limit` is a *global*
    per-series fill budget rather than a per-gap one.
    """
    out = segment.copy()
    is_nan = np.isnan(out)
    if not is_nan.any():
        return out
    edges = np.diff(np.concatenate(([0], is_nan.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap:
            out[s:e] = 0.0
    return out


def _causal_bandpass_1d(x: np.ndarray, sos: np.ndarray, max_gap: int = 5) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    valid = ~np.isnan(x)
    if not valid.any():
        return out
    first = int(np.argmax(valid))
    last = len(valid) - 1 - int(np.argmax(valid[::-1]))
    segment = x[first : last + 1].copy()

    internal_nan = np.isnan(segment)
    if internal_nan.any():
        # Bridge short internal gaps (trading halts, missing prints): each
        # run of at most max_gap consecutive NaNs is zero-filled. A gap
        # longer than max_gap is left as NaN entirely; sosfilt's IIR state
        # then stays NaN from that point through the *rest of the
        # segment* -- including any later stretch of otherwise-valid data
        # -- which conservatively excludes that ticker's tail rather than
        # fabricating a bridge across a multi-year hole. A name with a
        # genuine long absence should in practice get a fresh membership
        # interval (see universe.py), which causal_bandpass never sees
        # stitched together in the first place.
        segment = _bridge_short_gaps(segment, max_gap)

    filtered = sosfilt(sos, segment)
    out[first : last + 1] = filtered
    return out


def causal_bandpass(
    x: np.ndarray | pd.Series | pd.DataFrame,
    low_period_days: float,
    high_period_days: float,
    order: int = 2,
    fs: float = 1.0,
    max_gap: int = 5,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Causal Butterworth bandpass, applied independently per column.

    x: 1D (T,) or 2D (T, N) array-like of e.g. daily log returns. NaNs are
    treated as "not yet in the universe / delisted" at the edges (kept
    NaN in the output) and as internal gaps elsewhere: gaps of at most
    `max_gap` samples are linearly interpolated before filtering (trading
    halts, missing prints); longer gaps are left as NaN, which
    conservatively poisons the causal filter's state for the rest of that
    segment rather than bridging a multi-year hole.

    Returns the same type/shape as the input.
    """
    sos = _design_bandpass_sos(low_period_days, high_period_days, order, fs)

    if isinstance(x, pd.DataFrame):
        out = {col: _causal_bandpass_1d(x[col].to_numpy(dtype=float), sos, max_gap) for col in x.columns}
        return pd.DataFrame(out, index=x.index)
    if isinstance(x, pd.Series):
        return pd.Series(_causal_bandpass_1d(x.to_numpy(dtype=float), sos, max_gap), index=x.index, name=x.name)

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return _causal_bandpass_1d(arr, sos, max_gap)
    if arr.ndim == 2:
        return np.column_stack([_causal_bandpass_1d(arr[:, j], sos, max_gap) for j in range(arr.shape[1])])
    raise ValueError("x must be 1D or 2D")


def _rolling_analytic_phase_1d(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out

    windows = sliding_window_view(x, window_shape=window)  # shape (n - window + 1, window)
    valid_rows = ~np.isnan(windows).any(axis=1)
    if not valid_rows.any():
        return out

    analytic = np.full(windows.shape, np.nan, dtype=complex)
    analytic[valid_rows] = _hilbert(windows[valid_rows], axis=-1)
    last_point_phase = np.angle(analytic[:, -1])
    out[window - 1 :] = last_point_phase
    return out


def rolling_analytic_phase(
    x: np.ndarray | pd.Series | pd.DataFrame, window: int = 90
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Phase of the last point of the analytic signal on each rolling window.

    phi_t = angle(hilbert(x[t-window+1 : t+1]))[-1]

    This is the standard fix for Hilbert's global-transform look-ahead:
    computing the analytic signal on the *entire* series and reading off
    phi_t would let the filter "see" the whole future via the FFT; here
    each phi_t is computed from a window that ends at t and nothing after.
    """
    if isinstance(x, pd.DataFrame):
        out = {col: _rolling_analytic_phase_1d(x[col].to_numpy(dtype=float), window) for col in x.columns}
        return pd.DataFrame(out, index=x.index)
    if isinstance(x, pd.Series):
        return pd.Series(_rolling_analytic_phase_1d(x.to_numpy(dtype=float), window), index=x.index, name=x.name)

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return _rolling_analytic_phase_1d(arr, window)
    if arr.ndim == 2:
        return np.column_stack([_rolling_analytic_phase_1d(arr[:, j], window) for j in range(arr.shape[1])])
    raise ValueError("x must be 1D or 2D")


def kuramoto_order_parameter(
    phase: np.ndarray | pd.DataFrame, min_constituents: int = 5
) -> np.ndarray | pd.Series:
    """R_t = | mean_j exp(i * phi_{j,t}) |, amplitude-free cross-sectional coherence.

    phase: (T, N) matrix of phase angles (radians), NaN where a name is
    absent (not yet a constituent, insufficient window history, delisted).
    Rows with fewer than `min_constituents` valid phases return NaN.
    """
    is_df = isinstance(phase, pd.DataFrame)
    arr = phase.to_numpy(dtype=float) if is_df else np.asarray(phase, dtype=float)
    if arr.ndim != 2:
        raise ValueError("phase must be 2D (T, N)")

    cos_p = np.cos(arr)
    sin_p = np.sin(arr)
    n_valid = np.sum(~np.isnan(arr), axis=1)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows before burn-in
        mean_c = np.nanmean(cos_p, axis=1)
        mean_s = np.nanmean(sin_p, axis=1)
    r = np.sqrt(mean_c**2 + mean_s**2)
    r[n_valid < min_constituents] = np.nan

    if is_df:
        return pd.Series(r, index=phase.index, name="R")
    return r


def resample_weekly_last(daily: pd.Series) -> pd.Series:
    """Collapse a daily series to one value per week, labeled on Friday: the
    last available (non-NaN) trading-day observation that week -- a
    holiday-robust stand-in for "Friday close" (if Friday itself is a
    holiday, Thursday's close is used, matching how the strategy would
    actually observe the market).
    """
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise TypeError("daily must be indexed by DatetimeIndex")
    out = daily.resample("W-FRI").last()
    out.name = daily.name
    return out


def rolling_zscore(weekly: pd.Series, lookback_weeks: int, min_periods: int | None = None) -> pd.Series:
    """Trailing rolling Z-score, inclusive of the current observation.

    min_periods defaults to `lookback_weeks` (full burn-in required, per
    spec's trailing-104-week convention) but can be relaxed for shorter
    sample windows in tests.
    """
    mp = lookback_weeks if min_periods is None else min_periods
    roll = weekly.rolling(window=lookback_weeks, min_periods=mp)
    mean = roll.mean()
    std = roll.std(ddof=1)
    z = (weekly - mean) / std
    z[std == 0] = np.nan
    return z
