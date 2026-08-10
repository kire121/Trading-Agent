"""The Levy-area price/volume rotation signal.

Per-name construction, brief notation:

    u_s   = sign(r_s) * DV_s / ADV60        (signed, ADV-normalized dollar volume)
    P     = cumsum(r) over the trailing n-day window       -> z-scored -> P~
    V     = cumsum(u) over the trailing n-day window        -> z-scored -> V~
    A_i   = (1/2) * sum_s ( P~_{s-1} * dV~_s  -  V~_{s-1} * dP~_s )     (discrete Levy area)
    q_i   = -A_i                             (q > 0  <=>  volume led price)
    s_i   = z_cs(q_i, winsorized 1/99%) * sign(R_i^(n))

A note on the anchoring point, because it changes the number:

The brief also states an identity meant to double-check A -- "A = (1/2)
sum_{s'<s} (dP~_{s'} dV~_s - dP~_s dV~_{s'})", an aggregated,
antisymmetrized cross-covariance over all lags, i.e. a pure function of the
path's *increments*. That double-sum form is invariant to shifting the
whole (P~, V~) path by a constant (increments don't see a constant shift).
The shoelace form above is *not* shift-invariant in general -- expanding
the telescoping sum shows

    shoelace(P~, V~) = double_sum(P~, V~) + (P~_0 * V~_n - V~_0 * P~_n)

i.e. the two formulas differ by a boundary term that vanishes only when the
window's path starts at the origin (P~_0 = V~_0 = 0). A window's z-scored
levels do *not* generally start at zero (z-scoring centers on the window's
*mean*, not its first value), so applying the shoelace formula literally to
the raw z-scored series would make it disagree with the brief's own stated
identity by an O(1) boundary term -- not negligible at n=20.

We resolve this by anchoring the z-scored path at its own first point
before computing the area: P^ = P~ - P~_0, V^ = V~ - V~_0. This is (a) the
translation-invariant, standard rough-path/level-2-signature reading of a
"Levy area" (an antisymmetric functional of increments has no business
depending on where in (P,V)-space the window happens to start), and (b)
the reading under which the brief's shoelace formula and its own
double-sum identity are *exactly* equal, verified in
tests/test_signal.py::test_levy_area_shoelace_equals_double_sum_identity
rather than assumed.

Under this anchoring, dP^_s = dP~_s = r_s / std(P) and dV^_s = u_s / std(V)
(z-scoring only rescales increments by the window's own constant std), so

    A_i = (1/(std(P) * std(V))) * sum_{s'<s} (r_{s'} u_s - r_s u_{s'})
        = B(r, u) / (std(P) * std(V))                                  (*)

-- the Levy area of the z-scored paths is a rescaled Levy area of the raw
(r, u) pairs, where B(r, u) := sum_{s'<s}(r_s' u_s - r_s u_s') is the raw,
*unnormalized* double sum.

Two exact algebraic facts about the *unnormalized* shoelace of the raw
cumulative paths, W(r, u) := shoelace(cumsum(r), cumsum(u)) (no z-scoring,
i.e. B(r, u) before dividing by std(P)*std(V)), are proven and unit-tested
in tests/test_signal.py -- these are what the brief's "identity that forces
honesty" paragraph is actually describing ("aggregerad antisymmetriserad
korskovarians" -- a *covariance*, i.e. an unnormalized second-moment
object, not a standardized one):

  1. E[W] = 0 EXACTLY under a uniformly random permutation of day-order,
     for any fixed multiset of (r, u) pairs (verified by exact brute-force
     enumeration of all n! permutations at small n). This is the brief's
     "Huvudnull" ("E[A]=0 exakt under utbytbarhet").
  2. W flips sign exactly under day-order reversal: W(reverse(r, u)) =
     -W(r, u), for any (r, u).

Because std(P) and std(V) in (*) are computed from the window's own
*cumulative* path, they are themselves (weakly) order-dependent -- e.g.
std(cumsum(r)) is generally not exactly equal to std(cumsum(reverse(r))).
Both denominators are always positive, so fact 2 lifts cleanly to an exact
statement about the production A's SIGN (A(reverse) has the opposite sign
of A, always, exactly), but not to an exact statement about magnitude, and
E[W]=0 does not by itself imply E[A]=0 for the normalized ratio (a ratio of
dependent random variables). In practice the denominator's
order-dependence is small and E[A] is very close to zero under the shuffle
null -- but this is verified empirically (stats.py's Monte Carlo shuffle
test, 1000 reps), not claimed as an exact identity the way it is for W.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from vridmomentet.config import SignalParams
from vridmomentet.data import Panel


# --------------------------------------------------------------------------
# u_s: signed, ADV-normalized dollar volume
# --------------------------------------------------------------------------

def signed_dollar_volume(panel: Panel) -> pd.DataFrame:
    """u_s = sign(r_s) * DV_s / ADV60."""
    sign_r = np.sign(panel.log_returns)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = sign_r * panel.dollar_volume / panel.adv60
    return u.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# Core kernel: anchored, z-scored discrete Levy area of a single (r, u)
# window. Vectorized over an arbitrary leading batch dimension so the same
# function computes one window (tests, null-hypothesis resampling in
# stats.py) or thousands at once (the rolling panel computation below).
# --------------------------------------------------------------------------

def levy_area_of_windows(r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """r, u: arrays of shape (..., window). Returns shape (...,): the
    anchored, within-window-z-scored discrete Levy area A for each window.
    NaN wherever a window has any NaN input or a degenerate (zero-variance)
    cumulative path.
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)
    valid = ~np.isnan(r).any(axis=-1) & ~np.isnan(u).any(axis=-1)

    P = np.cumsum(r, axis=-1)
    V = np.cumsum(u, axis=-1)

    P_mean = P.mean(axis=-1, keepdims=True)
    V_mean = V.mean(axis=-1, keepdims=True)
    P_std = P.std(axis=-1, ddof=1, keepdims=True)
    V_std = V.std(axis=-1, ddof=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        P_tilde = (P - P_mean) / P_std
        V_tilde = (V - V_mean) / V_std

    P_hat = P_tilde - P_tilde[..., [0]]
    V_hat = V_tilde - V_tilde[..., [0]]

    area = 0.5 * np.sum(P_hat[..., :-1] * V_hat[..., 1:] - P_hat[..., 1:] * V_hat[..., :-1], axis=-1)

    degenerate = (P_std[..., 0] <= 0) | (V_std[..., 0] <= 0) | np.isnan(P_std[..., 0]) | np.isnan(V_std[..., 0])
    area = np.where(valid & ~degenerate, area, np.nan)
    return area


def _rolling_levy_area_1d(r: np.ndarray, u: np.ndarray, window: int) -> np.ndarray:
    n = len(r)
    out = np.full(n, np.nan)
    if n < window:
        return out
    r_windows = sliding_window_view(r, window)
    u_windows = sliding_window_view(u, window)
    out[window - 1 :] = levy_area_of_windows(r_windows, u_windows)
    return out


def rolling_levy_area(log_returns: pd.DataFrame, u: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-ticker trailing `window`-day discrete Levy area A, causal (the
    value at date t uses only the n days ending at and including t).
    """
    cols = {}
    r_vals = log_returns.values
    u_vals = u.values
    for j, col in enumerate(log_returns.columns):
        cols[col] = _rolling_levy_area_1d(r_vals[:, j], u_vals[:, j], window)
    return pd.DataFrame(cols, index=log_returns.index)


# --------------------------------------------------------------------------
# Formation return R^(n) and the tanh(R/sigma) neighborhood variant
# --------------------------------------------------------------------------

def formation_return(log_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """R_i^(n): cumulative log return over the trailing n-day window."""
    return log_returns.rolling(window, min_periods=window).sum()


def direction_factor(log_returns: pd.DataFrame, window: int, transform: str, tanh_scale_days: int) -> pd.DataFrame:
    """sign(R^(n)) (brief default) or tanh(R^(n) / (daily_vol * sqrt(n)))
    (declared neighborhood variant: R measured in trailing-vol units,
    passed through tanh for smooth saturation instead of a hard sign).
    """
    r_n = formation_return(log_returns, window)
    if transform == "sign":
        return np.sign(r_n)
    if transform == "tanh":
        daily_vol = log_returns.rolling(tanh_scale_days, min_periods=max(15, tanh_scale_days // 2)).std(ddof=1)
        scale = daily_vol * np.sqrt(window)
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = r_n / scale
        return np.tanh(scaled)
    raise ValueError(f"unknown return_transform {transform!r}")


# --------------------------------------------------------------------------
# Cross-sectional step: winsorize, z-score, multiply by direction
# --------------------------------------------------------------------------

def cross_sectional_zscore(df: pd.DataFrame, winsor_lo: float, winsor_hi: float) -> pd.DataFrame:
    """Per-row (per-date) winsorization at [winsor_lo, winsor_hi] quantiles,
    then per-row z-score. NaNs are ignored in both the quantile and the
    mean/std computation and left as NaN in the output. Rows that are
    entirely NaN (e.g. every date before any name has burned in its
    rolling windows) are an expected, not exceptional, occurrence here --
    numpy's nan* reductions warn on an all-NaN slice by design, so that
    warning is deliberately suppressed rather than left to spam the console
    on every one of those rows.
    """
    values = df.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanquantile(values, winsor_lo, axis=1, keepdims=True)
        hi = np.nanquantile(values, winsor_hi, axis=1, keepdims=True)
        clipped = np.clip(values, lo, hi)
        mean = np.nanmean(clipped, axis=1, keepdims=True)
        std = np.nanstd(clipped, axis=1, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (clipped - mean) / std
    z = np.where(np.isnan(values), np.nan, z)
    return pd.DataFrame(z, index=df.index, columns=df.columns)


class SignalResult:
    def __init__(self, u: pd.DataFrame, q: pd.DataFrame, r_n: pd.DataFrame, s: pd.DataFrame):
        self.u = u
        self.q = q
        self.r_n = r_n
        self.s = s


def compute_signal(panel: Panel, params: SignalParams) -> SignalResult:
    u = signed_dollar_volume(panel)
    a = rolling_levy_area(panel.log_returns, u, params.window_days)
    q = -a  # brief: q_i = -A_i, so q > 0 <=> volume led price
    r_n = formation_return(panel.log_returns, params.window_days)
    direction = direction_factor(panel.log_returns, params.window_days, params.return_transform, params.tanh_scale_days)
    z_q = cross_sectional_zscore(q, params.winsor_lo, params.winsor_hi)
    s = z_q * direction  # direction is already sign(R^(n)) or tanh(R^(n)/sigma), per params.return_transform
    return SignalResult(u=u, q=q, r_n=r_n, s=s)
