# Proveniens: research/timglaset/opclock.py, branch
# claude/strategy-spec-implementation-axn5co, commit 408c81f. Flyttad till
# lib/ vid lib-konsolideringen (docs/INSTRUKTION.md avsnitt 7).
# Generaliserad vid flytten: parametern "volume" heter nu "activity"
# (godtycklig aktivitetsproxy -- volym, realiserad varians, antal avslut,
# ...); tidigare Timglaset-specifika defaultvärden (window=252, cap=5.0,
# variance-fönster=5) borttagna ur signaturerna -- anroparen skickar dem
# nu uttryckligen från sin egen strategikonfig. Algoritmen och de tre
# kärnfunktionernas (compute_tau, compute_op_ewma, compute_signal)
# beteende är oförändrat.
"""Subordination / stochastic time-change clock machinery.

An operational-time clock: a persistence estimator (EWMA-style memory)
whose decay runs per unit of ACTIVITY instead of per calendar day. Busy
periods forget quickly; quiet periods retain memory longer. `activity` is
supplied by the caller as an arbitrary proxy with the same shape as the
returns panel -- trading volume, realized variance (see
`variance_clock_input`), number of trades/executions, or anything else
with a meaningful "more activity now" reading. The calendar twin is the
SAME code path with tau held at 1 (no parallel implementation) -- see
`calendar_twin_tau`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tau(activity: pd.DataFrame, window: int, cap: float) -> pd.DataFrame:
    """tau_t = clip(activity_t / rolling_median(activity, window).shift(1), 0, cap).

    NaN or undefined normalizer (< window obs) -> tau = 1.0 (neutral clock
    speed). A NaN activity_t itself, or a zero activity_t, is folded into
    the same "input not available -> neutral clock speed" fallback.

    min_periods=window (strict: the trailing `window` observations must
    all be non-null) is used for the rolling median: a single missing
    observation therefore extends the "undefined normalizer" fallback
    across the following ~window days, diluting real clock-speed signal
    into neutral tau=1 days rather than tolerating the gap.

    `activity` must already be in whatever normalized form the caller
    wants compared bar-to-bar (e.g. split-adjusted volume).
    """
    normalizer = activity.rolling(window, min_periods=window).median().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = activity / normalizer
    tau = raw.clip(lower=0.0, upper=cap)
    fallback = normalizer.isna() | activity.isna() | (activity == 0.0)
    return tau.where(~fallback, 1.0)


def compute_op_ewma(returns: pd.DataFrame, tau: pd.DataFrame, halflife_op: float):
    """kappa = ln(2) / halflife_op.

    M_t = r_t + exp(-kappa*tau_t) * M_{t-1}
    V_t = r_t^2 + exp(-2*kappa*tau_t) * V_{t-1}
    T_t = sum(tau)  (cumulative operational time)

    Init M = V = T = 0 at each column's own PIT start (its first non-NaN
    return in `returns`). NaN return -> M, V carry over UNCHANGED (no decay
    applied that day either) and tau is NOT accumulated into T that day.

    Returns (M, V, T), each shaped/indexed like `returns`.

    Vectorized across columns at each time step: a plain per-(row,col)
    Python double loop is correct but slow when re-run many times (grid
    search, shuffle nulls), so the inner loop here is over time only, with
    all columns updated together via numpy array ops each step.
    """
    kappa = np.log(2.0) / float(halflife_op)
    idx = returns.index
    cols = returns.columns
    r = returns.to_numpy(dtype=float)
    tau_arr = tau.reindex(index=idx, columns=cols).to_numpy(dtype=float)
    n_t, n_c = r.shape

    decay1 = np.exp(-kappa * tau_arr)
    decay2 = np.exp(-2.0 * kappa * tau_arr)

    valid = ~np.isnan(r)
    has_any = valid.any(axis=0)
    pit_idx = np.where(has_any, valid.argmax(axis=0), n_t)

    M = np.full((n_t, n_c), np.nan)
    V = np.full((n_t, n_c), np.nan)
    T = np.full((n_t, n_c), np.nan)

    m_prev = np.zeros(n_c)
    v_prev = np.zeros(n_c)
    t_prev = np.zeros(n_c)

    for t in range(n_t):
        active = t >= pit_idx
        if not active.any():
            continue
        rt = r[t, :]
        upd = active & ~np.isnan(rt)

        m_new = np.where(upd, rt + decay1[t, :] * m_prev, m_prev)
        v_new = np.where(upd, rt * rt + decay2[t, :] * v_prev, v_prev)
        t_new = np.where(upd, t_prev + tau_arr[t, :], t_prev)

        m_prev = np.where(active, m_new, m_prev)
        v_prev = np.where(active, v_new, v_prev)
        t_prev = np.where(active, t_new, t_prev)

        M[t, :] = np.where(active, m_prev, np.nan)
        V[t, :] = np.where(active, v_prev, np.nan)
        T[t, :] = np.where(active, t_prev, np.nan)

    M_df = pd.DataFrame(M, index=idx, columns=cols)
    V_df = pd.DataFrame(V, index=idx, columns=cols)
    T_df = pd.DataFrame(T, index=idx, columns=cols)
    return M_df, V_df, T_df


def compute_raw_z(M: pd.DataFrame, V: pd.DataFrame, T: pd.DataFrame, halflife_op: float) -> pd.DataFrame:
    """z = M / sqrt(max(V, 1e-12)); NaN where T < 2*halflife_op (burn-in).

    Factored out of compute_signal because callers sometimes need the raw
    z directly (e.g. a liveness assertion defined on |z| itself, before any
    f(z) transform) -- duplicating the z formula in two places would risk a
    divergent copy, so compute_signal calls this helper internally.
    """
    z = M / np.sqrt(V.clip(lower=1e-12))
    burn_in = T < (2.0 * float(halflife_op))
    return z.where(~burn_in, np.nan)


def compute_signal(M: pd.DataFrame, V: pd.DataFrame, T: pd.DataFrame, halflife_op: float,
                    transform: str) -> pd.DataFrame:
    """z = M / sqrt(max(V, 1e-12)); z = NaN where T < 2*halflife_op (burn-in).
    transform in {'sign', 'tanh', 'clip2'}: sign(z), tanh(z), clip(z, -2, 2).
    Returns f(z), the transformed tradeable signal."""
    z = compute_raw_z(M, V, T, halflife_op)
    if transform == "sign":
        return np.sign(z)
    if transform == "tanh":
        return np.tanh(z)
    if transform == "clip2":
        return z.clip(lower=-2.0, upper=2.0)
    raise ValueError(f"unknown transform: {transform!r} (expected 'sign', 'tanh', or 'clip2')")


def calendar_twin_tau(returns: pd.DataFrame) -> pd.DataFrame:
    """tau == 1 everywhere returns is defined, matching returns' shape --
    the calendar twin's tau input to compute_op_ewma. Not a parallel
    implementation of the EWMA recursion; only the clock input differs.
    Passing this as `tau` to compute_op_ewma recovers plain calendar-time
    EWMA exactly."""
    return pd.DataFrame(1.0, index=returns.index, columns=returns.columns)


def variance_clock_input(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean of r^2 over `window` bars -- a realized-variance
    activity proxy, fed into compute_tau exactly like any other activity
    measure (e.g. volume)."""
    return (returns ** 2).rolling(window, min_periods=window).mean()
