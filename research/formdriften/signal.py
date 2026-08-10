"""
Core Formdriften signal: the Wasserstein tangent-space shape residual.

For a 1-D distribution, W2(P_prev, P_curr)^2 = integral_0^1 (Q_curr(u) - Q_prev(u))^2 du,
closed form via quantile functions -- no OT solver needed. We project the
quantile displacement delta(u) = Q_curr(u) - Q_prev(u) onto span{1, Q_prev(u)}
in L^2(0,1) via OLS over a discrete grid: this affine part captures all
location (mean shift) and scale (vol change) movement. The residual s(u) is,
by construction, the part of the Wasserstein tangent vector orthogonal to
location-scale -- pure shape drift (tails, skew, bimodality).

D_L = mean(s(u)) for u in (0.02, 0.20]: a negative D_L means the left tail
has widened beyond what the volatility change alone explains.
"""
import numpy as np

U_GRID = np.round(np.arange(0.02, 0.981, 0.01), 4)
LEFT_TAIL_MASK = (U_GRID > 0.02) & (U_GRID <= 0.20)


def quantile_function(sample, u_grid=U_GRID):
    return np.quantile(np.asarray(sample), u_grid, method="linear")


def w2_squared_closed_form(prev_sample, curr_sample, u_grid=U_GRID):
    """Closed-form 1-D squared Wasserstein-2 distance via quantile functions.

    Uses trapezoidal integration over u_grid in [0, 1] so it is directly
    comparable to a full-support quantile-function OT distance (POT's
    wasserstein_1d), independent of the strategy's restricted u_grid choice.
    """
    full_grid = np.linspace(1e-4, 1 - 1e-4, 999)
    qp = np.quantile(prev_sample, full_grid)
    qc = np.quantile(curr_sample, full_grid)
    return np.trapezoid((qc - qp) ** 2, full_grid)


def shape_decompose(prev_sample, curr_sample, u_grid=U_GRID):
    """OLS-project Q_curr(u) on {1, Q_prev(u)} over u_grid; return residual s(u).

    Returns
    -------
    resid : ndarray, shape (len(u_grid),)
        s(u), the location-scale-orthogonal shape residual.
    beta : ndarray, shape (2,)
        [intercept, slope] of the affine location-scale fit (slope ~ ratio of
        curr/prev dispersion; intercept ~ mean shift net of the slope term).
    qp, qc : ndarray
        The two quantile functions actually used (for diagnostics).
    """
    qp = np.quantile(prev_sample, u_grid)
    qc = np.quantile(curr_sample, u_grid)
    X = np.column_stack([np.ones_like(u_grid), qp])
    beta, *_ = np.linalg.lstsq(X, qc, rcond=None)
    resid = qc - X @ beta
    return resid, beta, qp, qc


def d_l_signal(prev_returns, curr_returns, u_grid=U_GRID, tail_mask=LEFT_TAIL_MASK):
    """Primary signal: mean shape residual over the left-tail band u in (0.02, 0.20]."""
    resid, beta, qp, qc = shape_decompose(prev_returns, curr_returns, u_grid)
    d_l = resid[tail_mask].mean()
    return d_l, resid, beta


def rolling_d_l(returns, curr_window=126, prev_window=126, u_grid=U_GRID, tail_mask=LEFT_TAIL_MASK,
                 at_positions=None):
    """D_L at index t where a full disjoint curr/prev pair is available.

    returns: 1-D array of daily returns, chronologically ordered.
    curr = returns[t-curr_window : t]   (uses data through t-1)
    prev = returns[t-curr_window-prev_window : t-curr_window]  (disjoint, older)

    at_positions: optional iterable of integer positions to evaluate at
        (e.g. month-end rebalance dates) -- avoids the O(n) full-daily loop
        when only a sparse set of dates is actually needed downstream.

    Returns an array the same length as `returns` (or as `at_positions`),
    NaN where undefined.
    """
    r = np.asarray(returns)
    n = len(r)
    min_t = curr_window + prev_window
    positions = range(min_t, n) if at_positions is None else at_positions
    positions = list(positions)
    out = np.full(len(positions), np.nan)
    for i, t in enumerate(positions):
        if t < min_t or t >= n:
            continue
        curr = r[t - curr_window : t]
        prev = r[t - curr_window - prev_window : t - curr_window]
        try:
            d_l, _, _ = d_l_signal(prev, curr, u_grid, tail_mask)
        except Exception:
            d_l = np.nan
        out[i] = d_l
    if at_positions is None:
        full = np.full(n, np.nan)
        full[min_t:n] = out
        return full
    return out


def build_d_l_panel(returns_panel, eval_dates, curr_window=126, prev_window=126,
                     u_grid=U_GRID, tail_mask=LEFT_TAIL_MASK):
    """D_L for every column of `returns_panel`, evaluated only at `eval_dates`
    (e.g. monthly rebalance dates) -- the sparse-evaluation counterpart to
    rolling_d_l, used throughout the backtest/robustness/null-hypothesis code
    so we never pay for a daily-resolution signal we don't need.

    Returns a DataFrame indexed by eval_dates (dates not present in
    returns_panel.index are dropped), columns = returns_panel.columns.
    """
    import pandas as pd  # local import to keep this module numpy-only otherwise

    idx = returns_panel.index
    eval_dates = [d for d in eval_dates if d in idx]
    positions = [idx.get_loc(d) for d in eval_dates]
    out = {}
    for col in returns_panel.columns:
        r = returns_panel[col].values
        out[col] = rolling_d_l(r, curr_window, prev_window, u_grid, tail_mask, at_positions=positions)
    return pd.DataFrame(out, index=pd.DatetimeIndex(eval_dates))
