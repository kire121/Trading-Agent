"""Three independent time-irreversibility estimators.

1. hvg_irreversibility  -- directed horizontal visibility graph, KL(P_out || P_in)
2. ordinal_irreversibility -- Bandt-Pompe length-3 patterns vs their time-reversals
3. psi_irreversibility -- third-moment asymmetry statistic psi(tau)

All return a non-negative scalar for (1) and (2) (true KL divergences, 0 for a
perfectly time-symmetric process) and a signed, variance-normalized scalar for
(3) (its absolute value is the analogous "amount of irreversibility").
"""

import itertools

import numpy as np
import ts2vg

from . import config


def _kl_from_counts(counts_p, counts_q, smoothing=config.KL_SMOOTHING):
    """KL(P || Q) over a shared support, with add-`smoothing` pseudocounts."""
    counts_p = np.asarray(counts_p, dtype=float)
    counts_q = np.asarray(counts_q, dtype=float)
    p = counts_p + smoothing
    q = counts_q + smoothing
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def hvg_irreversibility(window):
    """KL(P_out || P_in) of a directed (left-to-right) horizontal visibility graph.

    P_out: distribution of out-degrees (edges pointing forward in time).
    P_in:  distribution of in-degrees (edges pointing backward in time).
    Zero for a time-reversible series (Lacasa et al. 2012).
    """
    window = np.array(window, dtype=float, copy=True)
    n = len(window)
    if n < 4:
        return np.nan
    g = ts2vg.HorizontalVG(directed="left_to_right")
    g.build(window)
    deg_out = g.degrees_out
    deg_in = g.degrees_in
    max_deg = int(max(deg_out.max(), deg_in.max()))
    bins = np.arange(0, max_deg + 2)  # bin edges 0..max_deg+1 -> max_deg+1 bins
    counts_out, _ = np.histogram(deg_out, bins=bins)
    counts_in, _ = np.histogram(deg_in, bins=bins)
    return _kl_from_counts(counts_out, counts_in)


# --- Ordinal (Bandt-Pompe) irreversibility -----------------------------

_PERMS3 = list(itertools.permutations(range(3)))
_PERM_INDEX = {p: i for i, p in enumerate(_PERMS3)}
_REVERSE_INDEX = np.array([_PERM_INDEX[p[::-1]] for p in _PERMS3])


def _ordinal_pattern_indices(window, m=3):
    """Bandt-Pompe pattern index (0..m!-1) for every overlapping length-m
    sub-sequence of `window`, using strict argsort (ties broken by position,
    i.e. earlier-equal-value counted as smaller -- negligible for continuous
    returns).
    """
    n = len(window)
    n_patterns = n - m + 1
    if n_patterns <= 0:
        return np.array([], dtype=int)
    # Build an (n_patterns, m) matrix of sliding windows without python loops.
    idx = np.arange(m) + np.arange(n_patterns)[:, None]
    sub = np.asarray(window)[idx]
    ranks = np.argsort(sub, axis=1, kind="stable")
    pattern_ids = np.zeros(n_patterns, dtype=int)
    for i, p in enumerate(_PERMS3):
        pattern_ids[np.all(ranks == np.array(p), axis=1)] = i
    return pattern_ids


def ordinal_irreversibility(window, m=config.ORDINAL_PATTERN_LENGTH):
    """KL(P(pi) || P(reverse(pi))) over length-m ordinal patterns.

    P(reverse(pi)) is the *same* empirical distribution re-indexed by the
    time-reversal map pi -> reverse(pi); this is the standard ordinal-pattern
    time-irreversibility statistic (c.f. Zanin et al. 2018, Martinez et al.).
    Zero iff the empirical pattern distribution is symmetric under time
    reversal.
    """
    if m != 3:
        raise NotImplementedError("only length-3 patterns are implemented")
    ids = _ordinal_pattern_indices(window, m=m)
    if len(ids) < 8:
        return np.nan
    counts = np.bincount(ids, minlength=len(_PERMS3))
    counts_rev = counts[_REVERSE_INDEX]
    return _kl_from_counts(counts, counts_rev)


# --- Third-moment (psi) irreversibility --------------------------------

def psi_irreversibility(window, tau=config.PSI_TAU):
    """psi(tau) = E[r_t^2 r_{t-tau}] - E[r_t r_{t-tau}^2], normalized by
    sigma^3 so it is comparable in magnitude to the KL-based estimators and
    across instruments/regimes with different volatility.
    """
    r = np.asarray(window, dtype=float)
    if len(r) <= tau + 1:
        return np.nan
    r_t = r[tau:]
    r_lag = r[:-tau]
    psi = np.mean(r_t ** 2 * r_lag) - np.mean(r_t * r_lag ** 2)
    sigma = r.std(ddof=0)
    if sigma <= 0:
        return np.nan
    return float(psi / sigma ** 3)


# --- Rolling application -------------------------------------------------

ESTIMATORS = {
    "hvg": hvg_irreversibility,
    "ordinal": ordinal_irreversibility,
    "psi": psi_irreversibility,
}


def rolling_estimator(returns, W, func, anchor_positions=None):
    """Apply `func` to trailing windows of length W ending at each anchor
    position (integer index into `returns`). If anchor_positions is None,
    use every valid position (W-1, ..., len-1) -- expensive; prefer passing
    weekly anchor positions from the pipeline.

    Returns a numpy array aligned to anchor_positions, NaN where a full
    window of *finite* (already-tradeable) data is unavailable.
    """
    values = np.array(returns, dtype=float, copy=True)
    n = len(values)
    if anchor_positions is None:
        anchor_positions = np.arange(W - 1, n)
    out = np.full(len(anchor_positions), np.nan)
    for i, pos in enumerate(anchor_positions):
        start = pos - W + 1
        if start < 0:
            continue
        window = values[start:pos + 1]
        if np.any(~np.isfinite(window)):
            continue
        out[i] = func(window)
    return out
