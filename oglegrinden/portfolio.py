"""Cross-sectional reversal portfolio construction.

Given a cross-section of formation-period returns (5d or 10d log returns,
one per eligible ticker), build dollar-neutral, gross-100%, single-name-
capped reversal weights: buy the week's losers, sell the winners, weighted
by (winsorized, demeaned) magnitude of the move.
"""

import numpy as np
import pandas as pd


def winsorize_mad(s: pd.Series, k: float = 3.0) -> pd.Series:
    """Clip `s` to [median - k*MAD, median + k*MAD], MAD = median absolute
    deviation from the median (raw MAD, not the 1.4826-scaled
    "normal-consistent" version -- the spec just says "+/-3 MAD").
    """
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s.copy()
    lower, upper = med - k * mad, med + k * mad
    return s.clip(lower, upper)


def cap_and_renormalize(w: pd.Series, cap: float = 0.15, tol: float = 1e-9, max_iter: int = 200) -> pd.Series:
    """Iterative "water-filling" cap: rescale weights so gross exposure
    equals the original gross target while ensuring no |w_i| exceeds `cap`.
    Names that would exceed the cap are locked at the cap and the
    remaining gross budget is redistributed proportionally among the
    still-free names, repeated until stable.

    If the cap is so tight relative to the number of names that the
    original gross target is unreachable (e.g. N * cap < gross target),
    the result falls short of the gross target rather than violating the
    cap -- the cap is a hard constraint, the gross target is not.
    """
    idx = w.index
    sign = np.sign(w.values)
    mag = np.abs(w.values).astype(float)
    gross_target = mag.sum()
    fixed = np.zeros(len(w), dtype=bool)

    for _ in range(max_iter):
        free = ~fixed
        free_sum = mag[free].sum()
        remaining_gross = gross_target - mag[fixed].sum()
        if free_sum <= 0 or remaining_gross <= 0:
            mag[free] = 0.0
            break
        scale = remaining_gross / free_sum
        new_mag_free = mag[free] * scale
        newly_over = new_mag_free > cap + tol
        if not newly_over.any():
            mag[free] = new_mag_free
            break
        free_positions = np.where(free)[0]
        over_positions = free_positions[newly_over]
        fixed[over_positions] = True
        mag[over_positions] = cap
    else:
        mag = np.minimum(mag, cap)

    return pd.Series(sign * mag, index=idx)


def formation_weights(formation_returns: pd.Series, cap: float = 0.15, winsor_k: float = 3.0) -> pd.Series:
    """Cross-sectional reversal weights from one week's formation returns.

    f_i = winsorized formation return
    s_i = -(f_i - cross-sectional mean(f))   [demeaned reversal signal]
    w_i = s_i / sum(|s_i|)                   [100% gross]
    then capped at +/-`cap` and renormalized (see `cap_and_renormalize`).

    Because s_i is built from a demeaned series, sum(s_i) == 0 by
    construction and the raw weights are already dollar-neutral; capping
    can perturb this only slightly (it does not target neutrality
    directly), which is why the spec calls Sigma w_i == 0 a consequence
    ("=>") rather than a constraint enforced downstream.
    """
    f = formation_returns.dropna()
    if f.empty:
        return pd.Series(dtype=float)
    if len(f) == 1:
        return pd.Series([0.0], index=f.index)

    f_w = winsorize_mad(f, k=winsor_k)
    s = -(f_w - f_w.mean())
    abs_sum = s.abs().sum()
    if abs_sum == 0:
        return pd.Series(0.0, index=f.index)
    w = s / abs_sum
    return cap_and_renormalize(w, cap=cap)
