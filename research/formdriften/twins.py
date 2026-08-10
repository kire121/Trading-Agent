"""
The four pre-registered "twin" factors D_L must beat with margin: standard
moment-estimator competitors constructed on the exact same curr/prev windows
as D_L, so the redundancy screen is apples-to-apples.
"""
import numpy as np
from scipy import stats

TWIN_NAMES = ["d_skew", "d_kurt", "d_qasym", "d_vol"]


def _twin_values(prev, curr):
    d_skew = stats.skew(curr) - stats.skew(prev)
    d_kurt = stats.kurtosis(curr) - stats.kurtosis(prev)  # excess kurtosis (Fisher)
    q05c, q50c, q95c = np.quantile(curr, [0.05, 0.5, 0.95])
    q05p, q50p, q95p = np.quantile(prev, [0.05, 0.5, 0.95])
    asym_c = (q95c - q50c) - (q50c - q05c)
    asym_p = (q95p - q50p) - (q50p - q05p)
    d_qasym = asym_c - asym_p
    d_vol = np.std(curr, ddof=1) - np.std(prev, ddof=1)
    return d_skew, d_kurt, d_qasym, d_vol


def rolling_twins(returns, curr_window=126, prev_window=126, at_positions=None):
    """Same disjoint curr/prev construction as signal.rolling_d_l, for the four twins."""
    r = np.asarray(returns)
    n = len(r)
    min_t = curr_window + prev_window
    positions = range(min_t, n) if at_positions is None else at_positions
    positions = list(positions)
    out = {name: np.full(len(positions), np.nan) for name in TWIN_NAMES}
    for i, t in enumerate(positions):
        if t < min_t or t >= n:
            continue
        curr = r[t - curr_window : t]
        prev = r[t - curr_window - prev_window : t - curr_window]
        try:
            vals = _twin_values(prev, curr)
        except Exception:
            vals = (np.nan,) * 4
        for name, v in zip(TWIN_NAMES, vals):
            out[name][i] = v
    return out


def build_twin_panels(returns_panel, eval_dates, curr_window=126, prev_window=126):
    """dict[twin_name] -> DataFrame[date, ticker], mirroring signal.build_d_l_panel."""
    import pandas as pd

    idx = returns_panel.index
    eval_dates = [d for d in eval_dates if d in idx]
    positions = [idx.get_loc(d) for d in eval_dates]
    panels = {name: {} for name in TWIN_NAMES}
    for col in returns_panel.columns:
        r = returns_panel[col].values
        vals = rolling_twins(r, curr_window, prev_window, at_positions=positions)
        for name in TWIN_NAMES:
            panels[name][col] = vals[name]
    return {name: pd.DataFrame(panels[name], index=pd.DatetimeIndex(eval_dates)) for name in TWIN_NAMES}
