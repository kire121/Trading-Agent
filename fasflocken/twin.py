"""
Fasflocken (PH-1) -- "the boring twin".

Identical pipeline to signals.py up to the bandpass step, but the
amplitude-free Kuramoto order parameter is replaced by the classic,
amplitude-weighted statistic: rolling mean pairwise Pearson correlation
across the same sector constituents, on the same bandpassed series, over
the same window.

The whole point of PH-1 is that phase coherence should beat this twin
out-of-sample; if it doesn't (Delta-Sharpe < 0.15 net, per the spec's
rejection criteria), R_s is "just" a monotone transform of correlation and
the phase machinery is decoration. See stats.py for the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_pairwise_correlation(x: pd.DataFrame, window: int) -> pd.Series:
    """Rolling mean pairwise Pearson correlation across columns of `x`.

    Uses pandas' built-in incremental pairwise rolling correlation
    (`DataFrame.rolling(window).corr()`), which is the efficient
    equivalent of computing the NxN correlation matrix at every t. A
    column only contributes at time t if it has a full, gap-free `window`
    of history there (min_periods=window) -- the same "fully valid window
    or nothing" convention rolling_analytic_phase uses for the Hilbert
    transform, so the two measures are compared on equal footing.
    """
    n = x.shape[1]
    if n < 2:
        return pd.Series(np.nan, index=x.index, name="mean_pairwise_corr")

    corr_panel = x.rolling(window=window, min_periods=window).corr()

    def _off_diagonal_mean(group: pd.DataFrame) -> float:
        vals = group.to_numpy(dtype=float)
        total = np.nansum(vals)
        diag = np.nansum(np.diag(vals))
        valid_total = np.sum(~np.isnan(vals))
        valid_diag = np.sum(~np.isnan(np.diag(vals)))
        valid_off = valid_total - valid_diag
        if valid_off <= 0:
            return np.nan
        return (total - diag) / valid_off

    result = corr_panel.groupby(level=0).apply(_off_diagonal_mean)
    result.name = "mean_pairwise_corr"
    result.index.name = x.index.name
    return result
