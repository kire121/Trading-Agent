"""Steg 2 redundancy-screen control battery (spec Sec.10): {13w realized
vol; 26w mean pairwise correlation; 13w skew; 26w |r| autocorr(1);
26w-rolling absorption ratio (PC1 share); strength u=|12m|/vol_ann}.

No existing implementation of this exact battery was found in the repo
(each strategy's own battery.py -- Smittotalet/Dammluckan/Omori -- reruns
ITS OWN full backtest pipeline on synthetic panels, a different concept
from a cross-sectional control regression; see docs/INSTRUKTION.md sec.7,
"projektspecifika battery.py/robustness.py-nullbatterier ... INTE
migrerade"). Built fresh per the spec's exact list.
"""
import numpy as np
import pandas as pd

from . import config
from . import scheduling


def _weekly_returns(panel) -> pd.DataFrame:
    daily_ret = panel.simple_returns()
    weekly_close = scheduling.week_end_values(panel.adjusted_close)
    return weekly_close.pct_change()


def realized_vol_13w(weekly_ret: pd.DataFrame) -> pd.DataFrame:
    return weekly_ret.rolling(13).std()


def mean_pairwise_corr_26w(weekly_ret: pd.DataFrame) -> pd.Series:
    """Market-wide (not per-instrument): rolling 26w average of the
    off-diagonal pairwise correlation matrix among all instruments."""
    def _avg_offdiag(window: pd.DataFrame) -> float:
        c = window.corr().to_numpy()
        n = c.shape[0]
        if n < 2:
            return np.nan
        off = c[~np.eye(n, dtype=bool)]
        return float(np.nanmean(off))

    out = pd.Series(np.nan, index=weekly_ret.index)
    for i in range(25, len(weekly_ret)):
        out.iloc[i] = _avg_offdiag(weekly_ret.iloc[i - 25: i + 1])
    return out


def skew_13w(weekly_ret: pd.DataFrame) -> pd.DataFrame:
    return weekly_ret.rolling(13).skew()


def abs_return_autocorr1_26w(weekly_ret: pd.DataFrame) -> pd.DataFrame:
    absr = weekly_ret.abs()

    def _roll_autocorr(s: pd.Series) -> pd.Series:
        return s.rolling(26).apply(lambda w: pd.Series(w).autocorr(lag=1), raw=False)

    return absr.apply(_roll_autocorr)


def absorption_ratio_26w(weekly_ret: pd.DataFrame) -> pd.Series:
    """Market-wide: rolling 26w PC1 share of total variance across the
    instrument panel (same PCA-share concept as K4.1, but time-varying/
    rolling and computed on returns rather than the tilt-X panel)."""
    out = pd.Series(np.nan, index=weekly_ret.index)
    for i in range(25, len(weekly_ret)):
        window = weekly_ret.iloc[i - 25: i + 1].dropna(axis=1, how="any")
        if window.shape[1] < 2:
            continue
        cov = np.cov(window.to_numpy(), rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        total = eigvals.sum()
        out.iloc[i] = float(eigvals.max() / total) if total > 0 else np.nan
    return out


def build_battery(panel, u_weekly: pd.DataFrame) -> dict:
    """Returns {name: weekly panel/series} for all six controls, aligned to
    the weekly (Friday) index."""
    weekly_ret = _weekly_returns(panel)
    return {
        "vol13w": realized_vol_13w(weekly_ret),
        "corr26w": mean_pairwise_corr_26w(weekly_ret),
        "skew13w": skew_13w(weekly_ret),
        "abs_autocorr1_26w": abs_return_autocorr1_26w(weekly_ret),
        "absorption26w": absorption_ratio_26w(weekly_ret),
        "strength_u": u_weekly,
    }
