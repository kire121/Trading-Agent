"""
Dammluckan -- record-hazard signal.

M_t   = rolling max/min of RAW close over [t-n, t-1]  (causal: excludes t)
E+_t  = 1{P_t > M_t}          (upside record)
E-_t  = 1{P_t < m_t}          (downside record)
O+_t  = share of days in [t-n, t-1] with P_s >= M_t * (1 - c * band_vol_t * sqrt(5))
O-_t  = share of days in [t-n, t-1] with P_s <= m_t * (1 + c * band_vol_t * sqrt(5))

band_vol_t is the asset's own trailing realized vol (config.VOL_LOOKBACK,
daily log returns), evaluated strictly through t-1 (shifted by one day
relative to data.Panel.band_vol) so the occupation band at decision time t
never uses day t's own return -- avoiding a same-day feedback loop between
the record event (which does use P_t) and the band width.

c is calibrated once (per grid window length n) on the primary universe's
IS-only block-bootstrap null so that the pooled null median of O+ (and, by
the mirrored construction, O-) is approximately 0.15 -- see
calibrate_band_constant(). theta_i is calibrated per-asset, per-n, as the
80th (grid: 70/80/90th) percentile of that asset's own block-bootstrap null
distribution of O+ -- see calibrate_theta().
"""
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config
from . import nulls


def rolling_max_excl_today(price: pd.DataFrame, n: int) -> pd.DataFrame:
    """M_t = max(P_s for s in [t-n, t-1])."""
    return price.shift(1).rolling(n, min_periods=n).max()


def rolling_min_excl_today(price: pd.DataFrame, n: int) -> pd.DataFrame:
    return price.shift(1).rolling(n, min_periods=n).min()


def record_events_high(price: pd.DataFrame, m_t: pd.DataFrame) -> pd.DataFrame:
    valid = m_t.notna() & price.notna()
    return (price > m_t).astype(float).where(valid)


def record_events_low(price: pd.DataFrame, m_t: pd.DataFrame) -> pd.DataFrame:
    valid = m_t.notna() & price.notna()
    return (price < m_t).astype(float).where(valid)


def _occupation_high_1d(prices: np.ndarray, band_vol_causal: np.ndarray, n: int, c: float) -> np.ndarray:
    """O+_t for one ticker's raw close series (see module docstring)."""
    T = len(prices)
    out = np.full(T, np.nan)
    if T <= n:
        return out
    shifted = prices[:-1]
    windows = sliding_window_view(shifted, n)          # windows[k] = prices[k:k+n], k=0..T-n-1 -> t=k+n
    valid_count = np.sum(~np.isnan(windows), axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        m_t = np.nanmax(windows, axis=1)
    band_vol_t = band_vol_causal[n:]
    thresh = m_t * (1.0 - c * band_vol_t * np.sqrt(config.BAND_HORIZON_DAYS))
    occ = np.sum(windows >= thresh[:, None], axis=1) / n
    occ = np.where((valid_count == n) & np.isfinite(thresh), occ, np.nan)
    out[n:] = occ
    return out


def _occupation_low_1d(prices: np.ndarray, band_vol_causal: np.ndarray, n: int, c: float) -> np.ndarray:
    T = len(prices)
    out = np.full(T, np.nan)
    if T <= n:
        return out
    shifted = prices[:-1]
    windows = sliding_window_view(shifted, n)
    valid_count = np.sum(~np.isnan(windows), axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        m_t = np.nanmin(windows, axis=1)
    band_vol_t = band_vol_causal[n:]
    thresh = m_t * (1.0 + c * band_vol_t * np.sqrt(config.BAND_HORIZON_DAYS))
    occ = np.sum(windows <= thresh[:, None], axis=1) / n
    occ = np.where((valid_count == n) & np.isfinite(thresh), occ, np.nan)
    out[n:] = occ
    return out


def occupation_high(price: pd.DataFrame, band_vol_causal: pd.DataFrame, n: int, c: float) -> pd.DataFrame:
    out = {}
    for col in price.columns:
        out[col] = _occupation_high_1d(price[col].to_numpy(), band_vol_causal[col].to_numpy(), n, c)
    return pd.DataFrame(out, index=price.index)


def occupation_low(price: pd.DataFrame, band_vol_causal: pd.DataFrame, n: int, c: float) -> pd.DataFrame:
    out = {}
    for col in price.columns:
        out[col] = _occupation_low_1d(price[col].to_numpy(), band_vol_causal[col].to_numpy(), n, c)
    return pd.DataFrame(out, index=price.index)


class SignalBundle:
    """All signal artifacts for one (n, c) configuration, for a given panel."""

    __slots__ = ("n", "c", "m_high", "m_low", "e_high", "e_low", "o_high", "o_low", "band_vol_causal")

    def __init__(self, panel, n: int, c: float):
        self.n = n
        self.c = c
        self.band_vol_causal = panel.band_vol.shift(1)
        self.m_high = rolling_max_excl_today(panel.raw_close, n)
        self.m_low = rolling_min_excl_today(panel.raw_close, n)
        self.e_high = record_events_high(panel.raw_close, self.m_high)
        self.e_low = record_events_low(panel.raw_close, self.m_low)
        self.o_high = occupation_high(panel.raw_close, self.band_vol_causal, n, c)
        self.o_low = occupation_low(panel.raw_close, self.band_vol_causal, n, c)


def build_signal(panel, n: int, c: float) -> SignalBundle:
    return SignalBundle(panel, n, c)


# ---------------------------------------------------------------------------
# Calibration: band constant c (per n) and per-asset theta_i (per n, per pctl)
# ---------------------------------------------------------------------------

def calibrate_band_constant(panel, n: int, is_start=config.IS_START, is_end=config.IS_END,
                             target_median=config.C_BAND_TARGET_MEDIAN,
                             n_draws=200, seed=0) -> float:
    """
    Bisection search for c such that the pooled (across all primary-universe
    assets, across all bootstrap draws) null median of O+ over the IS window
    is approximately target_median. IS-only, frozen for OOS use.
    """
    is_panel = panel.slice(is_start, is_end)
    lo, hi = 1e-4, 20.0

    def median_occ_for_c(c):
        pooled = nulls.pooled_null_occupation(is_panel, n=n, c=c, side="high", n_draws=n_draws, seed=seed)
        return np.nanmedian(pooled)

    med_lo, med_hi = median_occ_for_c(lo), median_occ_for_c(hi)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        med_mid = median_occ_for_c(mid)
        if med_mid > target_median:
            hi, med_hi = mid, med_mid
        else:
            lo, med_lo = mid, med_mid
        if abs(med_mid - target_median) < 1e-4:
            break
    return 0.5 * (lo + hi)


def calibrate_theta_multi(panel, n: int, c: float, pctls, side="high",
                           is_start=config.IS_START, is_end=config.IS_END,
                           n_draws=config.N_THETA_CALIB_DRAWS, seed=0) -> dict:
    """
    theta_i = pctl-th percentile of asset i's own block-bootstrap null
    distribution of O+ (side='high') / O- (side='low'), estimated IS-only,
    frozen for OOS use. Generates the null ONCE per asset and reads off every
    requested percentile from it (avoids redundant bootstrap draws across the
    theta_pctl grid).

    Returns {pctl: pd.Series indexed by ticker}.
    """
    is_panel = panel.slice(is_start, is_end)
    per_pctl = {p: {} for p in pctls}
    for i, ticker in enumerate(panel.tickers):
        null_vals = nulls.per_asset_null_occupation(
            is_panel, ticker=ticker, n=n, c=c, side=side, n_draws=n_draws, seed=seed + i
        )
        for p in pctls:
            per_pctl[p][ticker] = np.nanpercentile(null_vals, p) if len(null_vals) else np.nan
    return {p: pd.Series(vals) for p, vals in per_pctl.items()}
