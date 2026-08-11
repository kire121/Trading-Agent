"""Null-baseline twins, quantile-mapped to the primary's own unconditional
G distribution -- "Dammluckans matchade-bredd-krav oversatt till matchad
exponering: bara timingen far skilja."

T1 -- voltarget twin: (sigma*/sigma_hat_t)^kappa. Amplitude-based exposure
     control, the mechanism the brief argues Smittotalet beats structurally.
T2 -- count-EWMA twin: raw smoothed event-count level, no ratio/branching
     structure ("den brutala" -- kills R_hat if it's just disguised
     count-EWMA).
T3 -- broadcast twin (a la Runraden's T4): identical events->Cori pipeline,
     but run on the sleeve's own aggregated |return| as a single synthetic
     series instead of pooling per-asset event indicators. Tests whether
     cross-asset pooling (breadth) adds anything over one aggregate signal.
"""
import numpy as np
import pandas as pd

from . import config
from . import events as events_mod
from . import signal as signal_mod


def quantile_map_to(reference: pd.Series, raw: pd.Series) -> pd.Series:
    """Map `raw`'s time-ordering onto `reference`'s empirical value
    distribution: same marginal CDF as `reference`, only the timing comes
    from `raw`. This is the "matched exposure, timing-only" null construction.
    """
    common = pd.concat([reference, raw], axis=1, keys=["ref", "raw"]).dropna()
    ref_sorted = np.sort(common["ref"].to_numpy())
    n = len(ref_sorted)
    if n < 2:
        return pd.Series(np.nan, index=raw.index)
    rank = common["raw"].rank(method="average").to_numpy()  # 1..n, no pct scaling
    positions = rank - 1.0                                   # exact 0..n-1 when no ties
    mapped = np.interp(positions, np.arange(n), ref_sorted)
    return pd.Series(mapped, index=common.index).reindex(raw.index)


def voltarget_twin(base_returns: pd.Series, g_primary: pd.Series, kappa: float,
                    vol_lookback: int = config.TSMOM_VOL_LOOKBACK) -> pd.Series:
    """T1: (sigma*/sigma_hat_t)^kappa, sigma* = the base book's own
    unconditional (full-sample) realized vol, sigma_hat_t = rolling realized
    vol of the same book. Quantile-mapped onto g_primary's distribution."""
    sigma_hat = base_returns.rolling(vol_lookback).std() * np.sqrt(config.TRADING_DAYS_YEAR)
    sigma_star = base_returns.std() * np.sqrt(config.TRADING_DAYS_YEAR)
    raw = (sigma_star / sigma_hat) ** kappa
    return quantile_map_to(g_primary, raw)


def count_ewma_twin(x_t: pd.Series, g_primary: pd.Series, tau: int, kappa: float) -> pd.Series:
    """T2: raw smoothed event-count level, no branching-ratio structure."""
    ewma = x_t.fillna(0.0).ewm(span=tau, min_periods=tau).mean()
    baseline = ewma.mean()
    raw = (baseline / ewma) ** kappa
    return quantile_map_to(g_primary, raw)


def broadcast_aggregate_series(returns: pd.DataFrame) -> pd.Series:
    """Equal-weighted mean |return| across the universe -- the sleeve's own
    aggregated-return proxy, a la Runraden's T4 build_market_basket."""
    return returns.abs().mean(axis=1, skipna=True)


def broadcast_twin(returns: pd.DataFrame, g_primary: pd.Series, q: int, tau: int, kappa: float):
    """T3: identical events->Lambda->R_hat->G pipeline, run on a single
    aggregated |return| series instead of per-asset pooled events."""
    agg = broadcast_aggregate_series(returns)
    thresh = agg.shift(1).rolling(config.EVENT_LOOKBACK, min_periods=config.EVENT_LOOKBACK).quantile(q / 100.0)
    x_agg = (agg > thresh).astype(float)
    x_agg[thresh.isna()] = np.nan
    r_hat, g_raw, _ = signal_mod.build(x_agg, q=q, tau=tau, kappa=kappa)
    return quantile_map_to(g_primary, g_raw)
