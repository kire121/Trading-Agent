"""The three "active twins" the primary signal must beat.

Brief: "Aktiva tvillingar: ren momentum; momentum x turnover-niva
(Lee-Swaminathan); lag-1-korskorrelation u->r." Each twin reuses exactly
the same window, universe, portfolio construction (quintile, inverse-vol
sizing, caps), and backtest mechanics as the primary strategy -- only the
raw cross-sectional statistic changes -- so a comparison isolates whether
the Levy-area ROTATION construction adds anything beyond signals every
other strategy in this repository's "graveyard" already knows about.

Twin 2's exact construction is a DECLARED reading: Lee & Swaminathan (2000,
"Price Momentum and Trading Volume") sort independently on momentum and on
*share* turnover (volume / shares outstanding); we don't have
shares-outstanding data, so "turnover level" is proxied by trailing dollar
ADV (log-scaled, then cross-sectionally z-scored) -- a liquidity-level
proxy, not literal share turnover. The twin signal is the plain product of
the two z-scores, the standard "characteristic x characteristic"
interaction construction, deliberately kept simple to match the brief's
own framing of the twins as "kanda och ointressanta" (known and
uninteresting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from vridmomentet.config import TwinParams
from vridmomentet.data import Panel
from vridmomentet.signal import cross_sectional_zscore, formation_return, signed_dollar_volume


def momentum_twin(panel: Panel, params: TwinParams) -> pd.DataFrame:
    """Twin 1: ren momentum -- plain cross-sectional formation return."""
    r_n = formation_return(panel.log_returns, params.momentum_lookback_days)
    return cross_sectional_zscore(r_n, 0.01, 0.99)


def momentum_x_turnover_twin(panel: Panel, params: TwinParams) -> pd.DataFrame:
    """Twin 2: momentum x turnover-level (Lee & Swaminathan 2000)."""
    r_n = formation_return(panel.log_returns, params.momentum_lookback_days)
    z_mom = cross_sectional_zscore(r_n, 0.01, 0.99)

    turnover_level = np.log(panel.adv60.rolling(params.turnover_lookback_days, min_periods=45).mean())
    z_turnover = cross_sectional_zscore(turnover_level, 0.01, 0.99)

    return z_mom * z_turnover


def _rolling_lag1_corr_1d(r: np.ndarray, u: np.ndarray, window: int) -> np.ndarray:
    """corr(u_{s-1}, r_s) over a trailing `window`-day span, causal."""
    n = len(r)
    out = np.full(n, np.nan)
    if n < window + 1:
        return out
    r_windows = sliding_window_view(r, window)[1:]        # r_s for s in the window, aligned...
    u_lag_windows = sliding_window_view(u, window)[:-1]    # ...to u_{s-1} one day earlier
    valid = ~np.isnan(r_windows).any(axis=1) & ~np.isnan(u_lag_windows).any(axis=1)

    r_mean = r_windows.mean(axis=1, keepdims=True)
    u_mean = u_lag_windows.mean(axis=1, keepdims=True)
    r_c = r_windows - r_mean
    u_c = u_lag_windows - u_mean
    cov = (r_c * u_c).sum(axis=1)
    r_std = np.sqrt((r_c ** 2).sum(axis=1))
    u_std = np.sqrt((u_c ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (r_std * u_std)
    corr = np.where(valid & (r_std > 0) & (u_std > 0), corr, np.nan)
    out[window:] = corr
    return out


def lag1_cross_correlation_twin(panel: Panel, params: TwinParams) -> pd.DataFrame:
    """Twin 3: lag-1-korskorrelation u->r -- a purely linear, contemporary-
    correlation statistic (the symmetric-in-spirit competitor the brief's
    hypothesis paragraph contrasts the antisymmetric Levy area against).
    """
    u = signed_dollar_volume(panel)
    r_vals = panel.log_returns.values
    u_vals = u.values
    cols = {}
    for j, col in enumerate(panel.log_returns.columns):
        cols[col] = _rolling_lag1_corr_1d(r_vals[:, j], u_vals[:, j], params.lag1_lookback_days)
    raw = pd.DataFrame(cols, index=panel.log_returns.index)
    return cross_sectional_zscore(raw, 0.01, 0.99)


TWINS = {
    "momentum": momentum_twin,
    "momentum_x_turnover": momentum_x_turnover_twin,
    "lag1_signed_volume_corr": lag1_cross_correlation_twin,
}


def compute_all_twins(panel: Panel, params: TwinParams = TwinParams()) -> dict[str, pd.DataFrame]:
    return {name: fn(panel, params) for name, fn in TWINS.items()}
