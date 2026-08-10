"""
Diversification requirements: |beta| < 0.15 vs SPY, |corr| < 0.25 vs a TSMOM
proxy. There is no existing "TSMOM proxy infra" in this repo to reuse (the
hypothesis's mention of one does not correspond to anything actually present
here -- see README), so this module builds a small, self-contained,
documented stand-in: classic Moskowitz-Ooi-Pedersen-style time-series
momentum (sign of trailing 12-month return, held long/short, equal-weighted
across a liquid multi-asset-class ETF basket) as a proxy for "a documented
trend-following strategy at a different horizon."
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import SPEC

TSMOM_PROXY_BASKET = ["SPY", "TLT", "GLD", "DBC", "UUP"]
TSMOM_LOOKBACK_DAYS = 252


def align_returns(a: pd.Series, b: pd.Series) -> tuple:
    idx = a.dropna().index.intersection(b.dropna().index)
    return a.loc[idx], b.loc[idx]


def beta_to_market(strategy_returns: pd.Series, market_returns: pd.Series) -> dict:
    s, m = align_returns(strategy_returns, market_returns)
    if len(s) < 20 or m.var() == 0:
        return {"beta": np.nan, "n": len(s), "passed": False}
    beta = float(np.cov(s.values, m.values, ddof=1)[0, 1] / np.var(m.values, ddof=1))
    return {"beta": beta, "n": int(len(s)), "passed": bool(abs(beta) < SPEC.diversification.max_abs_beta_spy)}


def tsmom_proxy_returns(eod_by_instrument: Dict[str, pd.DataFrame],
                         lookback_days: int = TSMOM_LOOKBACK_DAYS) -> pd.Series:
    """Equal-weighted sign(trailing `lookback_days` return) x next-day return,
    averaged across the basket -- a minimal, standard TSMOM construction. Each
    instrument's own trailing return determines ITS OWN position sign
    (causal: uses only data available before the return being earned)."""
    legs = []
    for sym, df in eod_by_instrument.items():
        s = pd.Series(df["adjusted_close"].values, index=pd.to_datetime(df["date"]))
        ret = s.pct_change()
        trailing = s.pct_change(lookback_days).shift(1)
        position = np.sign(trailing)
        legs.append(position * ret)
    if not legs:
        return pd.Series(dtype=float)
    basket = pd.concat(legs, axis=1)
    return basket.mean(axis=1, skipna=True)


def correlation_to_tsmom(strategy_returns: pd.Series, tsmom_returns: pd.Series) -> dict:
    s, t = align_returns(strategy_returns, tsmom_returns)
    if len(s) < 20 or s.std() == 0 or t.std() == 0:
        return {"corr": np.nan, "n": len(s), "passed": False}
    corr = float(np.corrcoef(s.values, t.values)[0, 1])
    return {"corr": corr, "n": int(len(s)), "passed": bool(abs(corr) < SPEC.diversification.max_abs_corr_tsmom)}
