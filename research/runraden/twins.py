"""Twin strategies T1-T4, run through the identical position-construction
machinery as the real Runraden signal so PnL/Sharpe comparisons are risk-
matched by construction (same vol targeting, gross caps, no-trade band).

T1: the additive-only model (day-position effects, no interaction term g),
    traded identically -- isolates whether adding g(w) on top of T1 earns
    its keep once real portfolio mechanics (caps, bands, costs) are applied.
T2: 1-week reversal -- bet against this week's own realised return.
T3: 1-week time-series momentum (TSMOM-1w) -- bet with this week's own
    realised return.
T4: market-timing twin -- build a single equal-weighted "market basket" of
    the whole IS panel, run the *exact same* word/additive/shrinkage
    pipeline on that one synthetic series, and broadcast its ghat as an
    identical directional signal to every asset (scaled by each asset's own
    vol). This isolates a pure common-factor/market-timing explanation for
    any apparent edge in the per-asset word signal (see K1b).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from additive_model import WalkForwardAdditiveModel
from targets import attach_targets
from words import build_asset_week_panel, weekly_table


def additive_signal(scored_panel: pd.DataFrame) -> pd.Series:
    """T1: use the additive-model fitted value as the raw signal (already a
    column produced by WalkForwardAdditiveModel.score)."""
    return scored_panel["additive_pred"]


def reversal_signal(panel: pd.DataFrame) -> pd.Series:
    """T2: -week_return (bet against this week's own realised move)."""
    return -panel["week_return"]


def tsmom_signal(panel: pd.DataFrame) -> pd.Series:
    """T3: +week_return (bet with this week's own realised move)."""
    return panel["week_return"]


def build_market_basket(prices: dict[str, pd.Series]) -> pd.Series:
    """Equal-weighted synthetic 'market' adjusted-close index from daily
    returns of all assets active on each date."""
    rets = pd.concat(
        [series.sort_index().pct_change().rename(asset) for asset, series in prices.items()],
        axis=1,
    )
    market_ret = rets.mean(axis=1, skipna=True).dropna()
    market_index = (1.0 + market_ret).cumprod()
    return market_index


def fit_market_timing_signal(prices: dict[str, pd.Series], kappa: float, vol_window: int,
                              burn_in_years: int) -> pd.Series:
    """T4 building block: fit the full word pipeline on the equal-weighted
    market basket and return ghat_market indexed by t_signal date (pooling
    both the 5d and 4d tables into one date-indexed series)."""
    market_index = build_market_basket(prices)
    market_panel = build_asset_week_panel({"__MARKET__": market_index})
    if market_panel.empty:
        return pd.Series(dtype=float)
    market_panel = attach_targets(market_panel, {"__MARKET__": market_index}, vol_window=vol_window)

    pieces = []
    for word_len, table_id in ((5, "5d"), (4, "4d")):
        tbl = market_panel[market_panel["table_id"] == table_id]
        if tbl.empty:
            continue
        model = WalkForwardAdditiveModel(word_len, kappa=kappa, burn_in_years=burn_in_years)
        model.fit_walkforward(tbl)
        scored = model.score(tbl)
        piece = tbl[["t_signal"]].join(scored[["ghat"]])
        pieces.append(piece)
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces).set_index("t_signal")["ghat"]
    return out.sort_index()


def market_timing_signal(panel: pd.DataFrame, prices: dict[str, pd.Series], kappa: float,
                          vol_window: int, burn_in_years: int) -> pd.Series:
    """T4: broadcast ghat_market(w_market,t) to every asset for the matching week."""
    ghat_market = fit_market_timing_signal(prices, kappa, vol_window, burn_in_years)
    if ghat_market.empty:
        return pd.Series(np.nan, index=panel.index)
    mapped = panel["t_signal"].map(ghat_market)
    return mapped
