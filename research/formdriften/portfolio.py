"""
Cross-sectional portfolio construction for the Formdriften strategy.

Monthly rebalance at the close of the last trading day of the month, signal
computed on data through t-1. Cross-sectional z-score of D_L; long the top
quintile, short the bottom quintile. Hysteresis: an open position is only
closed once the name leaves the top/bottom 30% band. Sizing: inverse 63-day
realized vol, gross 100% (50% long / 50% short), net 0, capped at 10%/name.
No discretion -- every date below is decided by the rule set, not by hand.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import costs


def month_end_dates(index):
    s = pd.Series(index, index=index)
    return s.groupby([index.year, index.month]).apply(lambda x: x.iloc[-1]).values


def inverse_vol_weights(names, vol, leg_gross, max_weight):
    """Iteratively-capped inverse-vol weights for one leg (long or short), summing to leg_gross."""
    if len(names) == 0:
        return pd.Series(dtype=float)
    v = vol.reindex(names).astype(float)
    v = v.replace(0, np.nan)
    invvol = 1.0 / v
    invvol = invvol.fillna(invvol.median() if invvol.notna().any() else 1.0)
    w = invvol / invvol.sum() * leg_gross
    capped = pd.Series(False, index=w.index)
    for _ in range(len(names) + 1):
        over = (w > max_weight + 1e-12) & (~capped)
        if not over.any():
            break
        w[over] = max_weight
        capped[over] = True
        remaining = leg_gross - w[capped].sum()
        uncapped = w.index[~capped]
        if len(uncapped) == 0:
            break
        iv_u = invvol[uncapped]
        if remaining <= 0 or iv_u.sum() == 0:
            w[uncapped] = 0.0
            break
        w[uncapped] = iv_u / iv_u.sum() * remaining
    return w


@dataclass
class BacktestConfig:
    curr_window: int = 126
    prev_window: int = 126
    entry_pct: float = 0.20   # top/bottom quintile to enter
    exit_pct: float = 0.30    # hysteresis band: exit once outside top/bottom 30%
    vol_window: int = 63
    leg_gross: float = 0.5
    max_weight: float = 0.10
    min_names: int = 10
    n_legs: str = "quintile"  # "quintile" or "tertile"


def _entry_exit_pcts(cfg: BacktestConfig):
    if cfg.n_legs == "tertile":
        return 1.0 / 3.0, 0.40
    return cfg.entry_pct, cfg.exit_pct


def run_backtest(returns_panel, d_l_panel, eligible_panel, adv_panel, cfg: BacktestConfig,
                  rebalance_dates=None):
    """
    returns_panel: DataFrame[date, ticker] daily simple returns.
    d_l_panel:     DataFrame[date, ticker] D_L signal (already computed, NaN where undefined).
    eligible_panel: DataFrame[date, ticker] bool -- PIT ADV + history eligibility.
    adv_panel:     DataFrame[date, ticker] trailing dollar ADV (for cost bucketing).
    """
    entry_pct, exit_pct = _entry_exit_pcts(cfg)
    idx = returns_panel.index
    if rebalance_dates is None:
        rebalance_dates = month_end_dates(idx)
    rebalance_dates = [d for d in rebalance_dates if d in idx]

    vol_panel = returns_panel.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std()

    weights = pd.Series(0.0, index=returns_panel.columns)
    side = pd.Series("", index=returns_panel.columns)  # "long"/"short"/""
    daily_w = pd.DataFrame(0.0, index=idx, columns=returns_panel.columns)
    turnover_log = []
    cost_log = []
    holdings_log = []

    for i, dt in enumerate(rebalance_dates):
        elig = eligible_panel.loc[dt]
        elig_names = elig.index[elig.fillna(False)]
        dl = d_l_panel.loc[dt, elig_names].dropna()
        n = len(dl)
        if n < cfg.min_names:
            new_weights = pd.Series(0.0, index=returns_panel.columns)
            new_side = pd.Series("", index=returns_panel.columns)
        else:
            z = (dl - dl.mean()) / dl.std(ddof=0)
            rank_pct = z.rank(pct=True)  # 0 = worst (most negative D_L), 1 = best

            top_entry = rank_pct >= (1 - entry_pct)
            bot_entry = rank_pct <= entry_pct
            top_hold = rank_pct >= (1 - exit_pct)
            bot_hold = rank_pct <= exit_pct

            prev_side = side.reindex(dl.index).fillna("")
            keep_long = (prev_side == "long") & top_hold
            keep_short = (prev_side == "short") & bot_hold
            new_long_names = dl.index[(top_entry) | keep_long]
            new_short_names = dl.index[(bot_entry) | keep_short]
            # a name cannot be both; entry masks are disjoint by construction of
            # top/bottom percentile with entry_pct <= 0.5, so no conflict.

            vol_t = vol_panel.loc[dt]
            w_long = inverse_vol_weights(list(new_long_names), vol_t, cfg.leg_gross, cfg.max_weight)
            w_short = inverse_vol_weights(list(new_short_names), vol_t, cfg.leg_gross, cfg.max_weight)

            new_weights = pd.Series(0.0, index=returns_panel.columns)
            new_weights.loc[w_long.index] = w_long.values
            new_weights.loc[w_short.index] = -w_short.values

            new_side = pd.Series("", index=returns_panel.columns)
            new_side.loc[w_long.index] = "long"
            new_side.loc[w_short.index] = "short"

        delta = (new_weights - weights).abs()
        turnover = delta.sum() / 2.0
        adv_t = adv_panel.loc[dt].reindex(returns_panel.columns)
        cost_bps = costs.round_trip_cost_bps(adv_t.values)
        trade_cost = float((delta.values * cost_bps / 1e4).sum())  # fraction of NAV

        turnover_log.append({"date": dt, "turnover": turnover, "n_long": (new_weights > 0).sum(),
                              "n_short": (new_weights < 0).sum(), "n_eligible": n})
        cost_log.append({"date": dt, "cost": trade_cost})
        holdings_log.append(new_weights.copy())

        weights = new_weights
        side = new_side

        end_dt = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else idx[-1]
        period_mask = (idx > dt) & (idx <= end_dt)
        daily_w.loc[period_mask, :] = weights.values

    turnover_df = pd.DataFrame(turnover_log).set_index("date")
    cost_df = pd.DataFrame(cost_log).set_index("date")

    # daily_w[d] is already the weight decided at the most recent rebalance
    # strictly before d (trade executes at dt's close; the return earned on
    # day d = (P_d - P_{d-1})/P_{d-1} only starts accruing to the new
    # position from the day after the trade) -- no additional lag needed.
    gross_ret = (daily_w * returns_panel).sum(axis=1)
    cost_series = pd.Series(0.0, index=idx)
    cost_series.loc[cost_df.index] = cost_df["cost"].values
    net_ret = gross_ret - cost_series

    holdings_df = pd.DataFrame(holdings_log, index=turnover_df.index)

    return {
        "gross_returns": gross_ret,
        "net_returns": net_ret,
        "daily_weights": daily_w,
        "turnover": turnover_df,
        "costs": cost_df,
        "holdings": holdings_df,
    }
