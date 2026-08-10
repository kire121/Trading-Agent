"""
Entry/exit/sizing simulation, applied identically to the main cepstral signal
and to each twin (they all produce a {symbol: DataFrame[S_bar, D]} panel, so
one simulator serves all four horse-race legs).

Timing convention: S_bar/D on session_date d are computed from data available
through d's close (spec: "computed after close"). The resulting target weight
w_i,d is executed at d+1's open (spec: "order placed manually at next open")
and held until the next rebalance's execution at d+2's open -- so the return
earned against w_i,d is the open-to-open return from d+1 to d+2. This is the
only convention consistent with a daily-close-decided, next-open-executed
strategy without look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import universe as uni
from .config import SPEC


@dataclass
class _Position:
    side: int
    holding_days: int = 0


def cross_sectional_percentile(values: pd.Series) -> pd.Series:
    return values.rank(pct=True) * 100.0


def _cap_and_redistribute(raw_weights: pd.Series, cap: float, target_gross: float, max_iter: int = 20) -> pd.Series:
    """Scale |raw_weights| to sum to target_gross, then clip each to `cap`
    and redistribute the clipped excess proportionally over the still-
    uncapped names, iterating until stable (or every name is pinned at cap,
    in which case gross necessarily falls short of target_gross)."""
    if raw_weights.empty or raw_weights.abs().sum() == 0:
        return raw_weights * 0.0
    w = raw_weights.abs() / raw_weights.abs().sum() * target_gross
    for _ in range(max_iter):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        under_sum = w[under].sum()
        if under_sum <= 0:
            break
        w[under] = w[under] + excess * (w[under] / under_sum)
    return w * np.sign(raw_weights)


def _rank_tilt_weights(score: pd.Series, side: pd.Series, cap: float, target_gross_per_side: float) -> pd.Series:
    weights = pd.Series(0.0, index=score.index)
    for s in (1, -1):
        names = side[side == s].index
        if len(names) == 0:
            continue
        ranks = score.loc[names].rank(method="average")
        w = _cap_and_redistribute(ranks, cap, target_gross_per_side)
        weights.loc[names] = s * w.values
    return weights


def run_backtest(
    signal_by_symbol: Dict[str, pd.DataFrame],
    universe_panel: pd.DataFrame,
    eod_by_symbol: Dict[str, pd.DataFrame],
    spec=SPEC,
) -> dict:
    """Returns a dict with: 'daily' (DataFrame of gross return, cost, net
    return, turnover, n_long, n_short, n_positions per day) and 'weights'
    (DataFrame date x symbol of applied weights, for diagnostics/diversification)."""
    signal_by_symbol = {s: df.copy() for s, df in signal_by_symbol.items()}
    for df in signal_by_symbol.values():
        df.index = pd.to_datetime(df.index)

    open_by_symbol = {}
    for s, df in eod_by_symbol.items():
        idx = pd.to_datetime(df["date"])
        open_by_symbol[s] = pd.Series(df["adj_open"].values, index=idx)

    all_dates = sorted(set(universe_panel["date"].unique()) &
                        set().union(*[set(df.index) for df in signal_by_symbol.values()]))

    book: Dict[str, _Position] = {}
    prev_weights = pd.Series(dtype=float)
    daily_rows = []
    weight_rows = {}
    realized_returns_hist: list = []

    for i, d in enumerate(all_dates):
        universe_today = set(uni.universe_on(universe_panel, d))

        s_today = pd.Series({s: signal_by_symbol[s].loc[d, "S_bar"] for s in universe_today
                              if s in signal_by_symbol and d in signal_by_symbol[s].index})
        d_today = pd.Series({s: signal_by_symbol[s].loc[d, "D"] for s in universe_today
                              if s in signal_by_symbol and d in signal_by_symbol[s].index})
        s_today = s_today.dropna()
        d_today = d_today.reindex(s_today.index)
        pctile = cross_sectional_percentile(s_today) if len(s_today) else pd.Series(dtype=float)

        exited_today = set()
        for sym, pos in list(book.items()):
            pos.holding_days += 1
            missing = sym not in universe_today or sym not in pctile.index or pd.isna(d_today.get(sym, np.nan))
            hysteresis_exit = (not missing) and pctile[sym] < spec.entry_exit.exit_score_pctile
            flip_exit = (not missing) and int(np.sign(d_today[sym])) != 0 and int(np.sign(d_today[sym])) != pos.side
            time_exit = pos.holding_days >= spec.entry_exit.max_holding_days
            if missing or hysteresis_exit or flip_exit or time_exit:
                del book[sym]
                exited_today.add(sym)  # an exit and a same-day re-entry are not the same trading decision

        candidates = pctile[pctile >= spec.entry_exit.entry_score_pctile].index
        for sym in candidates:
            if sym in book or sym in exited_today:
                continue
            dv = d_today.get(sym, np.nan)
            if pd.isna(dv) or abs(dv) < spec.entry_exit.entry_min_abs_direction:
                continue
            book[sym] = _Position(side=int(np.sign(dv)), holding_days=0)

        held = list(book.keys())
        score_for_ranking = s_today.reindex(held)
        side_series = pd.Series({s: book[s].side for s in held})
        target_gross_per_side = spec.sizing.target_gross / 2.0
        base_weights = _rank_tilt_weights(score_for_ranking, side_series, spec.sizing.max_weight_per_name,
                                           target_gross_per_side)

        realized_vol = np.nan
        if len(realized_returns_hist) >= spec.sizing.vol_lookback_days // 3:
            recent = np.array(realized_returns_hist[-spec.sizing.vol_lookback_days:])
            realized_vol = recent.std() * np.sqrt(spec.sizing.trading_days_per_year)
        if realized_vol and realized_vol > 0 and not np.isnan(realized_vol):
            multiplier = float(np.clip(spec.sizing.vol_target_annual / realized_vol, 0.2, 3.0))
        else:
            multiplier = 1.0

        scaled = base_weights * multiplier
        longs = scaled[scaled > 0]
        shorts = scaled[scaled < 0]
        if len(longs):
            longs = _cap_and_redistribute(longs, spec.sizing.max_weight_per_name, longs.sum())
        if len(shorts):
            shorts = _cap_and_redistribute(shorts, spec.sizing.max_weight_per_name, -shorts.sum())
        target_weights = pd.concat([longs, shorts]) if (len(longs) or len(shorts)) else pd.Series(dtype=float)

        all_names = sorted(set(prev_weights.index) | set(target_weights.index))
        prev_aligned = prev_weights.reindex(all_names).fillna(0.0)
        target_aligned = target_weights.reindex(all_names).fillna(0.0)

        applied = prev_aligned.copy()
        for sym in all_names:
            wt, wp = target_aligned[sym], prev_aligned[sym]
            is_open_position = sym in book
            if not is_open_position:
                applied[sym] = 0.0  # exits always flatten, band-buffer doesn't apply to closing a position
                continue
            if wp == 0.0 or abs(wt - wp) > spec.sizing.band_buffer_frac * abs(wt):
                applied[sym] = wt
            else:
                applied[sym] = wp

        turnover = (applied - prev_aligned).abs().sum()
        cost = turnover * (spec.cost.commission_bps_per_side + spec.cost.fallback_half_spread_bps) / 10_000.0

        gross_return = 0.0
        if i + 2 < len(all_dates):
            d1, d2 = all_dates[i + 1], all_dates[i + 2]
            for sym, w in applied.items():
                if w == 0.0 or sym not in open_by_symbol:
                    continue
                px = open_by_symbol[sym]
                if d1 in px.index and d2 in px.index and px[d1] > 0:
                    gross_return += w * (px[d2] / px[d1] - 1.0)

        net_return = gross_return - cost
        realized_returns_hist.append(net_return)

        daily_rows.append({
            "date": d, "gross_return": gross_return, "cost": cost, "net_return": net_return,
            "turnover": turnover, "n_long": int((applied > 0).sum()), "n_short": int((applied < 0).sum()),
            "n_positions": int((applied != 0).sum()), "vol_target_multiplier": multiplier,
        })
        weight_rows[d] = applied

        prev_weights = applied[applied != 0.0]

    daily_df = pd.DataFrame(daily_rows).set_index("date")
    weights_df = pd.DataFrame(weight_rows).T.fillna(0.0).sort_index()
    return {"daily": daily_df, "weights": weights_df}
