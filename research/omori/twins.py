"""Pre-registered null-hypothesis baselines the strategy must beat net of
cost at matched effective exposure/horizon, or it is dead regardless of raw
profitability:

  T1 -- fixed-horizon twin: identical events/direction/sizing, but the
        adaptive exit clock is replaced by a single fixed horizon (the IS
        median of the adaptive clock's own realized tau_exit-driven holding
        days). If the clock can't beat this, the whole "when to exit" bet
        adds nothing over "just pick the average horizon".
  T2 -- p_tilde block-shuffled across events WITHIN instrument: each real
        event's realized holding-day count is swapped for another event's
        FROM THE SAME TICKER (a random within-ticker permutation). Tests
        whether the SPECIFIC pairing of an event to its own adaptive horizon
        carries information, vs. any horizon drawn from that instrument's
        own horizon distribution being about as good.
  T3 -- randomized entry days: each real event's (ticker, direction, weight,
        holding-days) tuple is replayed at a RANDOM valid entry date for
        that ticker instead of the day after its actual volume-shock event
        -- same exposure and horizon distribution by construction, but
        without the event trigger.

All three reuse the real backtest's own per-event (ticker, direction,
weight, holding_days) so exposure and horizon distributions are matched by
construction; only the entry-date-to-horizon assignment changes. Costs and
the hard stop are re-applied identically to the primary backtest (declared
simplification: the portfolio-level gross cap is NOT re-enforced for these
diagnostic replays -- they compare per-event and cost-adjusted daily P&L
distributions, not a literally re-constructed tradeable portfolio).
"""
import numpy as np
import pandas as pd

from research.omori import config, costs, data


def _replay(panel, ef, rows, entry_idx_of, horizon_of, apply_costs=True):
    n = len(panel.index)
    returns = ef.returns
    adv = ef.adv
    vol_target = config.VOL_TARGET_DAILY
    daily_port_return = np.zeros(n)
    net_returns = []

    for row in rows:
        ticker = row["ticker"]
        entry_idx = entry_idx_of(row)
        horizon = max(1, int(horizon_of(row)))
        if entry_idx is None or entry_idx >= n - 1:
            continue
        weight = row["weight"]  # already signed by direction -- see backtest.py's note
        adv_entry = adv[ticker].iloc[entry_idx]
        entry_cost = costs.trade_cost_return(adv_entry) if apply_costs else 0.0
        cum_return = 0.0
        exit_idx = min(entry_idx + horizon, n - 1)
        for t in range(entry_idx + 1, exit_idx + 1):
            r = returns[ticker].iloc[t]
            r = 0.0 if not np.isfinite(r) else r
            day_ret = weight * r
            daily_port_return[t] += day_ret
            cum_return += day_ret
            if cum_return <= config.HARD_STOP_MULT * vol_target:
                exit_idx = t
                break
        adv_exit = adv[ticker].iloc[exit_idx]
        exit_cost = costs.trade_cost_return(adv_exit) if apply_costs else 0.0
        cost_drag = abs(weight) * (entry_cost + exit_cost)
        daily_port_return[exit_idx] -= abs(weight) * exit_cost
        net_returns.append(cum_return - cost_drag)

    daily_returns = pd.Series(daily_port_return, index=panel.index)
    return daily_returns, np.array(net_returns)


def t1_fixed_horizon(panel, ef, closed_events_df, fixed_horizon=None, apply_costs=True):
    if fixed_horizon is None:
        adaptive = closed_events_df.loc[closed_events_df.exit_reason == "tau_exit", "holding_days"]
        fixed_horizon = int(round(adaptive.median())) if len(adaptive) else config.TAU_EXIT_CAP_DEFAULT
    rows = closed_events_df.to_dict("records")
    return _replay(panel, ef, rows, lambda r: r["entry_idx"], lambda r: fixed_horizon, apply_costs)


def t2_shuffled_within_instrument(panel, ef, closed_events_df, seed=2, apply_costs=True):
    rng = np.random.default_rng(seed)
    df = closed_events_df.copy()
    shuffled_horizon = df["holding_days"].copy()
    for ticker, grp in df.groupby("ticker"):
        idx = grp.index.values
        if len(idx) > 1:
            perm = rng.permutation(idx)
            shuffled_horizon.loc[idx] = df.loc[perm, "holding_days"].values
    df["_shuffled_horizon"] = shuffled_horizon
    rows = df.to_dict("records")
    return _replay(panel, ef, rows, lambda r: r["entry_idx"], lambda r: r["_shuffled_horizon"], apply_costs)


def t3_randomized_entry(panel, ef, closed_events_df, seed=3, apply_costs=True):
    """Reuses _replay_t3 rather than the generic _replay helper because the
    random entry date has to be drawn once per row up front (keyed by
    position, not recoverable from the row dict alone)."""
    rng = np.random.default_rng(seed)
    n = len(panel.index)
    close = panel.close
    rows = closed_events_df.to_dict("records")
    random_entries = {}
    for i, row in enumerate(rows):
        ticker = row["ticker"]
        valid = np.where(close[ticker].notna().values)[0]
        valid = valid[(valid > 0) & (valid < n - 2)]
        random_entries[i] = int(rng.choice(valid)) if len(valid) else None
    return _replay_t3(panel, ef, rows, random_entries, apply_costs)


def _replay_t3(panel, ef, rows, random_entries, apply_costs):
    n = len(panel.index)
    returns = ef.returns
    adv = ef.adv
    vol_target = config.VOL_TARGET_DAILY
    daily_port_return = np.zeros(n)
    net_returns = []
    for i, row in enumerate(rows):
        entry_idx = random_entries[i]
        if entry_idx is None or entry_idx >= n - 1:
            continue
        ticker, weight = row["ticker"], row["weight"]  # weight already signed by direction
        horizon = max(1, int(row["holding_days"]))
        adv_entry = adv[ticker].iloc[entry_idx]
        entry_cost = costs.trade_cost_return(adv_entry) if apply_costs else 0.0
        cum_return = 0.0
        exit_idx = min(entry_idx + horizon, n - 1)
        for t in range(entry_idx + 1, exit_idx + 1):
            r = returns[ticker].iloc[t]
            r = 0.0 if not np.isfinite(r) else r
            day_ret = weight * r
            daily_port_return[t] += day_ret
            cum_return += day_ret
            if cum_return <= config.HARD_STOP_MULT * vol_target:
                exit_idx = t
                break
        adv_exit = adv[ticker].iloc[exit_idx]
        exit_cost = costs.trade_cost_return(adv_exit) if apply_costs else 0.0
        cost_drag = abs(weight) * (entry_cost + exit_cost)
        daily_port_return[exit_idx] -= abs(weight) * exit_cost
        net_returns.append(cum_return - cost_drag)
    daily_returns = pd.Series(daily_port_return, index=panel.index)
    return daily_returns, np.array(net_returns)
