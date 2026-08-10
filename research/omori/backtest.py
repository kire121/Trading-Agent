"""Event-driven backtest engine: day-by-day causal simulation of entry,
sizing, the daily-updated exit clock, the hard stop, and costs.

Key discovered/declared design properties (documented here, restated in
REPORT.md):

  * The brief separates "Sizing (tilt, aldrig gate)" from "Exit -- karnan:
    ... tau_exit uppdateras dagligen av re-fiten": only the exit clock is
    textually specified as daily-updated. Sizing is therefore fixed AT ENTRY
    (close t0+1, tau=1) and never rebalanced -- avoiding an undocumented
    daily-rebalancing cost assumption the brief's cost model doesn't cover.
  * A mechanical consequence: at tau=1 there cannot yet be
    MIN_POSITIVE_EXCESS_DAYS (4) of data, so the Omori fit is *always*
    unidentified at entry -- p_tilde at entry is therefore always the
    (frozen) prior p_bar_i, never an event-specific fit. Entry-time sizing
    dispersion comes only from cross-instrument prior differences and the
    vol-target scalar; the event-specific information content lives
    entirely in the exit clock from day 5 onward. This matches the brief's
    own framing ("Signalen ar en klocka, inte en kompass").
  * Gross cap (150%) is enforced only against NEW entries (available
    headroom = cap - current gross exposure); existing positions are never
    force-unwound to make room, consistent with fixed-at-entry sizing.
  * The hard stop ("-2x dagsriskbudget kumulativt") is evaluated against the
    position's own NAV-level cumulative P&L (weight, already signed by
    direction, times cumulative underlying move), i.e. in the same units as
    VOL_TARGET_DAILY -- not against the raw underlying's unscaled return,
    which would trip almost immediately for any position sized well below
    100% notional.
  * `weight` (from sizing.raw_target_weight) is ALREADY signed by direction
    (traded_signal = direction * g(p_tilde)) -- P&L must use `weight` alone,
    never `direction * weight`, or short positions silently flip sign.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.omori import config, costs, data, events, priors, signal, sizing


@dataclass
class Position:
    ticker: str
    t0_idx: int
    entry_idx: int
    direction: float
    weight: float               # already signed by direction; see module docstring
    prior_p: float
    entry_cost: float
    p_tilde_entry: float        # = prior_p always, by construction (see below)
    cum_return: float = 0.0    # NAV-level cumulative P&L since entry (weight-scaled)
    p_tilde_trace: list = field(default_factory=list)
    tau_exit_trace: list = field(default_factory=list)


@dataclass
class ClosedEvent:
    ticker: str
    t0_idx: int
    entry_idx: int
    exit_idx: int
    direction: float
    weight: float
    exit_reason: str
    holding_days: int
    gross_return: float   # NAV-level, before costs
    net_return: float     # NAV-level, after entry+exit costs
    p_tilde_entry: float
    volume_z: float
    r0: float


class BacktestResult:
    def __init__(self, daily_returns, closed_events, panel_label):
        self.daily_returns = daily_returns
        self.closed_events = closed_events
        self.panel_label = panel_label

    def events_frame(self):
        rows = [vars(e) for e in self.closed_events]
        return pd.DataFrame(rows)


def run_backtest(panel: data.Panel, priors_dict, p_star,
                  z_threshold=config.Z_VOLUME_THRESHOLD,
                  sigma_mult=config.R0_SIGMA_THRESHOLD,
                  theta=config.THETA_DEFAULT,
                  tau_cap=config.TAU_EXIT_CAP_DEFAULT,
                  kappa=config.KAPPA_DEFAULT,
                  gross_cap=config.GROSS_CAP,
                  vol_target=config.VOL_TARGET_DAILY,
                  apply_costs=True, ef=None):
    """`ef` (an events.EventFields) can be precomputed once and reused
    across grid runs that only vary theta/tau_cap/kappa (the expensive
    rolling volume-z/MAD/sigma computation doesn't depend on those)."""
    if ef is None:
        ef = events.EventFields(panel)
    cands = ef.candidates_long(z_threshold, sigma_mult)
    cands_by_day = {idx: g for idx, g in cands.groupby("date_idx")}

    n = len(panel.index)
    returns = ef.returns
    adv = ef.adv
    sigma_vol_target = data.rolling_return_sigma(returns, config.VOL_TARGET_LOOKBACK)
    excess_cache = {t: ef.excess_volume_series(t) for t in panel.tickers}

    open_positions = {}     # ticker -> Position
    pending_entries = []    # list of dict(ticker, t0_idx, direction, volume_z, r0) to open today
    closed_events = []
    daily_port_return = np.zeros(n)

    for t in range(n):
        # 1. Open yesterday's accepted candidates at today's close.
        for pe in pending_entries:
            ticker = pe["ticker"]
            if ticker in open_positions:
                continue  # instrument re-fired before yesterday's entry landed; drop (rare edge case)
            sigma_hat = sigma_vol_target[ticker].iloc[t]
            prior_p = priors.prior_for(ticker, priors_dict)
            raw_w = sizing.raw_target_weight(pe["direction"], prior_p, p_star, sigma_hat, vol_target)
            if raw_w == 0.0 or not np.isfinite(raw_w):
                continue
            current_gross = sum(abs(p.weight) for p in open_positions.values())
            headroom = max(0.0, gross_cap - current_gross)
            if abs(raw_w) > headroom:
                if headroom <= 0:
                    continue
                raw_w = np.sign(raw_w) * headroom
            adv_t = adv[ticker].iloc[t]
            entry_cost = costs.trade_cost_return(adv_t) if apply_costs else 0.0
            daily_port_return[t] -= abs(raw_w) * entry_cost
            open_positions[ticker] = Position(
                ticker=ticker, t0_idx=pe["t0_idx"], entry_idx=t,
                direction=pe["direction"], weight=raw_w, prior_p=prior_p,
                entry_cost=entry_cost, p_tilde_entry=prior_p,
                # p_tilde AT ENTRY (tau=1) is mechanically always prior_p: a
                # fit needs >=MIN_POSITIVE_EXCESS_DAYS(4) positive-excess
                # days, and at tau=1 at most 1 day of data exists, so
                # fit_omori is always unidentified here -- see signal.shrink.
                # Snapshotting it directly (rather than taking the first
                # daily-re-fit p_tilde at tau>=FIT_START_TAU, which is a
                # DIFFERENT, later quantity) keeps this field's name honest.
            )
        pending_entries = []

        # 2. Accrue today's P&L for open positions (entry day itself
        #    contributes zero price return -- the position is established
        #    AT today's close on the entry day).
        for ticker, pos in open_positions.items():
            if t > pos.entry_idx:
                r = returns[ticker].iloc[t]
                r = 0.0 if not np.isfinite(r) else r
                # pos.weight is ALREADY signed by direction (it comes from
                # sizing.raw_target_weight -> signal.traded_signal, which
                # bakes sign(r0) in) -- do not multiply by direction again,
                # or every short position's P&L sign silently flips back to
                # long-like exposure.
                day_ret = pos.weight * r
                daily_port_return[t] += day_ret
                pos.cum_return += day_ret

        # 3. Daily re-fit (tau >= FIT_START_TAU) + exit checks.
        for ticker, pos in list(open_positions.items()):
            tau = t - pos.t0_idx
            if tau < 1:
                continue
            exit_reason = None

            if pos.cum_return <= config.HARD_STOP_MULT * vol_target:
                exit_reason = "hard_stop"

            if exit_reason is None and tau >= config.FIT_START_TAU:
                e_series = excess_cache[ticker]
                max_avail = min(tau, n - 1 - pos.t0_idx)
                e_path = e_series.values[pos.t0_idx + 1: pos.t0_idx + 1 + max_avail]
                fit = signal.fit_omori(max_avail, e_path)
                p_tilde = signal.shrink(fit, pos.prior_p, kappa)
                tex = signal.tau_exit(fit.c_hat, p_tilde, theta, config.TAU_EXIT_FLOOR, tau_cap)
                pos.p_tilde_trace.append(p_tilde)
                pos.tau_exit_trace.append(tex)
                if tau >= tex:
                    exit_reason = "tau_exit"

            if exit_reason is None and tau >= tau_cap:
                exit_reason = "cap"

            if exit_reason is not None:
                adv_t = adv[ticker].iloc[t]
                exit_cost = costs.trade_cost_return(adv_t) if apply_costs else 0.0
                cost_drag = abs(pos.weight) * (pos.entry_cost + exit_cost)
                daily_port_return[t] -= abs(pos.weight) * exit_cost

                closed_events.append(ClosedEvent(
                    ticker=ticker, t0_idx=pos.t0_idx, entry_idx=pos.entry_idx, exit_idx=t,
                    direction=pos.direction, weight=pos.weight, exit_reason=exit_reason,
                    holding_days=t - pos.entry_idx + 1,
                    gross_return=pos.cum_return,
                    net_return=pos.cum_return - cost_drag,
                    p_tilde_entry=pos.p_tilde_entry,
                    volume_z=ef.volume_z[ticker].iloc[pos.t0_idx],
                    r0=ef.returns[ticker].iloc[pos.t0_idx],
                ))
                del open_positions[ticker]

        # 4. Detect today's new candidate events; apply open-instrument dedup
        #    and same-day clustering; queue survivors to open tomorrow.
        today_cands = cands_by_day.get(t)
        if today_cands is not None and len(today_cands):
            eligible = [(row["ticker"], row["volume_z"]) for _, row in today_cands.iterrows()
                        if row["ticker"] not in open_positions]
            if eligible:
                corr = data.trailing_pairwise_corr(returns, config.CLUSTER_CORR_LOOKBACK, t)
                survivors = set(events.cluster_same_day(eligible, corr, config.CLUSTER_CORR_THRESHOLD))
                for _, row in today_cands.iterrows():
                    if row["ticker"] in survivors:
                        pending_entries.append({
                            "ticker": row["ticker"], "t0_idx": t,
                            "direction": row["direction"], "volume_z": row["volume_z"], "r0": row["r0"],
                        })
                        survivors.discard(row["ticker"])  # avoid double-queue on dup rows

    daily_returns = pd.Series(daily_port_return, index=panel.index)
    return BacktestResult(daily_returns, closed_events, panel.label)
