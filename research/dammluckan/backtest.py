"""
Dammluckan -- event-driven backtest engine.

Pipeline: generate_candidates() -> admit_candidates() (FIFO/gross-cap ledger)
-> assign_weights() (inverse-vol sizing + gross-cap haircut, event-driven,
no daily re-sizing) -> daily_returns() (exact open/close leg accounting).

Entry: decided at close t (E+/E- fires and O+/O- >= theta_i), executed at
t+1 open. Exit: h-day time stop or the first opposite-direction record,
whichever executes first; a new same-direction event while a position is
open is ignored (no pyramiding, no re-entry, per the brief). FIFO cap on
concurrent positions: an asset already holding a position ignores new
triggers; when the book is at MAX_CONCURRENT, a new trigger for a flat asset
is simply forgone (not queued) -- this is itself part of the hypothesis's
"episodic, capacity-limited" character, not a modeling shortcut.
"""
import heapq
import numpy as np
import pandas as pd

from . import config
from . import costs
from . import metrics
from .portfolio import CandidateTrade, Trade, inverse_vol_weight, trade_cost_fraction


def compute_exit(entry_pos, h, opp_events, T):
    """Given an entry position, the h-day time-stop horizon, and the
    opposite-direction event array, return (exit_pos, exit_reason) --
    whichever of time-stop / first-opposite-record executes first. Shared by
    generate_candidates() and the randomized-entry null (battery.py)."""
    exit_pos_timestop = min(entry_pos + h, T - 1)
    scan_end = min(entry_pos + h - 1, T)
    exit_reason, exit_pos = "time_stop", exit_pos_timestop
    if scan_end > entry_pos:
        window = opp_events[entry_pos:scan_end]
        hits = np.flatnonzero(window == 1)
        if len(hits):
            candidate_exit = entry_pos + hits[0] + 1
            if candidate_exit < exit_pos_timestop:
                exit_pos, exit_reason = candidate_exit, "opposite_record"
    return exit_pos, exit_reason


def generate_candidates(panel, sig, theta_high, theta_low, h, comparator="ge"):
    """comparator='ge': occupation-gated primary/Donchian-twin family
    (O >= theta, theta=-inf for the ungated Donchian twin). comparator='le':
    anti-twin family (O <= theta, a LOW-occupation threshold)."""
    calendar = panel.dates
    T = len(calendar)
    candidates = []
    cmp = (lambda o, th: o >= th) if comparator == "ge" else (lambda o, th: o <= th)

    for ticker in panel.tickers:
        e_high = sig.e_high[ticker].to_numpy()
        e_low = sig.e_low[ticker].to_numpy()
        o_high = sig.o_high[ticker].to_numpy()
        o_low = sig.o_low[ticker].to_numpy()
        th_high = theta_high.get(ticker, np.nan)
        th_low = theta_low.get(ticker, np.nan)
        adj_open = panel.adj_open[ticker].to_numpy()
        adv = panel.adv[ticker].to_numpy()

        # NaN means "no threshold available, skip this side"; +/-inf is a
        # deliberate always-true gate (the Donchian twin) and must NOT be
        # treated as unavailable -- np.isfinite(-inf) is False, so the guard
        # here checks isnan specifically, not isfinite.
        if np.isnan(th_high) and np.isnan(th_low):
            continue

        long_trigger = (e_high == 1) & cmp(o_high, th_high) if not np.isnan(th_high) else np.zeros(T, dtype=bool)
        short_trigger = (e_low == 1) & cmp(o_low, th_low) if not np.isnan(th_low) else np.zeros(T, dtype=bool)

        for direction, trigger, opp_events, o_arr in (
            (+1, long_trigger, e_low, o_high),
            (-1, short_trigger, e_high, o_low),
        ):
            for p in np.flatnonzero(trigger):
                entry_pos = p + 1
                if entry_pos >= T:
                    continue
                exit_pos, exit_reason = compute_exit(entry_pos, h, opp_events, T)
                entry_price, exit_price = adj_open[entry_pos], adj_open[exit_pos]
                if not (np.isfinite(entry_price) and np.isfinite(exit_price)) or exit_pos <= entry_pos:
                    continue
                candidates.append(CandidateTrade(
                    ticker=ticker, direction=direction,
                    decision_date=calendar[p], entry_date=calendar[entry_pos], entry_pos=entry_pos,
                    exit_date=calendar[exit_pos], exit_pos=exit_pos, exit_reason=exit_reason,
                    entry_price=entry_price, exit_price=exit_price, o_value=float(o_arr[p]),
                    entry_adv=adv[entry_pos - 1] if entry_pos > 0 else np.nan,
                    exit_adv=adv[exit_pos - 1] if exit_pos > 0 else np.nan,
                ))
    return candidates


def admit_candidates(candidates, max_concurrent=config.MAX_CONCURRENT):
    """FIFO/one-per-asset/max-concurrent admission ledger. Deterministic
    tie-break among same-day competitors: higher occupation (stronger
    conviction) admitted first, then ticker alphabetically."""
    ordered = sorted(candidates, key=lambda c: (c.entry_pos, -c.o_value, c.ticker))
    heap = []              # (exit_pos, ticker)
    open_tickers = set()
    admitted = []
    for c in ordered:
        while heap and heap[0][0] <= c.entry_pos:
            _, ticker = heapq.heappop(heap)
            open_tickers.discard(ticker)
        if c.ticker in open_tickers:
            continue
        if len(heap) >= max_concurrent:
            continue
        heapq.heappush(heap, (c.exit_pos, c.ticker))
        open_tickers.add(c.ticker)
        admitted.append(c)
    return admitted


def assign_weights(panel, admitted, k, gross_cap=config.GROSS_CAP,
                    vol_lookback=config.SIZING_VOL_LOOKBACK):
    """Event-driven inverse-vol sizing: w_i = direction * k / sigma_hat_i
    (sigma_hat_i annualized, measured through the day BEFORE entry). Gross
    exposure is only ever checked/enforced at entry events (no daily
    re-sizing); a new entry that would breach the gross cap is haircut down
    to whatever budget remains, existing open positions are never resized."""
    ordered = sorted(admitted, key=lambda c: c.entry_pos)
    heap = []                  # (exit_pos, abs_weight)
    running_gross = 0.0
    trades = []
    ann = np.sqrt(config.TRADING_DAYS_YEAR)

    for c in ordered:
        while heap and heap[0][0] <= c.entry_pos:
            _, w_abs = heapq.heappop(heap)
            running_gross -= w_abs
        sv = panel.sizing_vol[c.ticker]
        sigma_daily = sv.iloc[c.entry_pos - 1] if c.entry_pos > 0 else np.nan
        sigma_ann = sigma_daily * ann if np.isfinite(sigma_daily) else np.nan
        w_abs_raw = inverse_vol_weight(sigma_ann, k)
        headroom = max(0.0, gross_cap - running_gross)
        w_abs = min(w_abs_raw, headroom)
        w = c.direction * w_abs
        running_gross += w_abs
        heapq.heappush(heap, (c.exit_pos, w_abs))

        gross_return = c.direction * (c.exit_price / c.entry_price - 1.0)
        cost_frac = trade_cost_fraction(c.entry_adv, c.exit_adv)
        net_pnl = w * gross_return - abs(w) * cost_frac
        trades.append(Trade(
            ticker=c.ticker, direction=c.direction, decision_date=c.decision_date,
            entry_date=c.entry_date, entry_pos=c.entry_pos, exit_date=c.exit_date,
            exit_pos=c.exit_pos, exit_reason=c.exit_reason, entry_price=c.entry_price,
            exit_price=c.exit_price, o_value=c.o_value, entry_adv=c.entry_adv, exit_adv=c.exit_adv,
            weight=w, sigma_hat=sigma_ann, gross_return=gross_return, cost_frac=cost_frac,
            net_pnl_contribution=net_pnl,
        ))
    return trades


def daily_returns(panel, trades) -> pd.Series:
    """Exact per-leg accounting: entry day is open->close, held days are
    close->close, exit day is close[t-1]->open[t]; round-trip cost charged
    on the entry day."""
    calendar = panel.dates
    T = len(calendar)
    port_ret = np.zeros(T)
    close_cache, open_cache = {}, {}

    for tr in trades:
        if tr.ticker not in close_cache:
            close_cache[tr.ticker] = panel.adj_close[tr.ticker].to_numpy()
            open_cache[tr.ticker] = panel.adj_open[tr.ticker].to_numpy()
        aclose, aopen = close_cache[tr.ticker], open_cache[tr.ticker]
        p0, p1, w = tr.entry_pos, tr.exit_pos, tr.weight
        if p1 <= p0 or w == 0.0:
            continue
        if np.isfinite(aopen[p0]) and np.isfinite(aclose[p0]) and aopen[p0] != 0:
            port_ret[p0] += w * (aclose[p0] / aopen[p0] - 1.0)
        for t in range(p0 + 1, p1):
            if np.isfinite(aclose[t]) and np.isfinite(aclose[t - 1]) and aclose[t - 1] != 0:
                port_ret[t] += w * (aclose[t] / aclose[t - 1] - 1.0)
        if np.isfinite(aopen[p1]) and np.isfinite(aclose[p1 - 1]) and aclose[p1 - 1] != 0:
            port_ret[p1] += w * (aopen[p1] / aclose[p1 - 1] - 1.0)
        port_ret[p0] -= abs(w) * tr.cost_frac

    return pd.Series(port_ret, index=calendar)


def run_backtest(panel, sig, theta_high, theta_low, h, k, gross_cap=config.GROSS_CAP,
                  max_concurrent=config.MAX_CONCURRENT, comparator="ge"):
    candidates = generate_candidates(panel, sig, theta_high, theta_low, h, comparator=comparator)
    admitted = admit_candidates(candidates, max_concurrent=max_concurrent)
    trades = assign_weights(panel, admitted, k, gross_cap=gross_cap)
    rets = daily_returns(panel, trades)
    return {"candidates": candidates, "admitted": admitted, "trades": trades, "returns": rets}


def annualized_vol(returns: pd.Series) -> float:
    return metrics.ann_vol(returns)


def run_is_oos(panel, sig, theta_high, theta_low, h, is_start=config.IS_START, is_end=config.IS_END,
               target_vol=config.PORTFOLIO_VOL_TARGET, gross_cap=config.GROSS_CAP,
               max_concurrent=config.MAX_CONCURRENT, comparator="ge"):
    """
    Single continuous full-sample backtest (admission/FIFO never resets at
    the IS/OOS-1 boundary -- it's one book; data before is_start is a
    warm-up buffer only, present so the largest grid window (n=160) has a
    full rolling lookback by the first IS date), with the vol-target scalar
    k calibrated using ONLY the [is_start, is_end] slice of realized returns
    and then frozen for the whole sample (so OOS-1 sizing never peeks at
    OOS-1 vol). Returns full/IS/OOS-1 daily-return series plus the blotter.
    """
    candidates = generate_candidates(panel, sig, theta_high, theta_low, h, comparator=comparator)
    admitted = admit_candidates(candidates, max_concurrent=max_concurrent)

    k0 = target_vol
    trades0 = assign_weights(panel, admitted, k0, gross_cap=gross_cap)
    rets0 = daily_returns(panel, trades0)
    vol_is_at_k0 = annualized_vol(rets0.loc[is_start:is_end])
    k_final = k0 * (target_vol / vol_is_at_k0) if np.isfinite(vol_is_at_k0) and vol_is_at_k0 > 0 else k0

    trades = assign_weights(panel, admitted, k_final, gross_cap=gross_cap)
    rets = daily_returns(panel, trades)

    return {
        "candidates": candidates, "admitted": admitted, "trades": trades,
        "returns_full": rets, "returns_is": rets.loc[is_start:is_end],
        "returns_oos1": rets.loc[is_end:].iloc[1:],
        "k": k_final, "vol_is_at_k0": vol_is_at_k0,
        "achieved_vol_is": annualized_vol(rets.loc[is_start:is_end]),
    }


def calibrate_vol_scalar(panel, sig, theta_high, theta_low, h, target_vol=config.PORTFOLIO_VOL_TARGET,
                          gross_cap=config.GROSS_CAP, max_concurrent=config.MAX_CONCURRENT,
                          comparator="ge") -> float:
    """Single-shot linear rescale: run once at k=annualized-vol-units of 1.0
    (i.e. k expressed directly as an annualized per-position vol target),
    measure realized portfolio vol, rescale k proportionally. Valid as long
    as the gross cap doesn't bind materially at either k -- checked by the
    caller (run_calibration reports achieved gross utilization). Every
    strategy variant (primary, Donchian twin, anti-twin) gets its own k so
    each is independently targeting the same 8% ex-ante vol -- an
    apples-to-apples sizing convention, not just an apples-to-apples signal."""
    k0 = target_vol  # sane starting scale: same order of magnitude as target
    res0 = run_backtest(panel, sig, theta_high, theta_low, h, k=k0,
                         gross_cap=gross_cap, max_concurrent=max_concurrent, comparator=comparator)
    vol0 = annualized_vol(res0["returns"])
    if not np.isfinite(vol0) or vol0 <= 0:
        return k0
    k_final = k0 * (target_vol / vol0)
    return k_final
