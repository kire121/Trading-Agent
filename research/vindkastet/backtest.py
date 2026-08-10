"""
Vindkastet -- descriptive event backtest.

Implements Regler (3)-(4) literally:
  - Entry: MOC t+1, weights w_i ~ (1/sigma_i) * sign(proj) * [sum_{k=1..K} A^k v*]_i,
    scaled to 0.5% ex-ante portfolio risk, capped 30%/name, gross <= 150%.
  - Exit: close of t+1+K, no discretion; a new trigger while holding replaces
    the position (close old, open new, same MOC bar); max one position.
  - Costs: charged in bp per side on turnover (both the closing leg of an old
    position and the opening leg of a new one count as separate "sides").

This module is intentionally a DESCRIPTIVE backtest: given how few trigger
days survive the pre-registered rule (see grid_event_counts.csv), any Sharpe
computed here is not statistically powered and is reported for transparency
only, not as validation.
"""
import numpy as np
import pandas as pd


def run_backtest(panel, prices, returns, trig, K, risk_target=0.005, cap_name=0.30,
                  gross_cap=1.50, cost_bp_list=(2, 5)):
    idx = returns.index
    loc = {d: i for i, d in enumerate(idx)}
    trig_days = list(trig[trig].index)
    fdir = panel["forecast_dir"]
    sigma = panel["sigma"]
    cov_t = panel["cov_t"]

    events = []
    open_pos = None  # dict: entry_i, exit_i, assets, w (dict sym->weight)

    def size_weights(day):
        if day not in fdir:
            return None
        assets, fwd, proj = fdir[day]
        if day not in cov_t:
            return None
        Sigma = cov_t[day]
        sig = sigma.loc[day, assets].values
        direction = np.sign(proj) * fwd
        raw = direction / sig
        if np.linalg.norm(raw) < 1e-12:
            return None
        port_var = raw @ Sigma @ raw
        port_vol = np.sqrt(max(port_var, 1e-18))
        c = risk_target / port_vol
        w = c * raw
        w = np.clip(w, -cap_name, cap_name)
        gross = np.abs(w).sum()
        if gross > gross_cap:
            w = w * (gross_cap / gross)
        return dict(zip(assets, w))

    # walk through trigger days chronologically, applying replace-on-new-trigger
    open_pos = None
    log = []
    for t in trig_days:
        ti = loc[t]
        entry_i = ti + 1
        if entry_i >= len(idx):
            continue
        entry_day = idx[entry_i]
        w = size_weights(t)
        if w is None:
            continue
        # close any open position first: at its own K-day exit if that already
        # elapsed (natural exit), otherwise early at this new entry (replace)
        if open_pos is not None:
            close_i = min(open_pos["exit_i"], entry_i)
            ret = _segment_return(prices, open_pos["entry_i"], close_i, open_pos["w"])
            log.append(dict(entry=idx[open_pos["entry_i"]], exit=idx[close_i], w=open_pos["w"],
                             ret=ret, replaced=close_i == entry_i and close_i < open_pos["exit_i"]))
            open_pos = None
        exit_i = entry_i + K
        exit_i = min(exit_i, len(idx) - 1)
        open_pos = dict(entry_i=entry_i, exit_i=exit_i, w=w)

    if open_pos is not None:
        ret = _segment_return(prices, open_pos["entry_i"], open_pos["exit_i"], open_pos["w"])
        log.append(dict(entry=idx[open_pos["entry_i"]], exit=idx[open_pos["exit_i"]], w=open_pos["w"],
                         ret=ret, replaced=False))

    trades = pd.DataFrame(log)
    if trades.empty:
        return trades, {}

    results = {}
    for cost_bp in cost_bp_list:
        # cost = 2 sides (entry+exit) * bp * gross weight, applied per trade
        gross_per_trade = trades["w"].apply(lambda w: np.abs(np.array(list(w.values()))).sum())
        cost = 2 * (cost_bp / 1e4) * gross_per_trade
        net_ret = trades["ret"] - cost
        results[cost_bp] = dict(
            n_trades=len(trades),
            gross_mean=trades["ret"].mean(),
            net_mean=net_ret.mean(),
            net_sum=net_ret.sum(),
            net_std=net_ret.std(),
            sharpe_per_event=net_ret.mean() / net_ret.std() if net_ret.std() > 0 else np.nan,
            hit_rate=(net_ret > 0).mean(),
            net_returns=net_ret.values,
        )
    trades["net_ret_2bp"] = trades["ret"] - 2 * (2 / 1e4) * gross_per_trade
    trades["net_ret_5bp"] = trades["ret"] - 2 * (5 / 1e4) * gross_per_trade
    return trades, results


def _segment_return(prices, entry_i, exit_i, w):
    p_entry = prices.iloc[entry_i]
    p_exit = prices.iloc[exit_i]
    ret = 0.0
    for sym, wi in w.items():
        if pd.isna(p_entry[sym]) or pd.isna(p_exit[sym]):
            continue
        ret += wi * (p_exit[sym] / p_entry[sym] - 1.0)
    return ret
