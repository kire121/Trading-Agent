"""
Standard time-series momentum (TSMOM) proxy, built fresh for this repo since
no prior TSMOM implementation exists to reuse here. Used only as a
diversification check: the Formdriften strategy must show |corr| < 0.25
against it, i.e. it must not just be repackaged trend-following.

Classic Moskowitz-Ooi-Pedersen construction: monthly rebalance, position
sign = sign(trailing 12-month return), size ~ 1/63-day vol, gross 100%.
"""
import numpy as np
import pandas as pd


def tsmom_returns(returns_panel, eligible_panel, lookback=252, vol_window=63, max_weight=0.15,
                   rebalance_dates=None):
    idx = returns_panel.index
    if rebalance_dates is None:
        from .universe import month_end_dates
        rebalance_dates = month_end_dates(idx)
    rebalance_dates = [d for d in rebalance_dates if d in idx]

    vol_panel = returns_panel.rolling(vol_window, min_periods=vol_window // 2).std()
    daily_w = pd.DataFrame(0.0, index=idx, columns=returns_panel.columns)

    for i, dt in enumerate(rebalance_dates):
        pos = idx.get_loc(dt)
        if pos < lookback + 1:
            continue
        # trailing 12m return using data through t-1 (no lookahead), matching the strategy's own convention
        past_prices_ratio = (1 + returns_panel[returns_panel.columns].iloc[pos - lookback:pos]).prod() - 1
        sig = np.sign(past_prices_ratio)
        elig = eligible_panel.loc[dt] if dt in eligible_panel.index else pd.Series(True, index=returns_panel.columns)
        elig = elig.reindex(returns_panel.columns).fillna(False)
        sig = sig.where(elig, 0.0)

        vol_t = vol_panel.loc[dt].replace(0, np.nan)
        raw = sig / vol_t
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross = raw.abs().sum()
        w = raw / gross if gross > 0 else raw * 0.0
        w = w.clip(-max_weight, max_weight)
        # renormalize gross to 1.0 after clipping
        g2 = w.abs().sum()
        if g2 > 0:
            w = w / g2

        end_dt = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else idx[-1]
        period_mask = (idx > dt) & (idx <= end_dt)
        daily_w.loc[period_mask, :] = w.values

    ret = (daily_w * returns_panel.fillna(0.0)).sum(axis=1)
    return ret, daily_w
