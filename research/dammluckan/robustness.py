"""
Dammluckan -- Steg 1 estimator-level checks: event-level IC test (vs. block
null) and the mandatory redundancy screen (O+ / O- vs. vol, skew, |r|-
autocorrelation, n-day momentum, distance-to-max, and the TSMOM proxy).

This operates one level below the portfolio backtest: it asks whether
occupation, AT THE MOMENT OF AN UNCONDITIONAL RECORD BREAK, predicts what
happens over the next h trading days -- decoupled from FIFO/gross-cap
portfolio mechanics, which is exactly what "estimatornull före
portföljbygge" (estimator null before portfolio construction) means.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from . import config
from . import metrics
from . import nulls


def _control_frame(panel, n):
    """Side-agnostic market-state controls, evaluated through close t."""
    vol = panel.band_vol
    ret = panel.log_returns
    skew = ret.rolling(config.VOL_LOOKBACK, min_periods=config.VOL_MIN_PERIODS).skew()
    abs_r = ret.abs()
    autocorr = abs_r.rolling(config.VOL_LOOKBACK, min_periods=config.VOL_MIN_PERIODS).apply(
        lambda w: pd.Series(w).autocorr(lag=1), raw=False
    )
    momentum_n = panel.adj_close / panel.adj_close.shift(n) - 1.0
    tsmom_proxy = panel.adj_close / panel.adj_close.shift(252) - 1.0
    return {"vol": vol, "skew": skew, "autocorr_abs_r": autocorr,
            "momentum_n": momentum_n, "tsmom_proxy": tsmom_proxy}


def build_event_table(panel, sig, h, side, controls=None) -> pd.DataFrame:
    """One row per (ticker, event date) where E+ (side='high') / E- (side='low')
    fires -- unconditional on occupation. Columns: o, fwd_return (signed by
    side), dist_to_max, plus the redundancy-screen controls IF a
    precomputed `controls` dict (from _control_frame) is passed -- kept
    optional because _control_frame's autocorrelation term is expensive and
    must not be recomputed inside the (hundreds-of-draws) null-test loop,
    which only needs o/fwd_return."""
    e = sig.e_high if side == "high" else sig.e_low
    o = sig.o_high if side == "high" else sig.o_low
    m = sig.m_high if side == "high" else sig.m_low
    direction = 1.0 if side == "high" else -1.0
    px = panel.adj_close
    raw_px = panel.raw_close

    rows = []
    T = len(panel.dates)
    for ticker in panel.tickers:
        e_arr = e[ticker].to_numpy()
        hit_pos = np.flatnonzero(e_arr == 1)
        if not len(hit_pos):
            continue
        o_arr = o[ticker].to_numpy()
        px_arr = px[ticker].to_numpy()
        raw_arr = raw_px[ticker].to_numpy()
        m_arr = m[ticker].to_numpy()
        ctrl_arrs = {k: v[ticker].to_numpy() for k, v in controls.items()} if controls else {}
        for p in hit_pos:
            if p + h >= T or not np.isfinite(o_arr[p]):
                continue
            fwd = direction * (px_arr[p + h] / px_arr[p] - 1.0) if np.isfinite(px_arr[p]) and px_arr[p] != 0 else np.nan
            dist_to_max = (raw_arr[p] - m_arr[p]) / m_arr[p] if np.isfinite(m_arr[p]) and m_arr[p] != 0 else np.nan
            row = {"ticker": ticker, "date": panel.dates[p], "o": o_arr[p], "fwd_return": fwd,
                   "dist_to_max": dist_to_max}
            for k, arr in ctrl_arrs.items():
                row[k] = arr[p]
            rows.append(row)
    return pd.DataFrame(rows)


def event_level_ic(panel, sig, h, side="both") -> dict:
    """Spearman IC of occupation vs. signed h-day forward return, pooled
    across all (ticker, event-date) rows, unconditional on occupation."""
    tables = []
    if side in ("high", "both"):
        tables.append(build_event_table(panel, sig, h, "high"))
    if side in ("low", "both"):
        tables.append(build_event_table(panel, sig, h, "low"))
    df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    df = df.dropna(subset=["o", "fwd_return"])
    ic = metrics.information_coefficient(df["o"], df["fwd_return"]) if len(df) >= 5 else np.nan
    return {"ic": ic, "n_events": len(df), "table": df}


def ic_null_test(panel, sig, h, n_draws=config.N_BLOCK_DRAWS, seed=0) -> dict:
    """Block-bootstrap null for the pooled event-level IC: for each draw,
    resample every asset's own IS return history (circular block bootstrap),
    rebuild E/O/forward-return on the synthetic path with the SAME (n, c),
    recompute the pooled IC, and derive a two-sided empirical p-value."""
    from . import signal as signal_mod

    real = event_level_ic(panel, sig, h, side="both")
    real_ic = real["ic"]

    rng = np.random.default_rng(seed)
    null_ics = []
    for draw in range(n_draws):
        synth_cols = {}
        for ticker in panel.tickers:
            r = nulls._asset_log_returns(panel, ticker)
            if len(r) <= sig.n + config.VOL_LOOKBACK + h:
                continue
            path = nulls.synthetic_price_path(r, len(r), rng, block=config.BLOCK_LENGTH)
            synth_cols[ticker] = pd.Series(path[: len(panel.dates)], index=panel.dates[: len(path)])
        if not synth_cols:
            continue
        synth_df = pd.DataFrame(synth_cols).reindex(panel.dates)
        synth_panel = _wrap_synthetic_panel(panel, synth_df)
        synth_sig = signal_mod.build_signal(synth_panel, n=sig.n, c=sig.c)
        res = event_level_ic(synth_panel, synth_sig, h, side="both")
        if np.isfinite(res["ic"]):
            null_ics.append(res["ic"])

    null_ics = np.array(null_ics)
    if np.isfinite(real_ic) and len(null_ics):
        pval = (np.sum(np.abs(null_ics) >= abs(real_ic)) + 1) / (len(null_ics) + 1)
    else:
        pval = np.nan
    return {"real_ic": real_ic, "null_ics": null_ics, "p_value": pval, "n_draws": len(null_ics)}


def _wrap_synthetic_panel(panel, synth_raw_close):
    """Build a lightweight Panel-like object around a synthetic raw-close
    path, reusing the real panel's adjustment-factor structure isn't
    meaningful for synthetic data, so adj_close == raw_close and
    adj_open == raw_close (no separate open series needed for the IC test,
    which only touches raw_close/adj_close/dates)."""
    from . import data as data_mod

    return data_mod.Panel(
        tickers=panel.tickers,
        raw_close=synth_raw_close, raw_open=synth_raw_close,
        high=synth_raw_close, low=synth_raw_close,
        adj_close=synth_raw_close, adj_open=synth_raw_close,
        volume=pd.DataFrame(np.nan, index=synth_raw_close.index, columns=synth_raw_close.columns),
    )


def redundancy_regression(panel, sig, h) -> dict:
    """Pooled OLS: O ~ vol + skew + |r|-autocorr + n-day-momentum +
    distance-to-max + TSMOM-proxy. Kill if R^2 > 0.5 (O is redundant with
    already-known state variables)."""
    controls_frame = _control_frame(panel, sig.n)
    tables = [build_event_table(panel, sig, h, "high", controls=controls_frame),
              build_event_table(panel, sig, h, "low", controls=controls_frame)]
    df = pd.concat(tables, ignore_index=True)
    controls = ["vol", "skew", "autocorr_abs_r", "momentum_n", "dist_to_max", "tsmom_proxy"]
    df = df.dropna(subset=["o"] + controls)
    if len(df) < 30:
        return {"r2": np.nan, "n_obs": len(df), "spearman": {}}

    X = sm.add_constant(df[controls].to_numpy())
    y = df["o"].to_numpy()
    model = sm.OLS(y, X).fit()
    r2 = model.rsquared

    spearman = {}
    for c in controls:
        rho, _ = scipy_stats.spearmanr(df["o"], df[c])
        spearman[c] = rho

    return {"r2": r2, "n_obs": len(df), "spearman": spearman, "params": dict(zip(["const"] + controls, model.params))}
