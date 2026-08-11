"""Primary signal (spec SS2.2) and twin signal constructions (spec SS8).

All signals here are computed on a MultiIndex(ticker,date) panel and reduce
to a per-(ticker, decision-date) g in [-1,1] (or, for T4, a single common
g_t broadcast to all tickers). Decision dates are weekly (Friday close, or
the ISO week's last trading day if Friday is closed) -- see
weekly_decision_dates. Position sizing/costs are NOT done here (see
sizing.py) -- "alla tvillingar delar identisk sizing-/kostnadsvag" (spec
SS4) means every twin below hands its raw per-asset (or common) tilt to the
exact same sizing.solve_k_for_target_vol/apply_gross_cap engine.
"""
import numpy as np
import pandas as pd

from . import config
from . import intrabar


def weekly_decision_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO week present in `dates` (spec SS4:
    "varje fredag close (sista handelsdag i ISO-veckan om stangt)")."""
    s = pd.Series(dates, index=dates)
    iso = s.index.isocalendar()
    key = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    last_per_week = s.groupby(key).max()
    return pd.DatetimeIndex(sorted(last_per_week.to_numpy()))


def compute_g(S: pd.Series, z_star: float) -> pd.Series:
    """g_i(t) = clip(S_i(t)/z*, -1, +1). NaN S stays NaN here (a distinct
    'no data' marker from a genuine g=0); callers convert to a 0 tilt at
    the position-construction stage (spec SS12.2: "K_eff < 0.8K => S=NaN
    => g=0 (position 0, inget fel)")."""
    return (S / z_star).clip(lower=-1.0, upper=1.0)


def primary_signal(shadow: pd.DataFrame, K: int, z_star: float, demean) -> dict:
    """Primary cell (spec SS2.2/SS9 Steg4 grid): S = rolling_tstat(s or
    s_tilde, K), g = clip(S/z*, -1, 1). `demean` is None ("ingen") or an int
    window (252, "252d") -- grid parameter (spec SS9 Steg4)."""
    s = shadow["s"]
    s_input = intrabar.fe_demean(s, K, window=demean) if demean else s
    S = intrabar.rolling_tstat(s_input, K, min_valid=config.K_EFF_MIN_FRACTION)
    g = compute_g(S, z_star)
    return {"s": s, "s_input": s_input, "S": S, "g": g}


def tug_of_war_signal(O: pd.Series, H: pd.Series, L: pd.Series, C: pd.Series,
                       K: int, z_star: float) -> dict:
    """T2 -- "skarpaste redundanstvillingen": d_t = r_ON - r_ID, r_ON =
    ln(O_t/C_{t-1}), r_ID = ln(C_t/O_t); rolling t-stat (K=40), same
    tilt/clip mechanism as the primary cell, NO shadow information at all
    (four-point O/C info only)."""
    if isinstance(C.index, pd.MultiIndex) and "ticker" in (C.index.names or []):
        prev_C = C.groupby(level="ticker").shift(1)
    else:
        prev_C = C.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_on = np.log(O / prev_C)
        r_id = np.log(C / O)
    d = r_on - r_id
    S = intrabar.rolling_tstat(d, K, min_valid=config.K_EFF_MIN_FRACTION)
    g = compute_g(S, z_star)
    return {"d": d, "S": S, "g": g}


def broadcast_signal(s: pd.Series, K: int, z_star: float) -> pd.Series:
    """T4 -- cross-sectional mean s_bar_t (a SINGLE series, not per-ticker),
    same rolling-tstat/clip tilt mechanism, then applied as a common,
    equal-weighted overlay across every asset (spec SS8: "fangar
    gemensam-komponent-alternativet")."""
    if isinstance(s.index, pd.MultiIndex) and "ticker" in (s.index.names or []):
        s_bar = s.groupby(level="date").mean()
    else:
        s_bar = s
    S = intrabar.rolling_tstat(s_bar, K, min_valid=config.K_EFF_MIN_FRACTION)
    return compute_g(S, z_star)


def t5_naive_signal(s_input: pd.Series, K: int) -> pd.Series:
    """T5 -- "enkelhetsregel": mean(sign(s_tilde)) over the trailing K
    window, K_eff>=0.8K else NaN (same NaN discipline as rolling_tstat).
    Already bounded in [-1,1] by construction (an average of +-1/0 signs)
    -- used DIRECTLY as the position tilt, not passed through
    clip(.../z*,-1,1) again, since T5 is a complete alternative FORM of the
    g-computation (spec: "om SR_T5 >= SR_primar - 0.05 => adoptera T5:s
    form"), not an extra layer on top of the primary form."""
    def _one(series: pd.Series) -> pd.Series:
        sign = np.sign(series)
        r = sign.rolling(K, min_periods=1)
        k_eff = series.rolling(K, min_periods=1).count()
        out = r.mean()
        return out.where(k_eff >= config.K_EFF_MIN_FRACTION * K)

    if isinstance(s_input.index, pd.MultiIndex) and "ticker" in (s_input.index.names or []):
        return s_input.groupby(level="ticker", group_keys=False).apply(_one)
    return _one(s_input)


def raw_position_from_g(g: pd.Series, sigma_hat: pd.DataFrame) -> pd.DataFrame:
    """raw_i = g_i / sigma_hat_i (spec SS4), reshaped wide (date x ticker).
    NaN g -> 0 raw (position 0, no error, spec SS12.2)."""
    if isinstance(g.index, pd.MultiIndex):
        g_wide = g.unstack("ticker")
    else:
        g_wide = g
    g_wide = g_wide.reindex(columns=sigma_hat.columns).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = g_wide / sigma_hat
    return raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def annualized_log_return_vol(adjC: pd.DataFrame, lookback: int = config.SIGNAL_VOL_LOOKBACK,
                               periods_per_year: int = config.TRADING_DAYS_YEAR) -> pd.DataFrame:
    """sigma_hat_i: rolling `lookback`-day std of daily LOG returns (adj
    close), annualized (spec SS4, pinned verbatim -- see config.py's
    provenance note on why this differs from Smittotalet's own base-book
    vol convention)."""
    log_ret = np.log(adjC / adjC.shift(1))
    return log_ret.rolling(lookback).std() * np.sqrt(periods_per_year)
