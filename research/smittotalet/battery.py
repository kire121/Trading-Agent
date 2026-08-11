"""Redundancy screen: is R_hat just a monotone transform of the count-EWMA/
vol-clustering twins ("dor mot count-EWMA-tvillingen i Steg 0")?

Two checks, both required to survive, per config.REDUNDANCY_R2_KILL /
config.REDUNDANCY_MIN_DELTA_R2 (the Runraden post-mortem lesson: pair every
null-percentile-style comparison with an absolute-magnitude floor, not just
a threshold that can itself sit at zero):

(a) Direct redundancy: regress R_hat_t on [count_ewma_t, garch_var_proxy_t].
    R^2 > 0.5 -> R_hat is redundant with known twins, killed.
(b) Incremental forward-looking value: regress forward realized vol_{t+h} on
    controls alone vs controls+R_hat. Incremental delta-R^2 must clear
    REDUNDANCY_MIN_DELTA_R2 AND be NW-significant, or it's killed.

GARCH-persistence proxy (DECLARED approximation, same house convention as
the ADV-bucket spread caveat): a RiskMetrics-style EWMA(lambda=0.94) variance
of the sleeve's own aggregated |return| series, used as a stand-in for
fitted per-asset GARCH(1,1) persistence pooled across the universe -- not a
literal MLE GARCH fit.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config
from . import twins as twins_mod


def garch_var_proxy(returns: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    agg = twins_mod.broadcast_aggregate_series(returns)
    return (agg ** 2).ewm(alpha=1 - lam, min_periods=20).mean()


def _ols_r2(y: pd.Series, X: pd.DataFrame):
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 30:
        return np.nan, None
    Xc = sm.add_constant(df.drop(columns="y"))
    model = sm.OLS(df["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model.rsquared, model


def build_battery_frame(r_hat: pd.Series, x_t: pd.Series, returns: pd.DataFrame, tau: int,
                         forward_horizon: int = 21) -> pd.DataFrame:
    count_ewma = x_t.fillna(0.0).ewm(span=tau, min_periods=tau).mean()
    garch = garch_var_proxy(returns)
    agg = twins_mod.broadcast_aggregate_series(returns)
    fwd_vol = agg.rolling(forward_horizon).std().shift(-forward_horizon) * np.sqrt(config.TRADING_DAYS_YEAR)
    return pd.DataFrame({
        "r_hat": r_hat, "count_ewma": count_ewma, "garch_var_proxy": garch, "fwd_vol": fwd_vol,
    })


def redundancy_screen(battery: pd.DataFrame) -> dict:
    df = battery.dropna()
    direct_r2, _ = _ols_r2(df["r_hat"], df[["count_ewma", "garch_var_proxy"]])

    controls_r2, _ = _ols_r2(df["fwd_vol"], df[["count_ewma", "garch_var_proxy"]])
    full_r2, full_model = _ols_r2(df["fwd_vol"], df[["count_ewma", "garch_var_proxy", "r_hat"]])
    delta_r2 = full_r2 - controls_r2 if np.isfinite(full_r2) and np.isfinite(controls_r2) else np.nan
    nw_t = float(full_model.tvalues["r_hat"]) if full_model is not None else np.nan

    direct_kill = np.isfinite(direct_r2) and direct_r2 > config.REDUNDANCY_R2_KILL
    incremental_kill = (
        not np.isfinite(delta_r2)
        or delta_r2 < config.REDUNDANCY_MIN_DELTA_R2
        or not np.isfinite(nw_t)
        or abs(nw_t) < 1.96
    )
    return {
        "direct_r2": direct_r2,
        "controls_r2": controls_r2,
        "full_r2": full_r2,
        "delta_r2": delta_r2,
        "nw_t_r_hat": nw_t,
        "direct_kill": bool(direct_kill),
        "incremental_kill": bool(incremental_kill),
        "killed": bool(direct_kill or incremental_kill),
    }
