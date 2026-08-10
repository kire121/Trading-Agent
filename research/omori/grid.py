"""Parameter-neighborhood grid (z x theta x tau_cap), sign-stability check,
three-era sub-period consistency, and the DSR trial pool these feed.
"""
import itertools

import numpy as np
import pandas as pd

from research.omori import backtest, config, data, events, metrics


def run_grid(panel, priors_dict, p_star, kappa=config.KAPPA_DEFAULT,
             z_grid=config.Z_VOLUME_GRID, theta_grid=config.THETA_GRID,
             tau_cap_grid=config.TAU_EXIT_CAP_GRID, apply_costs=True):
    ef = events.EventFields(panel)
    rows = []
    for z, theta, tau_cap in itertools.product(z_grid, theta_grid, tau_cap_grid):
        res = backtest.run_backtest(panel, priors_dict, p_star, z_threshold=z, theta=theta,
                                     tau_cap=tau_cap, kappa=kappa, apply_costs=apply_costs, ef=ef)
        sharpe = metrics.annualized_sharpe(res.daily_returns)
        total_ret = float(res.daily_returns.sum())
        rows.append({
            "z_threshold": z, "theta": theta, "tau_cap": tau_cap,
            "sharpe": sharpe, "total_return": total_ret, "n_events": len(res.closed_events),
            "is_primary": (z == config.Z_VOLUME_THRESHOLD and theta == config.THETA_DEFAULT
                           and tau_cap == config.TAU_EXIT_CAP_DEFAULT),
        })
    return pd.DataFrame(rows)


def sign_stability(grid_df, sign_col="sharpe"):
    primary = grid_df.loc[grid_df.is_primary]
    if not len(primary):
        return {"primary_sign": None, "frac_matching": np.nan, "stable": False}
    primary_sign = np.sign(primary[sign_col].iloc[0])
    matching = np.sign(grid_df[sign_col]) == primary_sign
    frac = float(matching.mean())
    return {"primary_sign": float(primary_sign), "frac_matching": frac, "stable": bool(frac == 1.0)}


def era_consistency(panel, priors_dict, p_star, kappa=config.KAPPA_DEFAULT,
                     era_boundaries=config.ERA_BOUNDARIES, apply_costs=True):
    ef = events.EventFields(panel)
    res_full = backtest.run_backtest(panel, priors_dict, p_star, kappa=kappa,
                                      apply_costs=apply_costs, ef=ef)
    out = []
    bounds = pd.to_datetime(era_boundaries)
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        mask = (res_full.daily_returns.index > start) & (res_full.daily_returns.index <= end)
        era_returns = res_full.daily_returns[mask]
        sharpe = metrics.annualized_sharpe(era_returns)
        out.append({
            "era": f"{start.date()}..{end.date()}", "sharpe": sharpe,
            "total_return": float(era_returns.sum()), "n_days": int(mask.sum()),
        })
    return pd.DataFrame(out)


def dsr_from_grid(daily_returns, grid_df):
    sharpe = metrics.annualized_sharpe(daily_returns)
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    skew = float(pd.Series(r).skew()) if len(r) > 2 else 0.0
    kurt = float(pd.Series(r).kurtosis()) + 3.0 if len(r) > 2 else 3.0
    return metrics.deflated_sharpe_ratio(sharpe, len(r), grid_df["sharpe"].values, skew=skew, kurtosis=kurt)
