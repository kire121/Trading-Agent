"""
Dammluckan -- 27-variant pre-registered grid (n x theta_pctl x h), DSR
deflation, and the neighborhood sign-majority robustness check.
"""
import itertools
import numpy as np
import pandas as pd

from . import config
from . import signal as signal_mod
from . import backtest
from . import metrics


def run_grid(panel, calib, is_end=config.IS_END, target_vol=config.PORTFOLIO_VOL_TARGET):
    """Runs all 27 (n, theta_pctl, h) cells. Signal (which depends only on n)
    is built once per n and reused across the 9 (theta_pctl, h) sub-cells
    that share it."""
    rows = []
    cell_returns = {}
    for n in config.N_GRID:
        c = calib["c"][n]
        sig = signal_mod.build_signal(panel, n=n, c=c)
        for pctl in config.THETA_PCTL_GRID:
            th_high = calib["theta_high"][n][pctl]
            th_low = calib["theta_low"][n][pctl]
            for h in config.H_GRID:
                res = backtest.run_is_oos(panel, sig, th_high, th_low, h, is_end=is_end, target_vol=target_vol)
                cell_id = f"n{n}_p{pctl}_h{h}"
                is_sharpe = metrics.sharpe(res["returns_is"])
                oos_sharpe = metrics.sharpe(res["returns_oos1"])
                rows.append({
                    "cell_id": cell_id, "n": n, "theta_pctl": pctl, "h": h,
                    "is_sharpe": is_sharpe, "oos1_sharpe": oos_sharpe,
                    "is_sharpe_pp": is_sharpe / np.sqrt(config.TRADING_DAYS_YEAR) if np.isfinite(is_sharpe) else np.nan,
                    "oos1_sharpe_pp": oos_sharpe / np.sqrt(config.TRADING_DAYS_YEAR) if np.isfinite(oos_sharpe) else np.nan,
                    "n_trades": len(res["trades"]),
                    "n_trades_is": sum(1 for t in res["trades"]
                                       if pd.Timestamp(config.IS_START) <= t.entry_date <= pd.Timestamp(is_end)),
                    "is_primary": (n == config.N_DEFAULT and pctl == config.THETA_PCTL_DEFAULT and h == config.H_DEFAULT),
                })
                cell_returns[cell_id] = res["returns_full"]
    table = pd.DataFrame(rows)
    return {"table": table, "returns_by_cell": cell_returns}


def neighborhood_isolation_check(table: pd.DataFrame, target_cell_id: str, sharpe_col="is_sharpe",
                                  neighbor_frac_threshold=0.5) -> dict:
    """Flags a lone-winner fluke: among cells that differ from the target in
    exactly ONE grid axis (n, theta_pctl, or h), what fraction share the
    target's Sharpe SIGN? A real effect should show a sign majority in its
    immediate neighborhood, not an isolated spike."""
    row = table.loc[table.cell_id == target_cell_id].iloc[0]
    target_sign = np.sign(row[sharpe_col])
    neighbors = table[
        ((table.n == row.n) & (table.theta_pctl == row.theta_pctl) & (table.h != row.h)) |
        ((table.n == row.n) & (table.h == row.h) & (table.theta_pctl != row.theta_pctl)) |
        ((table.theta_pctl == row.theta_pctl) & (table.h == row.h) & (table.n != row.n))
    ]
    same_sign = (np.sign(neighbors[sharpe_col]) == target_sign).sum()
    frac = same_sign / len(neighbors) if len(neighbors) else np.nan
    return {"target_sign": target_sign, "n_neighbors": len(neighbors), "n_same_sign": int(same_sign),
            "frac_same_sign": frac, "passes": (frac >= neighbor_frac_threshold) if np.isfinite(frac) else False}


def dsr_from_grid(table: pd.DataFrame, returns_by_cell: dict, primary_cell_id: str,
                   period="is") -> dict:
    """DSR of the primary cell's Sharpe against the 27-cell grid as the
    trial pool (per-period Sharpes throughout, matching every sibling
    branch's own explicit convention)."""
    sharpe_col = f"{period}_sharpe_pp"
    trials = table[sharpe_col].dropna().to_numpy()
    primary_row = table.loc[table.cell_id == primary_cell_id].iloc[0]
    sr_hat = primary_row[sharpe_col]
    key = "returns_is" if period == "is" else "returns_oos1"
    r = returns_by_cell[primary_cell_id]
    if period == "is":
        r = r.loc[:config.IS_END]
    else:
        r = r.loc[config.OOS1_START:]
    r = r.dropna()
    n_obs = len(r)
    skew = r.skew() if n_obs > 2 else 0.0
    kurt = r.kurtosis() + 3.0 if n_obs > 2 else 3.0  # pandas kurtosis is excess; DSR wants Pearson convention
    return metrics.deflated_sharpe_ratio(sr_hat, n_obs, trials, skew=skew, kurtosis=kurt)
