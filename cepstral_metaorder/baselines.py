"""
Three "twin" alternative signals the main cepstral score has to beat net of
costs, per the pre-registered rejection criterion "loses to any twin -> dead".
The spec names them tersely; this module pins down one concrete, defensible
construction for each (documented per-function) since the one-line spec names
underdetermine the exact formula:

  (a) turnover_z    -- abnormal turnover, WITHOUT any periodicity structure,
                        directed by plain recent trend. Tests whether "this
                        name is unusually active lately" is enough on its own.
  (b) unmasked_flow -- the SAME cepstral entry/exit gate (S_bar, tau*) as the
                        main strategy, but direction computed from ALL minutes
                        instead of just the comb-filter's burst-phase minutes.
                        Isolates whether the comb mask's minute-selection adds
                        anything over plain signed order flow.
  (c) reversal_5d   -- classic short-horizon reversal: gate = magnitude of the
                        past 5-day move, direction = fade it. Guards against
                        the main signal being repackaged reversal.

twin_is_alive() implements the "twin-liveness assertion" (Dammluckan-lesson):
a twin that is degenerate (near-constant score, direction collapsed to one
sign or to ~0 everywhere) must not be allowed to silently validate the main
signal by "losing" to it -- a broken comparator always loses. Only a twin
that passes liveness counts in the rejection criterion.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import signal as sig
from .config import SPEC


def _cross_sectional_z_panel(raw_by_symbol: Dict[str, pd.Series], min_cross_section: int = 10) -> Dict[str, pd.Series]:
    """Same median/MAD standardization as signal.py, for a scalar-per-day
    (not per-quefrency) series -- used by the daily-bar-only twins."""
    all_dates = sorted(set().union(*[s.index for s in raw_by_symbol.values()]))
    symbols = list(raw_by_symbol.keys())
    aligned = pd.DataFrame({s: raw_by_symbol[s].reindex(all_dates) for s in symbols})
    out = pd.DataFrame(np.nan, index=all_dates, columns=symbols)
    for date in all_dates:
        row = aligned.loc[date]
        valid = row.dropna()
        if len(valid) < min_cross_section:
            continue
        med = valid.median()
        mad = (valid - med).abs().median()
        if mad == 0 or np.isnan(mad):
            continue
        out.loc[date, valid.index] = (valid - med) / mad
    return {s: out[s] for s in symbols}


# ---------------------------------------------------------------------------
# (a) abnormal turnover, no periodicity, trend-following direction
# ---------------------------------------------------------------------------

def turnover_twin_raw(enriched_eod_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Per symbol: score = cross-sectional z of the 5-day mean of relative
    turnover (dollar volume / ADV63); direction = sign of the trailing 5-day
    return. Indexed by date to line up with the main signal's daily index."""
    turnover_raw = {}
    for sym, df in enriched_eod_by_symbol.items():
        s = pd.Series(df["turnover_rel"].values, index=pd.Index(df["date"], name="date"))
        turnover_raw[sym] = s.rolling(SPEC.cepstrum.slicing_score_avg_days,
                                       min_periods=SPEC.cepstrum.slicing_score_avg_days).mean()

    z = _cross_sectional_z_panel(turnover_raw)

    out = {}
    for sym, df in enriched_eod_by_symbol.items():
        ret5 = pd.Series(df["ret_5d"].values, index=pd.Index(df["date"], name="date"))
        frame = pd.DataFrame({"S_bar": z[sym]})
        frame["D"] = np.sign(ret5.reindex(frame.index))
        out[sym] = frame
    return out


# ---------------------------------------------------------------------------
# (b) same cepstral gate, unmasked signed volume imbalance
# ---------------------------------------------------------------------------

def unmasked_flow_twin(main_signal_frame: pd.DataFrame, vol_wide: pd.DataFrame, ret_w: pd.DataFrame,
                        window: int = SPEC.cepstrum.direction_window_days) -> pd.DataFrame:
    """Reuses the main strategy's own S_bar/tau_star_window (identical entry/
    exit gate) but recomputes direction over ALL minutes in the window, i.e.
    burst_phases = every phase, mask always on."""
    dates = list(main_signal_frame.index)
    D_naive = pd.Series(np.nan, index=dates)
    for i in range(window - 1, len(dates)):
        d = dates[i]
        if d not in vol_wide.index:
            continue
        window_dates = [dt for dt in dates[i - window + 1: i + 1] if dt in vol_wide.index]
        if len(window_dates) < window:
            continue
        v = vol_wide.loc[window_dates].values
        r = ret_w.loc[window_dates].values
        weight = np.nan_to_num(v, nan=0.0)
        denom = weight.sum()
        if denom > 0:
            D_naive.loc[d] = float(np.nansum(np.sign(r) * weight) / denom)
    return pd.DataFrame({"S_bar": main_signal_frame["S_bar"], "D": D_naive})


# ---------------------------------------------------------------------------
# (c) 5-day reversal
# ---------------------------------------------------------------------------

def reversal_twin_raw(enriched_eod_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """score = cross-sectional z of |5-day return| (how much the name has
    moved), direction = fade it (contrarian)."""
    abs_ret5_raw = {}
    for sym, df in enriched_eod_by_symbol.items():
        s = pd.Series(df["ret_5d"].abs().values, index=pd.Index(df["date"], name="date"))
        abs_ret5_raw[sym] = s

    z = _cross_sectional_z_panel(abs_ret5_raw)

    out = {}
    for sym, df in enriched_eod_by_symbol.items():
        ret5 = pd.Series(df["ret_5d"].values, index=pd.Index(df["date"], name="date"))
        frame = pd.DataFrame({"S_bar": z[sym]})
        frame["D"] = -np.sign(ret5.reindex(frame.index))
        out[sym] = frame
    return out


# ---------------------------------------------------------------------------
# Twin-liveness assertion
# ---------------------------------------------------------------------------

def twin_is_alive(twin_frame: pd.DataFrame, min_coverage: float = 0.5,
                   min_score_dispersion: float = 1e-6, min_direction_sign_balance: float = 0.05) -> dict:
    """A twin must be minimally non-degenerate before a loss to it is allowed
    to count against the main signal. Checks:
      - coverage: fraction of days with a defined score AND direction
      - score dispersion: cross-time std of S_bar is not ~0 (not a constant)
      - direction is not collapsed onto a single sign (or all-zero): the
        minority-sign share must exceed min_direction_sign_balance
    Returns a dict with the individual checks and an overall 'alive' bool --
    every check is reported, not just the verdict, since a rejected/accepted
    twin needs to be auditable, not a black box.
    """
    n = len(twin_frame)
    defined = twin_frame[["S_bar", "D"]].notna().all(axis=1)
    coverage = defined.mean() if n else 0.0

    score_std = twin_frame.loc[defined, "S_bar"].std() if defined.any() else 0.0

    d_vals = twin_frame.loc[defined, "D"].dropna()
    if len(d_vals) == 0:
        sign_balance = 0.0
    else:
        pos = (d_vals > 0).mean()
        neg = (d_vals < 0).mean()
        sign_balance = min(pos, neg)

    checks = {
        "coverage": float(coverage),
        "coverage_ok": bool(coverage >= min_coverage),
        "score_std": float(score_std) if score_std == score_std else 0.0,
        "score_dispersion_ok": bool(score_std >= min_score_dispersion),
        "direction_sign_balance": float(sign_balance),
        "direction_balance_ok": bool(sign_balance >= min_direction_sign_balance),
    }
    checks["alive"] = checks["coverage_ok"] and checks["score_dispersion_ok"] and checks["direction_balance_ok"]
    return checks
