"""Steg 0b K0b.4 -- the oracle ceiling (spec Sec.10, Sec.12.7 note: "tvarsnitts-
variant av Smittotalets rearrangement-orakel").

Perfect foresight of REMAINING episode length (survival_trend.residual_life)
replaces the noisy observed-age signal in EXACTLY the same tilt/percentile/
solve pipeline (kappa=0.5, fixed): if even perfect knowledge of "how much
longer this trend will run" cannot clear +0.30 net-SR over T0 on IS-A, the
whole age-conditioning axis has no room and Metusalem is a total kill before
any real estimation is attempted.

This is NOT lib.orakel.rearrangement_oracle (a different concept entirely --
same-multiset time-reordering of a scalar series; see that module's own
docstring on the two unrelated "orakel" meanings in this repo's history).
This module also is not a rearrangement-inequality construction: it is the
tilt_weights pipeline fed with the oracle's own (non-tradable, look-ahead)
residual-life panel instead of the real, causal age panel -- reusing
survival_trend.tilt_weights verbatim, since "percentiltilt genom exakt
samma pipeline" is a literal, mechanical instruction, not a new formula.
"""
import numpy as np
import pandas as pd

from . import config
from . import survival_trend as st


def residual_life_panel(s_panel: pd.DataFrame) -> pd.DataFrame:
    """Converts survival_trend.residual_life's {(instrument,week): weeks} dict
    into a DataFrame shaped exactly like `s_panel` (weeks x instruments),
    NaN wherever residual life is undefined (censored episodes, or weeks
    outside any completed episode)."""
    episodes = st.extract_episodes(s_panel)
    rl = st.residual_life(episodes, freq=f"W-{config.REBALANCE_WEEKDAY}")
    out = pd.DataFrame(np.nan, index=s_panel.index, columns=s_panel.columns)
    for (inst, week), val in rl.items():
        if inst in out.columns and week in out.index:
            out.loc[week, inst] = val
    return out


def oracle_ceiling_test(panel, s_panel_weekly: pd.DataFrame, w_bas_raw_daily: pd.DataFrame,
                         is_start, is_end, one_way_bps: float, t0_book: dict) -> dict:
    """Builds the oracle book (kappa=0.5 fixed) and compares its net Sharpe
    to T0 on IS-A. `w_bas_raw_daily` and `s_panel_weekly` must come from the
    SAME underlying panel/window (caller's responsibility, matching Sec.10's
    "genom exakt samma pipeline")."""
    from lib import metrics as lib_metrics
    from . import signal_construction as sc

    rl_weekly = residual_life_panel(s_panel_weekly)
    # broadcast the weekly oracle score to the daily index basbok operates on
    # (same convention as tilt: the tilt multiplier m is a weekly quantity,
    # elementwise-multiplied onto the daily-cadence w_bas_raw).
    rl_daily = rl_weekly.copy()
    rl_daily.index = rl_daily.index.to_timestamp(how="end").normalize()
    rl_daily = rl_daily.reindex(w_bas_raw_daily.index, method="ffill")

    oracle_raw = st.tilt_weights(w_bas_raw_daily, rl_daily, kappa=config.KAPPA_PRIMARY)
    oracle_book = sc.solved_book(panel, oracle_raw, is_start, is_end, one_way_bps=one_way_bps)

    from . import basbok as _basbok  # weekly compounding before Sharpe -- see basbok.weekly_return
    sr_oracle = lib_metrics.sharpe_ratio(_basbok.weekly_return(oracle_book["returns"].loc[is_start:is_end]))
    sr_t0 = lib_metrics.sharpe_ratio(_basbok.weekly_return(t0_book["returns"].loc[is_start:is_end]))
    uplift = sr_oracle - sr_t0

    return {
        "sr_oracle": sr_oracle,
        "sr_t0": sr_t0,
        "uplift": uplift,
        "K0b_4": bool(np.isfinite(uplift) and uplift >= config.K0B4_MIN_ORACLE_SR_UPLIFT),
        "oracle_book": oracle_book,
    }
