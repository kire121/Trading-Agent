"""Steg 0c: clock-choice oracle (rearrangement cap), adapted per spec §1.3
("Orakelmaskinen backtest.oracle_cap_test (rearrangement-tak) -- anpassas
till klockvals-orakel (§8 Steg 0c)").

Reuses lib.orakel.rearrangement_oracle (the primitive itself: reassigns the
SAME multiset of `values` to the time index of `target` in the order that
maximizes sum(values*target) -- a perfect-timing ceiling, not a tradeable
variant). That primitive is verbatim-migrated from
research/smittotalet/backtest.py::oracle_g; smittotalet's own WRAPPER
(oracle_cap_test) was deliberately NOT migrated to lib/ because it is
strategy-specific policy (its own hardcoded 0.15 SR threshold, and a
MULTIPLICATIVE tilt-on-base-book composition: oracle_returns = base_r * g).

Timglaset is additive by construction (spec §0: "Det statistiska objektet
är overlayn -- skillnadsserien mot en kalendertvilling"), not a
multiplicative tilt, so the wrapper here composes ADDITIVELY instead:
values = the overlay's own realized weekly returns (op - calendar),
target = the calendar twin's (T1) own weekly returns, oracle_returns =
T1 + rearrangement_oracle(overlay, T1). This is the adaptation §1.3 asks
for; the exact composition rule is not given verbatim by the spec (unlike
opclock's byte-complete §13 spec) and is logged as an interpretation choice
in AVVIKELSER.md, using spec's own explicit numeric threshold (T1 + 0.40 SR)
as the pass/fail criterion rather than smittotalet's 0.15.
"""
import numpy as np
import pandas as pd

from lib.orakel import rearrangement_oracle

from . import config


def weekly_compound(daily_returns: pd.Series) -> pd.Series:
    """Compound daily net returns within each ISO (Mon-Fri, W-FRI) week --
    matches research/smittotalet/backtest.py::weekly_return exactly (same
    provenance as rearrangement_oracle itself)."""
    week_period = daily_returns.index.to_period(f"W-{config.REBALANCE_WEEKDAY}")
    return (1.0 + daily_returns.fillna(0.0)).groupby(week_period).prod() - 1.0


def weekly_sharpe(weekly_returns: pd.Series) -> float:
    r = weekly_returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(config.WEEKS_YEAR))


def clock_oracle_test(op_daily_returns: pd.Series, cal_daily_returns: pd.Series) -> dict:
    """Steg 0c: is there rearrangement-inequality headroom, given the
    overlay's OWN realized weekly-return distribution, for a perfectly-timed
    version of it to beat T1 by at least the spec's threshold?"""
    op_week = weekly_compound(op_daily_returns)
    cal_week = weekly_compound(cal_daily_returns)
    overlay_week = (op_week - cal_week).dropna()

    oracle_overlay = rearrangement_oracle(overlay_week, cal_week)
    common = pd.concat([cal_week, oracle_overlay], axis=1, keys=["cal", "oracle_overlay"]).dropna()
    oracle_returns = common["cal"] + common["oracle_overlay"]

    t1_sr = weekly_sharpe(common["cal"])
    oracle_sr = weekly_sharpe(oracle_returns)
    increment = oracle_sr - t1_sr
    passes = bool(np.isfinite(increment) and increment >= config.STEG0C_MIN_ORACLE_SR_INCREMENT_OVER_T1)
    return {
        "t1_sr": t1_sr, "oracle_sr": oracle_sr, "increment": increment,
        "threshold": config.STEG0C_MIN_ORACLE_SR_INCREMENT_OVER_T1, "passes": passes,
    }
