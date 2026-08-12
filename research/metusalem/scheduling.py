"""Weekly rebalance-timing helpers.

Provenance (rule 2): `friday_lag_apply`/`week_end_values` are ported near-
verbatim from research/smittotalet/scheduling.py (branch claude/smittotalet-
portfolio-overlay-0bl1sh, commit a67df1b), generalized to take the weekday
from this package's own config instead of importing smittotalet's. This is
the base book's own tested rebalance convention (spec Sec.5 "Basbokens
testade exekveringskonvention ar auktoritativ") -- used UNCHANGED for the
base book itself and for the primary (exec_lag=1) tilt cell.

`extra_day_lag_apply` is NEW: Metusalem's own exekveringslag-robustness cell
(exec_lag=2) has no precedent anywhere in the repo (grep confirmed no other
branch parametrizes execution lag beyond the single Friday->Monday
convention baked into friday_lag_apply/EXECUTION_LAG_WEEKS-style code in
Smittotalet/Runraden). See its docstring and AVVIKELSER.md for the exact
operationalization chosen.
"""
import pandas as pd

from . import config


def friday_lag_apply(obj, weekday: str = config.REBALANCE_WEEKDAY):
    """obj: pd.Series or pd.DataFrame, daily-indexed. Returns the same shape,
    where every day within ISO week W carries the value observed at week
    (W-1)'s Friday close. This is the base book's own frozen cadence and the
    PRIMARY (exec_lag=1) tilt cadence -- signal at Friday close, in effect
    from the following week's Monday (spec Sec.5 "lag 1 dag, primar")."""
    week_period = obj.index.to_period(f"W-{weekday}")
    week_end_date = pd.Series(obj.index, index=week_period).groupby(level=0).last()
    at_week_end = obj.loc[week_end_date.values].copy()
    at_week_end.index = week_end_date.index
    applied = at_week_end.shift(1)
    daily = applied.loc[week_period]
    daily.index = obj.index
    return daily


def week_end_values(obj, weekday: str = config.REBALANCE_WEEKDAY):
    """The value sampled at each week's Friday close (no lag), indexed by
    weekly PeriodIndex."""
    week_period = obj.index.to_period(f"W-{weekday}")
    week_end_date = pd.Series(obj.index, index=week_period).groupby(level=0).last()
    at_week_end = obj.loc[week_end_date.values].copy()
    at_week_end.index = week_end_date.index
    return at_week_end


def weekly_score_to_daily(weekly_score, daily_index: pd.DatetimeIndex, extra_lag_days: int = 0,
                           weekday: str = config.REBALANCE_WEEKDAY):
    """Broadcasts an ALREADY-weekly (PeriodIndex, one row per week, value
    observed at that week's Friday close, no lag yet -- e.g. survival_trend.
    age_panel's output) score onto a daily index, applying the same
    Friday-close -> following-Monday lag as friday_lag_apply (extra_lag_days
    =0, the exec_lag=1 primary cell), plus `extra_lag_days` additional
    trading-day rows of delay for the exec_lag=2 robustness cell -- the
    weekly-input analogue of friday_lag_apply/extra_day_lag_apply, needed
    because age/P/m are natively weekly quantities, not daily ones re-
    sampled at week-end like the base book's own raw_signal."""
    daily_periods = daily_index.to_period(f"W-{weekday}")
    shifted = weekly_score.shift(1)
    daily = shifted.reindex(daily_periods)
    daily.index = daily_index
    if extra_lag_days:
        daily = daily.shift(extra_lag_days)
    return daily


def extra_day_lag_apply(obj, weekday: str = config.REBALANCE_WEEKDAY):
    """The exec_lag=2 robustness cell (spec Sec.5 "lag 2 som robusthetscell"):
    ONE additional trading day of delay on top of friday_lag_apply's own
    Friday-close -> Monday cadence, applied ONLY to the tilt multiplier `m`
    (never to the frozen base book -- spec Sec.5 "Basbokens parametrar ar
    frysta och raknas inte som frihetsgrader. Inga andra reglage existerar":
    T0 has a single, grid-invariant definition, so exec_lag can only be a
    tilt-side dial). Mechanically: take friday_lag_apply's own daily output
    (already effective from Monday) and shift it forward by one additional
    ROW along the object's own daily trading-calendar index -- so Monday of
    week W still carries the PRIOR week's multiplier, and the new week's
    multiplier only takes effect from Tuesday. `.shift(1)` on the daily
    index (not the weekly one) is exactly "one more trading day", matching
    "exekvering nasta handelsdags stangning (lag ... dagar)" read literally
    as trading-day count. See AVVIKELSER.md for the full derivation of why
    this -- and not re-lagging the base book itself -- is the operationalization
    chosen for a spec sentence that is textually about the whole portfolio's
    rebalance cadence but is pinned to be tilt-only by the Parametrar
    paragraph and by T0's single definition."""
    primary = friday_lag_apply(obj, weekday=weekday)
    return primary.shift(1)
