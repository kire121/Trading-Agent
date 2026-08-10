"""Weekly walk-forward backtest engine.

Brief: "Berakna q fredag stangning; exekvera manuellt mandag stangning
(primar; mandag oppning som variant) ... Full ersattning av portfoljen
varje vecka; inga stoppar, ingen diskretion."

Week i's weights are decided using data through Friday close f_i (causal:
signal.py's rolling windows never see past f_i), then entered at the first
trading day after f_i -- at that day's CLOSE for the primary convention, or
its OPEN for the variant. The position is held, unchanged, until the
following week's entry point (i.e. exit_i = entry_{i+1}): no stops, no
discretion, matching the brief exactly.

A name that stops trading mid-holding-period (delisted/acquired) does not
silently drop out of the return calculation -- its exit price is the last
available print at or before the intended exit date, which is exactly the
delisting return the point-in-time universe construction in universe.py/
data.py was built to capture. A name that has already gone fully NaN by
the entry date (never traded, or delisted before this week even starts)
is filtered out of the eligible set *before* weights are constructed
(matching Oglegrinden's fix for the "filter after weights are built"
adversarial-review bug: post-hoc filtering can leave gross exposure
silently short of target instead of properly renormalizing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vridmomentet.config import EXECUTION_VARIANTS, PRIMARY_EXECUTION, CostModel, PortfolioParams
from vridmomentet.data import Panel
from vridmomentet.portfolio import WeeklyPortfolio, build_target_weights, turnover
from vridmomentet.signal import SignalResult


def weekly_decision_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO calendar week present in `calendar` --
    holiday-robust "Friday close" (if Friday is a holiday, the preceding
    Thursday is that week's decision date).
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    iso = cal.isocalendar()
    key = pd.MultiIndex.from_arrays([iso["year"].values, iso["week"].values])
    s = pd.Series(cal, index=key)
    last_per_week = s.groupby(level=[0, 1]).last()
    return pd.DatetimeIndex(sorted(last_per_week.values))


def next_trading_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = calendar.searchsorted(date, side="right")
    if pos >= len(calendar):
        return None
    return calendar[pos]


def _last_valid_price_in_range(price_df: pd.DataFrame, names: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """For each name, the last non-NaN price at or before `end`, restricted
    to the (start, end] window -- captures a mid-week delisting exit
    without unboundedly forward-filling stale data.
    """
    window = price_df.loc[start:end, names]
    return window.ffill().iloc[-1]


@dataclass
class BacktestResult:
    weekly_gross_returns: pd.Series
    weekly_costs: pd.Series
    weekly_returns: pd.Series           # net of costs
    weekly_turnover: pd.Series
    n_long: pd.Series
    n_short: pd.Series
    decision_dates: pd.DatetimeIndex
    entry_dates: pd.Series
    exit_dates: pd.Series
    weights_by_date: dict = field(default_factory=dict, repr=False)
    longs_by_date: dict = field(default_factory=dict, repr=False)
    shorts_by_date: dict = field(default_factory=dict, repr=False)
    execution: str = PRIMARY_EXECUTION
    params: PortfolioParams = field(default_factory=PortfolioParams)


def _entry_exit_price_frame(panel: Panel, execution: str) -> pd.DataFrame:
    if execution == "monday_close":
        return panel.adj_close
    if execution == "monday_open":
        return panel.adj_open
    raise ValueError(f"unknown execution convention {execution!r}, must be one of {EXECUTION_VARIANTS}")


def run_backtest(
    panel: Panel,
    signal: SignalResult,
    decision_dates: pd.DatetimeIndex,
    portfolio_params: PortfolioParams = PortfolioParams(),
    cost_model: CostModel = CostModel(),
    execution: str = PRIMARY_EXECUTION,
    price_min: float = 5.0,
    adv_min: float = 20_000_000.0,
) -> BacktestResult:
    price_frame = _entry_exit_price_frame(panel, execution)
    calendar = panel.dates

    rows = []
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    longs_by_date: dict[pd.Timestamp, list[str]] = {}
    shorts_by_date: dict[pd.Timestamp, list[str]] = {}
    prev_weights = pd.Series(dtype=float)

    valid_decisions = [d for d in decision_dates if d in signal.s.index]
    for i, f_i in enumerate(valid_decisions):
        entry_date = next_trading_day(calendar, f_i)
        next_f = valid_decisions[i + 1] if i + 1 < len(valid_decisions) else None
        exit_date = next_trading_day(calendar, next_f) if next_f is not None else None
        if entry_date is None or exit_date is None:
            continue

        eligible = panel.eligible_on(f_i, price_min=price_min, adv_min=adv_min)
        s_row = signal.s.loc[f_i].reindex(eligible).dropna()
        # Restrict to names with a usable entry AND exit price *before*
        # constructing weights (see module docstring).
        entry_px = price_frame.loc[entry_date, s_row.index] if entry_date in price_frame.index else pd.Series(dtype=float)
        tradable = entry_px[entry_px.notna() & (entry_px > 0)].index
        s_row = s_row.reindex(tradable).dropna()

        vol_row = panel.vol60.loc[f_i].reindex(s_row.index)
        port: WeeklyPortfolio = build_target_weights(s_row, vol_row, portfolio_params)

        if port.weights.empty:
            rows.append(dict(decision=f_i, entry=entry_date, exit=exit_date, gross_return=0.0, turnover=0.0, n_long=0, n_short=0))
            weights_by_date[f_i] = port.weights
            longs_by_date[f_i] = []
            shorts_by_date[f_i] = []
            prev_weights = port.weights
            continue

        names = list(port.weights.index)
        entry_p = price_frame.loc[entry_date, names]
        exit_p = _last_valid_price_in_range(price_frame, names, entry_date, exit_date)
        name_return = (exit_p / entry_p) - 1.0
        gross_return = float((port.weights * name_return).sum())

        wk_turnover = turnover(prev_weights, port.weights)

        rows.append(dict(
            decision=f_i, entry=entry_date, exit=exit_date, gross_return=gross_return,
            turnover=wk_turnover, n_long=len(port.long_names), n_short=len(port.short_names),
        ))
        weights_by_date[f_i] = port.weights
        longs_by_date[f_i] = port.long_names
        shorts_by_date[f_i] = port.short_names
        prev_weights = port.weights

    table = pd.DataFrame(rows).set_index("decision")
    # portfolio.turnover() already sums both legs of trading (each exiting
    # name's |prev_w| and each entering name's |new_w|), so the ONE-WAY
    # rate is what prices it -- see CostModel.one_way_bps()'s docstring.
    cost_rate = cost_model.one_way_bps() / 10_000.0
    weekly_costs = table["turnover"] * cost_rate
    weekly_net = table["gross_return"] - weekly_costs

    return BacktestResult(
        weekly_gross_returns=table["gross_return"],
        weekly_costs=weekly_costs,
        weekly_returns=weekly_net,
        weekly_turnover=table["turnover"],
        n_long=table["n_long"],
        n_short=table["n_short"],
        decision_dates=pd.DatetimeIndex(table.index),
        entry_dates=table["entry"],
        exit_dates=table["exit"],
        weights_by_date=weights_by_date,
        longs_by_date=longs_by_date,
        shorts_by_date=shorts_by_date,
        execution=execution,
        params=portfolio_params,
    )


def benchmark_weekly_returns(price_series: pd.Series, calendar: pd.DatetimeIndex,
                              decision_dates: pd.DatetimeIndex, execution: str = PRIMARY_EXECUTION) -> pd.Series:
    """Same entry/exit timing convention as the strategy, applied to a single
    benchmark price series (e.g. SPY), for apples-to-apples beta/correlation.
    """
    valid_decisions = list(decision_dates)  # per-week entry/exit availability is checked in the loop below
    out = {}
    for i, f_i in enumerate(valid_decisions):
        entry_date = next_trading_day(calendar, f_i)
        next_f = valid_decisions[i + 1] if i + 1 < len(valid_decisions) else None
        exit_date = next_trading_day(calendar, next_f) if next_f is not None else None
        if entry_date is None or exit_date is None:
            continue
        if entry_date not in price_series.index:
            continue
        window = price_series.loc[entry_date:exit_date].ffill()
        if window.empty or pd.isna(window.iloc[0]) or window.iloc[0] == 0:
            continue
        out[f_i] = float(window.iloc[-1] / window.iloc[0] - 1.0)
    return pd.Series(out)
