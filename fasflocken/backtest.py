"""
Fasflocken (PH-1) -- weekly backtest engine.

Execution convention (spec: "Fredag stangning ... Exekvering mandag
oppning. Inga intraveckojusteringar."):

  * Signal computed using data through Friday close f_i (causal by
    construction -- see signals.py).
  * Weights decided at f_i are held for every trading day strictly after
    f_i through the *next* Friday f_{i+1} inclusive -- i.e. the following
    Monday-through-Friday, whatever the calendar actually contains
    between those two signal dates. This sidesteps hardcoding "Monday"
    against a calendar that may or may not have a holiday-shifted open.
  * Portfolio return within that window is the additive (log-return)
    approximation sum_etf w_etf * r_etf,day, summed across the window's
    trading days. Turnover cost is charged once, against the week's
    total return, on the weight change vs. the prior week.

Only ETFs from config.SECTOR_ETFS are traded; the underlying constituent
data only feeds the Z_s(t) signal (see pipeline.py).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fasflocken.config import (
    GICS_SECTORS,
    SECTOR_TO_ETF,
    SignalParams,
    CostModel,
    DEFAULT_COSTS,
    VOL_TARGET_ANN,
    MAX_GROSS,
    COV_WINDOW_DAYS,
    WEEKS_PER_YEAR,
)
from fasflocken.pipeline import compute_sector_signal, build_z_panel, SectorSignal
from fasflocken.portfolio import (
    HysteresisState,
    build_target_weights,
    eligible_sectors,
    trailing_cov_matrix,
    compute_turnover,
)
from fasflocken.universe import UniverseProvider


@dataclass
class BacktestResult:
    weekly_returns: pd.Series          # net of costs, indexed by signal (decision) Friday
    weekly_gross_returns: pd.Series    # before costs
    weights: pd.DataFrame              # decision Friday x ETF -> weight held the following week
    turnover: pd.Series                # sum(|Delta w|) per rebalance
    k_scale: pd.Series                 # vol-target leverage multiplier per rebalance
    longs: pd.Series                   # list[str] of long sectors per rebalance
    shorts: pd.Series                  # list[str] of short sectors per rebalance
    daily_returns: pd.Series           # net daily portfolio returns, for diagnostics
    sector_signals: dict[str, SectorSignal] = field(repr=False)
    z_panel: pd.DataFrame = field(repr=False)
    params: SignalParams = None
    cost_model: CostModel = None


def _week_windows(signal_dates: pd.DatetimeIndex, daily_dates: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """For each signal date f_i, the daily dates in (f_i, f_{i+1}] (or (f_i, end] for the last)."""
    windows = []
    for i in range(len(signal_dates)):
        start = signal_dates[i]
        end = signal_dates[i + 1] if i + 1 < len(signal_dates) else daily_dates.max()
        mask = (daily_dates > start) & (daily_dates <= end)
        windows.append(daily_dates[mask])
    return windows


def run_backtest(
    provider: UniverseProvider,
    start: _dt.date,
    end: _dt.date,
    params: SignalParams = SignalParams(),
    target_vol_ann: float = VOL_TARGET_ANN,
    max_gross: float = MAX_GROSS,
    cov_window_days: int = COV_WINDOW_DAYS,
    cost_model: CostModel = DEFAULT_COSTS,
    half_spread_bps: float | None = None,
    membership_refresh_days: int = 5,
    sectors: tuple[str, ...] = GICS_SECTORS,
    z_override: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the full weekly Fasflocken backtest over [start, end].

    z_override: if given, skip recomputing sector signals from the
    provider and use this (weekly_date x sector) Z panel directly -- used
    by the null-hypothesis bootstrap (stats.py) to replay the same
    portfolio construction / cost model against a resampled signal
    without re-running the (expensive) Hilbert pipeline.
    """
    daily_dates = provider.trading_calendar(start, end)

    if z_override is not None:
        z_panel = z_override
        sector_signals: dict[str, SectorSignal] = {}
    else:
        sector_signals = {
            s: compute_sector_signal(provider, s, start, end, params, membership_refresh_days)
            for s in sectors
        }
        z_panel = build_z_panel(sector_signals, twin=False)

    etfs = [SECTOR_TO_ETF[s] for s in sectors]
    etf_prices = provider.sector_etf_prices(start, end, etfs).reindex(index=daily_dates)
    etf_log_returns = np.log(etf_prices).diff()

    signal_dates = z_panel.index[(z_panel.index >= pd.Timestamp(start)) & (z_panel.index <= pd.Timestamp(end))]
    windows = _week_windows(signal_dates, daily_dates)

    cost_bps = cost_model.total_bps_per_side(half_spread_bps)
    cost_rate = cost_bps / 10_000.0

    state = HysteresisState()
    prev_weights = pd.Series(dtype=float)

    weekly_gross, weekly_net, turnover_hist, k_hist = [], [], [], []
    longs_hist, shorts_hist, weight_rows = [], [], []
    daily_ret_pieces = []

    for f_i, window in zip(signal_dates, windows):
        as_of = f_i.date()
        elig = eligible_sectors(as_of, sectors)
        z_row = z_panel.loc[f_i, elig] if elig else pd.Series(dtype=float)

        cov_matrix = trailing_cov_matrix(etf_log_returns, f_i, cov_window_days)

        try:
            state, final_weights, info = build_target_weights(
                z_row, state, params.n_legs, params.hysteresis_band, cov_matrix, target_vol_ann, max_gross, elig
            )
        except ValueError:
            final_weights = pd.Series(dtype=float)
            info = {"long_sectors": [], "short_sectors": [], "k": 0.0}

        turnover = compute_turnover(prev_weights, final_weights)
        fw = final_weights

        if len(window) == 0:
            week_gross_ret = 0.0
            day_rets = pd.Series(dtype=float)
        else:
            r = etf_log_returns.loc[window, fw.index.intersection(etf_log_returns.columns)]
            w_aligned = fw.reindex(r.columns, fill_value=0.0)
            day_rets = r.mul(w_aligned, axis=1).sum(axis=1)
            week_gross_ret = float(day_rets.sum())

        week_cost = turnover * cost_rate
        week_net_ret = week_gross_ret - week_cost

        if len(day_rets) > 0:
            day_rets = day_rets.copy()
            day_rets.iloc[0] -= week_cost
            daily_ret_pieces.append(day_rets)

        weekly_gross.append(week_gross_ret)
        weekly_net.append(week_net_ret)
        turnover_hist.append(turnover)
        k_hist.append(info["k"])
        longs_hist.append(list(info["long_sectors"]))
        shorts_hist.append(list(info["short_sectors"]))
        weight_rows.append(fw)

        prev_weights = final_weights

    weights_df = pd.DataFrame(weight_rows, index=signal_dates).fillna(0.0)
    daily_returns = pd.concat(daily_ret_pieces).sort_index() if daily_ret_pieces else pd.Series(dtype=float)

    return BacktestResult(
        weekly_returns=pd.Series(weekly_net, index=signal_dates, name="net_return"),
        weekly_gross_returns=pd.Series(weekly_gross, index=signal_dates, name="gross_return"),
        weights=weights_df,
        turnover=pd.Series(turnover_hist, index=signal_dates, name="turnover"),
        k_scale=pd.Series(k_hist, index=signal_dates, name="k"),
        longs=pd.Series(longs_hist, index=signal_dates, name="longs"),
        shorts=pd.Series(shorts_hist, index=signal_dates, name="shorts"),
        daily_returns=daily_returns,
        sector_signals=sector_signals,
        z_panel=z_panel,
        params=params,
        cost_model=cost_model,
    )


def annualized_sharpe(weekly_returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> float:
    r = weekly_returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_return(weekly_returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> float:
    r = weekly_returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float(r.mean() * periods_per_year)


def annualized_vol(weekly_returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> float:
    r = weekly_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def since(weekly_returns: pd.Series, eval_start: _dt.date | None) -> pd.Series:
    """Slice a return series to dates >= eval_start.

    Running with an early `start` (e.g. 2002 for burn-in, per
    config.SAMPLE_WINDOW) intentionally produces weeks of forced-zero
    return before enough Z-score history exists to trade. Those weeks are
    real (not a bug -- see run_backtest's docstring), but must be excluded
    before computing Sharpe/DSR/etc., or performance stats get diluted by
    however many burn-in weeks happened to be in the sample -- unevenly so
    across grid cells with different z_lookback_weeks. Pass eval_start=None
    to skip slicing (whole series, unchanged).
    """
    if eval_start is None:
        return weekly_returns
    return weekly_returns.loc[weekly_returns.index >= pd.Timestamp(eval_start)]


def max_drawdown(weekly_returns: pd.Series) -> float:
    r = weekly_returns.dropna()
    if len(r) == 0:
        return float("nan")
    cum = r.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    return float(dd.min())
