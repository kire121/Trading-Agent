"""Weekly walk-forward backtest engine.

Mechanics (per the strategy brief):
  - Friday close: observe the gate state (already computed in signal.py).
  - If ON: form the cross-sectional reversal portfolio from this week's
    (5d or 10d) formation returns, execute at the next trading day's open
    ("Monday open"), hold one week, full turnover at every rebalance.
  - If OFF: sit in cash.
  - Costs: 2bp/side commission + half the bid-ask spread, charged on both
    the entry and the exit of every week's position (full turnover is
    explicit in the brief, so no netting across weeks).

Execution price convention: adjusted-open is reconstructed as
`open * (adjclose / close)` -- Yahoo's chart API gives split/dividend
adjusted *close* but not adjusted *open*, so we carry the same-day
adjustment factor over to the open. Over a one-week holding period this is
an immaterial approximation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from oglegrinden.data import Panel
from oglegrinden.portfolio import formation_weights
from oglegrinden.signal import GateBundle
from oglegrinden.universe import MIN_ADV_USD, BENCHMARK


def adjusted_open(panel: Panel) -> pd.DataFrame:
    factor = (panel.adjclose / panel.close).replace([np.inf, -np.inf], np.nan)
    return panel.open * factor


def next_trading_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = calendar.searchsorted(date, side="right")
    if pos >= len(calendar):
        return None
    return calendar[pos]


@dataclass
class BacktestResult:
    weekly_returns: pd.Series  # net-of-cost simple return per decision Friday, indexed by signal date
    gross_pnl: pd.Series  # same, before transaction costs
    gross_exposure: pd.Series
    turnover_cost: pd.Series
    gate_state: pd.Series
    n_names: pd.Series
    entry_dates: pd.Series
    exit_dates: pd.Series
    params: dict = field(default_factory=dict)


def run_backtest(
    panel: Panel,
    gate_bundle: GateBundle,
    all_fridays: pd.DatetimeIndex,
    signal_name: str = "L",
    direction: str = "primary",
    formation_days: int = 5,
    cap: float = 0.15,
    winsor_k: float = 3.0,
    commission_bps: float = 2.0,
    half_spread_bps: float = 1.0,
    min_adv: float = MIN_ADV_USD,
    min_names_to_trade: int = 6,
) -> BacktestResult:
    """Run the full weekly gated cross-sectional reversal strategy.

    `all_fridays` should span the entire tradable history (not just the
    weeks where the topology signal was computable) so weeks with an
    unavailable signal correctly show up as a held-over gate state rather
    than being silently dropped from the equity curve.
    """
    gate_raw = gate_bundle.gates[(signal_name, direction)]
    gate_full = gate_raw.reindex(all_fridays).ffill().fillna(False)

    adj_open = adjusted_open(panel)
    trading_calendar = panel.close.index

    returns, gross_pnls, gross, cost, n_names, entries, exits = [], [], [], [], [], [], []
    cost_frac_per_side = (commission_bps + half_spread_bps) / 10_000.0

    def _append_flat(entry_date, exit_date):
        returns.append(0.0)
        gross_pnls.append(0.0)
        gross.append(0.0)
        cost.append(0.0)
        n_names.append(0)
        entries.append(entry_date)
        exits.append(exit_date)

    for i, t in enumerate(all_fridays):
        is_on = bool(gate_full.loc[t])
        entry_date = next_trading_day(trading_calendar, t)
        # next week's decision Friday, or (for the last week) None
        next_t = all_fridays[i + 1] if i + 1 < len(all_fridays) else None
        exit_date = next_trading_day(trading_calendar, next_t) if next_t is not None else None

        if not is_on or entry_date is None or exit_date is None:
            _append_flat(entry_date, exit_date)
            continue

        eligible = [tk for tk in panel.eligible_on(t, min_adv=min_adv) if tk != BENCHMARK]
        formation_window = panel.log_returns.loc[:t, eligible].tail(formation_days)
        formation_window = formation_window.dropna(axis=1, how="any")
        if formation_window.shape[1] < min_names_to_trade or formation_window.shape[0] < formation_days:
            _append_flat(entry_date, exit_date)
            continue

        f = formation_window.sum(axis=0)  # cumulative log return over the formation window

        # Restrict to names with a usable (adjusted) entry and exit price
        # *before* constructing weights, not after: winsorization,
        # demeaning, and capping must run over the actually-tradable set,
        # or dropping names post-hoc would silently leave gross exposure
        # below the 100% target and perturb dollar-neutrality for names
        # that happen to have a missing/invalid open price that week.
        entry_px_candidates = adj_open.loc[entry_date, f.index]
        exit_px_candidates = adj_open.loc[exit_date, f.index]
        tradable = entry_px_candidates.notna() & exit_px_candidates.notna() & (entry_px_candidates > 0)
        f = f[tradable]
        if f.shape[0] < min_names_to_trade:
            _append_flat(entry_date, exit_date)
            continue

        w = formation_weights(f, cap=cap, winsor_k=winsor_k)
        if w.empty:
            _append_flat(entry_date, exit_date)
            continue

        entry_px = adj_open.loc[entry_date, w.index]
        exit_px = adj_open.loc[exit_date, w.index]
        simple_ret = exit_px / entry_px - 1.0
        gross_exposure = w.abs().sum()
        gross_pnl = float((w * simple_ret).sum())
        week_cost = gross_exposure * 2.0 * cost_frac_per_side  # entry + exit, full turnover
        net_pnl = gross_pnl - week_cost

        returns.append(net_pnl)
        gross_pnls.append(gross_pnl)
        gross.append(gross_exposure)
        cost.append(week_cost)
        n_names.append(int(w.shape[0]))
        entries.append(entry_date)
        exits.append(exit_date)

    return BacktestResult(
        weekly_returns=pd.Series(returns, index=all_fridays, name="net_return"),
        gross_pnl=pd.Series(gross_pnls, index=all_fridays, name="gross_pnl"),
        gross_exposure=pd.Series(gross, index=all_fridays, name="gross_exposure"),
        turnover_cost=pd.Series(cost, index=all_fridays, name="cost"),
        gate_state=gate_full,
        n_names=pd.Series(n_names, index=all_fridays, name="n_names"),
        entry_dates=pd.Series(entries, index=all_fridays, name="entry_date"),
        exit_dates=pd.Series(exits, index=all_fridays, name="exit_date"),
        params=dict(
            signal_name=signal_name,
            direction=direction,
            formation_days=formation_days,
            cap=cap,
            winsor_k=winsor_k,
            commission_bps=commission_bps,
            half_spread_bps=half_spread_bps,
            corr_window=gate_bundle.corr_window,
        ),
    )


def always_on_baseline(
    panel: Panel,
    all_fridays: pd.DatetimeIndex,
    formation_days: int = 5,
    cap: float = 0.15,
    winsor_k: float = 3.0,
    commission_bps: float = 2.0,
    half_spread_bps: float = 1.0,
    min_adv: float = MIN_ADV_USD,
    min_names_to_trade: int = 6,
) -> BacktestResult:
    """Same reversal engine, but always ON (no topology gate) -- the
    baseline the gated strategy must beat on both Sharpe and drawdown.
    """
    always_on = pd.Series(True, index=all_fridays)

    class _FakeBundle:
        corr_window = None
        gates = {("always_on", "primary"): always_on}

    return run_backtest(
        panel,
        _FakeBundle(),
        all_fridays,
        signal_name="always_on",
        direction="primary",
        formation_days=formation_days,
        cap=cap,
        winsor_k=winsor_k,
        commission_bps=commission_bps,
        half_spread_bps=half_spread_bps,
        min_adv=min_adv,
        min_names_to_trade=min_names_to_trade,
    )


def benchmark_weekly_returns(panel: Panel, all_fridays: pd.DatetimeIndex, benchmark: str = BENCHMARK) -> pd.Series:
    """SPY (or other benchmark) weekly return on the exact same
    Monday-open-to-Monday-open convention used for the strategy legs, so
    beta-hedging and diversification stats are computed on a consistent
    basis with the strategy's own P&L.
    """
    adj_open = adjusted_open(panel)
    trading_calendar = panel.close.index
    rets = []
    for i, t in enumerate(all_fridays):
        entry_date = next_trading_day(trading_calendar, t)
        next_t = all_fridays[i + 1] if i + 1 < len(all_fridays) else None
        exit_date = next_trading_day(trading_calendar, next_t) if next_t is not None else None
        if entry_date is None or exit_date is None:
            rets.append(np.nan)
            continue
        e = adj_open.loc[entry_date, benchmark]
        x = adj_open.loc[exit_date, benchmark]
        rets.append(float(x / e - 1) if pd.notna(e) and pd.notna(x) and e > 0 else np.nan)
    return pd.Series(rets, index=all_fridays, name=f"{benchmark}_weekly_return")


def hedge_to_beta(
    weekly_returns: pd.Series,
    spy_weekly_returns: pd.Series,
    window: int = 26,
    hedge_cost_bps: float = 3.0,
) -> pd.Series:
    """Declared variant: hedge the realized residual beta of the reversal
    book against SPY using a trailing rolling-beta estimate (no
    look-ahead: beta_t is estimated from weeks strictly before t, then
    applied to week t's return). A small extra cost is charged on the
    hedge leg's turnover (|beta_t - beta_{t-1}| notional traded).
    """
    spy = spy_weekly_returns.reindex(weekly_returns.index)
    cov = weekly_returns.rolling(window).cov(spy)
    var = spy.rolling(window).var()
    beta = (cov / var).shift(1).fillna(0.0)
    hedge_pnl = -beta * spy
    hedge_turnover = beta.diff().abs().fillna(beta.abs())
    hedge_cost = hedge_turnover * (hedge_cost_bps / 10_000.0)
    return weekly_returns + hedge_pnl - hedge_cost
