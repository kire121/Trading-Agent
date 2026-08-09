import numpy as np
import pandas as pd

from oglegrinden.data import Panel
from oglegrinden.signal import weekly_fridays
from oglegrinden.backtest import run_backtest, always_on_baseline, next_trading_day


def _make_synthetic_panel(n_tickers=20, n_days=400, seed=0, mean_reverting=True, include_spy=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    histories = {}
    tickers = [f"T{i:02d}" for i in range(n_tickers)] + (["SPY"] if include_spy else [])
    for tk in tickers:
        base = 100.0
        if mean_reverting and tk != "SPY":
            # weekly-frequency mean reversion: AR(1) on log-price deviations
            # from a slow-moving mean, so last week's losers tend to bounce.
            noise = rng.normal(0, 0.01, n_days)
            dev = np.zeros(n_days)
            for i in range(1, n_days):
                dev[i] = 0.7 * dev[i - 1] + noise[i]
            trend = np.linspace(0, 0.05, n_days)
            log_px = np.log(base) + trend + dev
        else:
            log_px = np.log(base) + np.cumsum(rng.normal(0.0002, 0.01, n_days))
        close = np.exp(log_px)
        open_ = close * (1 + rng.normal(0, 0.001, n_days))
        high = np.maximum(open_, close) * 1.002
        low = np.minimum(open_, close) * 0.998
        volume = rng.uniform(5_000_000, 8_000_000, n_days)  # ensures ADV > $20M at price ~100
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "adjclose": close},
            index=dates,
        )
        histories[tk] = df
    return Panel(histories)


class _FakeBundle:
    def __init__(self, gate_series, corr_window=60):
        self.corr_window = corr_window
        self.gates = {("L", "primary"): gate_series, ("L", "mirror"): ~gate_series}


def test_backtest_all_off_produces_zero_returns_and_zero_cost():
    panel = _make_synthetic_panel(n_tickers=20, n_days=300, seed=1)
    all_fridays = weekly_fridays(panel.close.index)
    gate_off = pd.Series(False, index=all_fridays)
    bundle = _FakeBundle(gate_off)

    result = run_backtest(panel, bundle, all_fridays, signal_name="L", direction="primary")
    assert (result.weekly_returns == 0.0).all()
    assert (result.turnover_cost == 0.0).all()
    assert (result.gross_exposure == 0.0).all()


def test_backtest_all_on_respects_cap_and_charges_costs():
    panel = _make_synthetic_panel(n_tickers=20, n_days=300, seed=2)
    all_fridays = weekly_fridays(panel.close.index)
    gate_on = pd.Series(True, index=all_fridays)
    bundle = _FakeBundle(gate_on)

    result = run_backtest(panel, bundle, all_fridays, signal_name="L", direction="primary", formation_days=5, cap=0.15)
    # First ~13 weeks are an expected ADV-eligibility warmup (eligible_on
    # requires 63 trading days of history); after that essentially every
    # week should trade.
    warmup_weeks = 13
    assert (result.n_names.iloc[warmup_weeks:] > 0).mean() > 0.95

    assert (result.gross_exposure[result.n_names > 0] <= 1.0 + 1e-6).all()
    assert (result.turnover_cost[result.n_names > 0] > 0).all()
    # net pnl = gross pnl - cost, exactly, every week
    assert np.allclose(
        result.weekly_returns.values,
        (result.gross_pnl - result.turnover_cost).values,
    )


def test_backtest_missing_open_price_does_not_leave_gross_exposure_short():
    """If one name has an unusable (NaN) entry price on a given trading
    day, it must be excluded *before* weights are constructed -- not
    dropped from an already-computed weight vector, which would silently
    leave that week's gross exposure below the 100% target (regression
    test for a real, if never fired on live data, bug found in
    adversarial review)."""
    panel = _make_synthetic_panel(n_tickers=20, n_days=300, seed=6)
    all_fridays = weekly_fridays(panel.close.index)
    gate_on = pd.Series(True, index=all_fridays)
    bundle = _FakeBundle(gate_on)

    # find a real entry date well past warmup and null out one ticker's
    # open price on it (close/adjclose stay valid, only the open used for
    # execution is broken).
    warmup_weeks = 13
    t = all_fridays[warmup_weeks + 5]
    entry_date = next_trading_day(panel.close.index, t)
    panel.open.loc[entry_date, "T03"] = np.nan

    result = run_backtest(panel, bundle, all_fridays, signal_name="L", direction="primary", formation_days=5, cap=0.15)
    n_at_t = result.n_names.loc[t]
    if n_at_t > 0:  # T03 was actually in that week's tradable set
        assert result.gross_exposure.loc[t] >= min(1.0, n_at_t * 0.15) - 1e-6


def test_backtest_reversal_extracts_positive_gross_pnl_on_mean_reverting_data():
    """The whole point of the strategy: on genuinely mean-reverting
    cross-sectional data, buying losers / selling winners should have
    positive expected gross (pre-cost) PnL."""
    panel = _make_synthetic_panel(n_tickers=20, n_days=500, seed=3, mean_reverting=True)
    all_fridays = weekly_fridays(panel.close.index)
    gate_on = pd.Series(True, index=all_fridays)
    bundle = _FakeBundle(gate_on)

    result = run_backtest(panel, bundle, all_fridays, signal_name="L", direction="primary", formation_days=5)
    active = result.gross_pnl[result.n_names > 0]
    assert active.mean() > 0


def test_gate_reindex_holds_state_between_computable_weeks():
    panel = _make_synthetic_panel(n_tickers=20, n_days=300, seed=4)
    all_fridays = weekly_fridays(panel.close.index)
    # sparse gate signal: only every 3rd Friday has a reading, always True
    sparse_index = all_fridays[::3]
    gate_sparse = pd.Series(True, index=sparse_index)
    bundle = _FakeBundle(gate_sparse)

    result = run_backtest(panel, bundle, all_fridays, signal_name="L", direction="primary")
    # gate should hold ON for the weeks between sparse readings too
    assert result.gate_state.sum() >= len(sparse_index)


def test_always_on_baseline_never_holds_cash_once_universe_is_eligible():
    panel = _make_synthetic_panel(n_tickers=20, n_days=300, seed=5)
    all_fridays = weekly_fridays(panel.close.index)
    result = always_on_baseline(panel, all_fridays, formation_days=5)
    warmup_weeks = 13
    assert (result.n_names.iloc[warmup_weeks:] > 0).mean() > 0.95


def test_next_trading_day():
    cal = pd.bdate_range("2021-01-04", periods=10)
    fri = pd.Timestamp("2021-01-08")
    nxt = next_trading_day(cal, fri)
    assert nxt == pd.Timestamp("2021-01-11")
    beyond = next_trading_day(cal, cal[-1])
    assert beyond is None
