import numpy as np
import pandas as pd
import pytest

from research.omori import backtest, config, data, events


def _flat_panel(n=300, tickers=("SPIKE", "QUIET", "OTHER"), base_vol=1_000_000.0, seed=0):
    """Deterministic (tiny-noise) panel: near-constant price/volume so no
    spurious candidate events fire from noise alone."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    price = pd.DataFrame(100.0, index=idx, columns=tickers)
    price += rng.normal(0, 1e-4, size=(n, len(tickers))).cumsum(axis=0)  # negligible drift/noise
    volume = pd.DataFrame(base_vol, index=idx, columns=tickers)
    volume *= (1.0 + rng.normal(0, 1e-4, size=(n, len(tickers))))

    panel = data.Panel.__new__(data.Panel)
    panel.label = "synthetic"
    panel.raw = {
        "open": price.copy(), "high": price * 1.001, "low": price * 0.999,
        "close": price, "adjusted_close": price, "volume": volume,
    }
    panel.tickers = list(tickers)
    panel.index = idx
    return panel


def _inject_event(panel, ticker, t0_idx, direction=-1.0, ret_mag=0.08, vol_mult=25.0, decay_days=15):
    """Injects a volume-z/return event at t0_idx and a roughly Omori-shaped
    excess-volume decay over the following `decay_days`."""
    close = panel.raw["close"]
    volume = panel.raw["volume"]
    prev_close = close[ticker].iloc[t0_idx - 1]
    new_close = prev_close * (1 + direction * ret_mag)
    shift_factor = new_close / prev_close
    # carry the price shift FORWARD for every subsequent day (not just the
    # event day itself), or day t0+1's return would show an artificial
    # reversal back to the old baseline and spuriously qualify as its own
    # event.
    idx_tail = panel.index[t0_idx:]
    for field in ("close", "adjusted_close", "open", "high", "low"):
        panel.raw[field].loc[idx_tail, ticker] = panel.raw[field].loc[idx_tail, ticker] * shift_factor
    panel.raw["high"].loc[panel.index[t0_idx], ticker] = max(new_close, prev_close)
    panel.raw["low"].loc[panel.index[t0_idx], ticker] = min(new_close, prev_close)
    base = volume[ticker].iloc[t0_idx - 1]
    volume.loc[panel.index[t0_idx], ticker] = base * vol_mult
    for d in range(1, decay_days + 1):
        i = t0_idx + d
        if i >= len(panel.index):
            break
        excess_mult = 1.0 + (vol_mult - 1.0) * (d ** -0.9)
        volume.loc[panel.index[i], ticker] = base * excess_mult
    return t0_idx


@pytest.fixture
def priors_dict():
    return {"global": 0.7, "per_instrument": {}}


class TestBacktestMechanics:
    def test_no_events_on_flat_panel(self, priors_dict):
        panel = _flat_panel()
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7)
        assert len(res.closed_events) == 0

    def test_single_injected_event_produces_one_trade(self, priors_dict):
        panel = _flat_panel()
        t0 = _inject_event(panel, "SPIKE", 150)
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7)
        assert len(res.closed_events) == 1
        ev = res.closed_events[0]
        assert ev.ticker == "SPIKE"
        assert ev.t0_idx == t0

    def test_entry_is_exactly_t0_plus_1(self, priors_dict):
        panel = _flat_panel()
        t0 = _inject_event(panel, "SPIKE", 150)
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7)
        assert res.closed_events[0].entry_idx == t0 + 1

    def test_direction_matches_sign_of_r0(self, priors_dict):
        panel = _flat_panel()
        _inject_event(panel, "SPIKE", 150, direction=-1.0)
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7)
        assert res.closed_events[0].direction == -1.0

        panel2 = _flat_panel()
        _inject_event(panel2, "SPIKE", 150, direction=1.0)
        res2 = backtest.run_backtest(panel2, priors_dict, p_star=0.7)
        assert res2.closed_events[0].direction == 1.0

    def test_holding_days_within_floor_and_cap(self, priors_dict):
        panel = _flat_panel()
        _inject_event(panel, "SPIKE", 150)
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7, tau_cap=20)
        hd = res.closed_events[0].holding_days
        assert 1 <= hd <= 20 + 1  # +1 for entry-day-inclusive counting

    def test_costs_reduce_net_return_below_gross(self, priors_dict):
        panel = _flat_panel()
        _inject_event(panel, "SPIKE", 150)
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7, apply_costs=True)
        ev = res.closed_events[0]
        if ev.weight != 0:
            assert ev.net_return < ev.gross_return

    def test_no_costs_matches_gross_return(self, priors_dict):
        panel = _flat_panel()
        _inject_event(panel, "SPIKE", 150)
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7, apply_costs=False)
        ev = res.closed_events[0]
        assert ev.net_return == pytest.approx(ev.gross_return)

    def test_open_instrument_dedup_suppresses_second_event(self, priors_dict):
        panel = _flat_panel()
        _inject_event(panel, "SPIKE", 150, decay_days=25)  # keeps position open a while
        _inject_event(panel, "SPIKE", 152, decay_days=5)   # would-be 2nd event, same ticker, still open
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7)
        # only the first event's position should have opened (second candidate
        # is suppressed by the open-instrument rule)
        assert len(res.closed_events) == 1
        assert res.closed_events[0].t0_idx == 150

    def test_hard_stop_triggers_on_severe_adverse_move(self, priors_dict):
        panel = _flat_panel()
        t0 = _inject_event(panel, "SPIKE", 150, direction=1.0)  # long position
        # crash the price hard the day after entry -> should trip the hard stop
        entry_idx = t0 + 1
        crash_idx = entry_idx + 1
        crash_date = panel.index[crash_idx]
        panel.raw["close"].loc[crash_date, "SPIKE"] *= 0.5
        panel.raw["adjusted_close"].loc[crash_date, "SPIKE"] *= 0.5
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7)
        # the -50% crash is itself large enough to independently qualify as
        # a brand new event on that same day (once the original position
        # has already exited) -- so assert on the ORIGINAL event specifically
        # rather than assuming it is the only one.
        original = [e for e in res.closed_events if e.t0_idx == t0]
        assert len(original) == 1
        assert original[0].exit_reason == "hard_stop"

    def test_gross_cap_limits_new_entries(self, priors_dict):
        panel = _flat_panel(tickers=tuple(f"T{i}" for i in range(10)))
        for i, t in enumerate([f"T{j}" for j in range(10)]):
            _inject_event(panel, t, 150 + i, decay_days=25)  # staggered, overlapping opens
        res = backtest.run_backtest(panel, priors_dict, p_star=0.7, gross_cap=1.5)
        # at every point in time, gross exposure of currently-open positions
        # implied by entries must never have been allowed to exceed the cap
        # -- checked indirectly via each individual entry's own |weight|
        # never exceeding the cap outright.
        assert all(abs(e.weight) <= 1.5 + 1e-9 for e in res.closed_events)
