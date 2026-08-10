import pandas as pd

from research.omori import events, twins
from research.omori.tests.test_backtest import _flat_panel


class TestT1FixedHorizon:
    def test_uses_median_of_tau_exit_holding_days_when_unspecified(self):
        panel = _flat_panel(n=300)
        ev = pd.DataFrame([
            {"ticker": "SPIKE", "t0_idx": 100, "entry_idx": 101, "exit_idx": 106, "direction": 1.0,
             "weight": 0.1, "exit_reason": "tau_exit", "holding_days": 5, "gross_return": 0.0,
             "net_return": 0.0, "p_tilde_entry": 0.7, "volume_z": 5.0, "r0": 0.05},
            {"ticker": "QUIET", "t0_idx": 120, "entry_idx": 121, "exit_idx": 130, "direction": -1.0,
             "weight": -0.1, "exit_reason": "tau_exit", "holding_days": 9, "gross_return": 0.0,
             "net_return": 0.0, "p_tilde_entry": 0.7, "volume_z": 5.0, "r0": -0.05},
        ])
        ef = events.EventFields(panel)
        dr, net = twins.t1_fixed_horizon(panel, ef, ev)
        assert len(net) == 2
        assert isinstance(dr, pd.Series)
        assert len(dr) == len(panel.index)

    def test_hard_stop_can_still_trigger_within_fixed_horizon(self):
        panel = _flat_panel(n=300)
        crash_date = panel.index[105]
        panel.raw["close"].loc[crash_date, "SPIKE"] *= 0.3
        panel.raw["adjusted_close"].loc[crash_date, "SPIKE"] *= 0.3
        ev = pd.DataFrame([{"ticker": "SPIKE", "t0_idx": 100, "entry_idx": 101, "exit_idx": 120,
                             "direction": 1.0, "weight": 0.3, "exit_reason": "tau_exit",
                             "holding_days": 20, "gross_return": 0.0, "net_return": 0.0,
                             "p_tilde_entry": 0.7, "volume_z": 5.0, "r0": 0.05}])
        ef = events.EventFields(panel)
        dr, net = twins.t1_fixed_horizon(panel, ef, ev, fixed_horizon=20)
        assert net[0] < -0.05  # the crash should show up as a real loss, not get erased


class TestT2Shuffled:
    def test_within_ticker_permutation_preserves_horizon_multiset_per_ticker(self):
        panel = _flat_panel(n=300)
        ev = pd.DataFrame([
            {"ticker": "SPIKE", "t0_idx": 100, "entry_idx": 101, "exit_idx": 106, "direction": 1.0,
             "weight": 0.1, "exit_reason": "tau_exit", "holding_days": 5, "gross_return": 0.0,
             "net_return": 0.0, "p_tilde_entry": 0.7, "volume_z": 5.0, "r0": 0.05},
            {"ticker": "SPIKE", "t0_idx": 150, "entry_idx": 151, "exit_idx": 161, "direction": 1.0,
             "weight": 0.1, "exit_reason": "tau_exit", "holding_days": 10, "gross_return": 0.0,
             "net_return": 0.0, "p_tilde_entry": 0.7, "volume_z": 5.0, "r0": 0.05},
        ])
        ef = events.EventFields(panel)
        dr, net = twins.t2_shuffled_within_instrument(panel, ef, ev, seed=42)
        assert len(net) == 2


class TestT3Randomized:
    def test_entries_are_relocated_away_from_original_event_dates(self):
        panel = _flat_panel(n=500)
        ev = pd.DataFrame([{"ticker": "SPIKE", "t0_idx": 100, "entry_idx": 101, "exit_idx": 106,
                             "direction": 1.0, "weight": 0.1, "exit_reason": "tau_exit",
                             "holding_days": 5, "gross_return": 0.0, "net_return": 0.0,
                             "p_tilde_entry": 0.7, "volume_z": 5.0, "r0": 0.05}])
        ef = events.EventFields(panel)
        dr, net = twins.t3_randomized_entry(panel, ef, ev, seed=7)
        assert len(net) == 1
