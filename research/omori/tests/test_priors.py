import numpy as np

from research.omori import priors
from research.omori.tests.test_backtest import _flat_panel, _inject_event


class TestTerminalFits:
    def test_no_candidates_on_flat_panel(self):
        panel = _flat_panel()
        fits = priors.terminal_fits(panel)
        assert fits == []

    def test_injected_event_produces_one_fit(self):
        panel = _flat_panel()
        _inject_event(panel, "SPIKE", 150, decay_days=15)
        fits = priors.terminal_fits(panel)
        assert len(fits) == 1
        assert fits[0]["ticker"] == "SPIKE"
        assert fits[0]["identified"]
        assert fits[0]["p_hat"] > 0

    def test_insufficient_trailing_data_marked_unidentified(self):
        panel = _flat_panel(n=160)
        # t0=157 leaves only 2 trading days of history (n-1-t0_idx=2), below
        # MIN_POSITIVE_EXCESS_DAYS=4 -> must be unidentified (full shrinkage).
        _inject_event(panel, "SPIKE", 157, decay_days=15)
        fits = priors.terminal_fits(panel)
        assert len(fits) == 1
        assert not fits[0]["identified"]


class TestCalibratePriors:
    def test_global_and_per_instrument_fallback(self, tmp_path, monkeypatch):
        panel = _flat_panel(tickers=("A", "B", "C"))
        _inject_event(panel, "A", 150, decay_days=15)
        _inject_event(panel, "B", 160, decay_days=15)
        monkeypatch.setattr(priors, "OUTPUT_DIR", str(tmp_path))
        out = priors.calibrate_priors(panel, save=True)
        assert "global" in out
        assert out["per_instrument"]["A"] > 0
        # C has no events at all -> must fall back to the global prior
        assert out["per_instrument"]["C"] == out["global"]

    def test_prior_for_unknown_ticker_falls_back_to_global(self):
        priors_dict = {"global": 0.55, "per_instrument": {"A": 0.7}}
        assert priors.prior_for("ZZZ_NOT_IN_IS", priors_dict) == 0.55
        assert priors.prior_for("A", priors_dict) == 0.7
