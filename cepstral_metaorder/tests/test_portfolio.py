import numpy as np
import pandas as pd
import pytest

from cepstral_metaorder import portfolio as pf
from cepstral_metaorder import universe as uni
from cepstral_metaorder.config import SPEC


def _make_universe_panel(symbols, dates):
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({"date": d, "symbol": s, "close": 50.0, "adv": 1e9,
                         "eligible": True, "adv_rank": 1.0, "eligible_adv_rank": 1.0,
                         "in_universe": True})
    return pd.DataFrame(rows)


def _make_eod(dates, open_price=100.0, drift=0.0):
    return pd.DataFrame({"date": dates, "adj_open": [open_price * (1 + drift) ** i for i in range(len(dates))]})


def test_cap_and_redistribute_respects_cap_and_preserves_gross_when_feasible():
    raw = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0], index=list("ABCDE"))
    out = pf._cap_and_redistribute(raw, cap=0.30, target_gross=1.0)
    assert (out.abs() <= 0.30 + 1e-9).all()
    assert out.sum() == pytest.approx(1.0, abs=1e-6)


def test_rank_tilt_weights_gives_more_to_higher_score_same_side():
    score = pd.Series([1.0, 5.0, 3.0], index=["low", "high", "mid"])
    side = pd.Series([1, 1, 1], index=["low", "high", "mid"])
    w = pf._rank_tilt_weights(score, side, cap=0.5, target_gross_per_side=0.5)
    assert w["high"] > w["mid"] > w["low"]
    assert w.sum() == pytest.approx(0.5, abs=1e-6)


def test_entry_requires_both_score_percentile_and_direction_threshold():
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    symbols = [f"S{i}" for i in range(20)]
    panel = _make_universe_panel(symbols, dates)

    signal = {}
    for i, s in enumerate(symbols):
        S_bar = pd.Series(float(i), index=dates)  # rank purely by i -> S19 highest percentile
        D = pd.Series(0.05 if s != "S19" else 0.5, index=dates)  # only S19 clears |D|>=0.1
        signal[s] = pd.DataFrame({"S_bar": S_bar, "D": D})

    eod = {s: _make_eod(dates) for s in symbols}
    result = pf.run_backtest(signal, panel, eod)
    weights = result["weights"]

    # S19 is top-percentile score AND clears the direction bar -> should get a position
    assert (weights["S19"] != 0).any()
    # a mid-score name with too-small |D| should never get a position despite decent rank,
    # i.e. it should never even acquire a nonzero-weight column
    assert "S10" not in weights.columns or (weights["S10"] == 0).all()


def test_exit_on_time_stop_after_max_holding_days():
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    symbols = ["A"] + [f"P{i}" for i in range(15)]
    panel = _make_universe_panel(symbols, dates)

    signal = {}
    S_bar_A = pd.Series(99.0, index=dates)  # always top percentile
    D_A = pd.Series(0.5, index=dates)
    signal["A"] = pd.DataFrame({"S_bar": S_bar_A, "D": D_A})
    for i, s in enumerate([x for x in symbols if x != "A"]):
        signal[s] = pd.DataFrame({"S_bar": pd.Series(float(i), index=dates), "D": pd.Series(0.02, index=dates)})

    eod = {s: _make_eod(dates) for s in symbols}
    result = pf.run_backtest(signal, panel, eod)
    a_weight = result["weights"]["A"]

    # A's signal never changes (permanently top-percentile, permanently
    # clears |D|), so the ONLY thing that can end a holding spell is the
    # 15-day time-stop -- and since A still qualifies afterward, it should
    # re-enter on a later day rather than staying flat. What must hold is
    # that no single continuous holding spell exceeds max_holding_days.
    is_held = (a_weight != 0).astype(int)
    spell_id = (is_held.diff().fillna(is_held.iloc[0]) == 1).cumsum() * is_held
    longest_spell = is_held.groupby(spell_id).sum().drop(0, errors="ignore").max()
    assert longest_spell <= SPEC.entry_exit.max_holding_days
    assert is_held.sum() > SPEC.entry_exit.max_holding_days  # confirms it DID re-enter later, not just stop trading


def test_band_buffer_suppresses_small_rebalances():
    dates = pd.date_range("2024-01-02", periods=15, freq="B")
    symbols = ["A", "B"]
    panel = _make_universe_panel(symbols, dates)
    signal = {
        "A": pd.DataFrame({"S_bar": pd.Series(90.0, index=dates), "D": pd.Series(0.5, index=dates)}),
        "B": pd.DataFrame({"S_bar": pd.Series(10.0, index=dates), "D": pd.Series(-0.5, index=dates)}),
    }
    eod = {s: _make_eod(dates) for s in symbols}
    result = pf.run_backtest(signal, panel, eod)
    daily = result["daily"]
    # once positions stabilize (constant scores/directions every day), later
    # days should show ~zero turnover thanks to the no-trade band
    assert daily["turnover"].iloc[-3:].sum() < 1e-6
