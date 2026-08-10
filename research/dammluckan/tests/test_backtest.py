import numpy as np
import pandas as pd
import pytest

from research.dammluckan import backtest, config, data
from research.dammluckan.portfolio import CandidateTrade, Trade


def test_compute_exit_time_stop_when_no_opposite_event():
    T = 30
    opp = np.zeros(T)
    exit_pos, reason = backtest.compute_exit(entry_pos=5, h=10, opp_events=opp, T=T)
    assert exit_pos == 15
    assert reason == "time_stop"


def test_compute_exit_opposite_record_wins_when_earlier():
    T = 30
    opp = np.zeros(T)
    opp[8] = 1.0  # decision date 8 -> execution 9, strictly before time-stop exec 15
    exit_pos, reason = backtest.compute_exit(entry_pos=5, h=10, opp_events=opp, T=T)
    assert exit_pos == 9
    assert reason == "opposite_record"


def test_compute_exit_time_stop_wins_on_tie():
    T = 30
    opp = np.zeros(T)
    opp[13] = 1.0  # decision 13 -> execution 14 == entry(5)+h-1(9)? check boundary precisely
    # entry=5, h=10 -> exec_time_stop = 15. scan window is [5, 5+10-1)=[5,14).
    # opp[13]=1 -> candidate_exit = 13+1 = 14 < 15 -> should be admitted as opposite_record.
    exit_pos, reason = backtest.compute_exit(entry_pos=5, h=10, opp_events=opp, T=T)
    assert exit_pos == 14
    assert reason == "opposite_record"
    # Now place the opposite event exactly at the scan boundary (excluded):
    opp2 = np.zeros(T)
    opp2[14] = 1.0  # outside [5,14) scan window -> ignored, time-stop wins
    exit_pos2, reason2 = backtest.compute_exit(entry_pos=5, h=10, opp_events=opp2, T=T)
    assert exit_pos2 == 15
    assert reason2 == "time_stop"


def test_generate_candidates_donchian_twin_theta_minus_inf_not_treated_as_unavailable():
    """Regression test: theta=-inf (the Donchian twin's "always admit" gate)
    must NOT be treated the same as theta=NaN ("no threshold available").
    np.isfinite(-inf) is False, so a naive `if not isfinite(theta): skip`
    guard silently drops every ticker for the Donchian twin -- this must not
    happen."""
    from research.dammluckan import signal as signal_mod

    rng = np.random.default_rng(3)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    close = pd.DataFrame({"TST": prices}, index=dates)
    vol = pd.DataFrame({"TST": [1_000_000.0] * n}, index=dates)
    panel = data.Panel(tickers=["TST"], raw_close=close, raw_open=close, high=close, low=close,
                        adj_close=close, adj_open=close, volume=vol)
    sig = signal_mod.build_signal(panel, n=40, c=1.0)

    gated_theta = pd.Series({"TST": 0.9})  # a real, restrictive threshold
    gated = backtest.generate_candidates(panel, sig, gated_theta, gated_theta, h=10)

    neg_inf_theta = pd.Series({"TST": -np.inf})  # Donchian twin: always admit
    ungated = backtest.generate_candidates(panel, sig, neg_inf_theta, neg_inf_theta, h=10)

    assert len(ungated) > 0
    assert len(ungated) >= len(gated)


def test_compute_exit_clips_at_end_of_data():
    T = 12
    opp = np.zeros(T)
    exit_pos, reason = backtest.compute_exit(entry_pos=5, h=10, opp_events=opp, T=T)
    assert exit_pos == T - 1
    assert reason == "time_stop"


def _candidate(ticker, direction, entry_pos, exit_pos, o_value=0.5):
    return CandidateTrade(
        ticker=ticker, direction=direction, decision_date=pd.Timestamp("2020-01-01"),
        entry_date=pd.Timestamp("2020-01-02"), entry_pos=entry_pos,
        exit_date=pd.Timestamp("2020-01-03"), exit_pos=exit_pos, exit_reason="time_stop",
        entry_price=100.0, exit_price=101.0, o_value=o_value, entry_adv=1e9, exit_adv=1e9,
    )


def test_admit_respects_max_concurrent():
    # 15 candidates all overlapping [entry=0, exit=20), different tickers ->
    # only MAX_CONCURRENT should be admitted.
    candidates = [_candidate(f"T{i}", 1, 0, 20) for i in range(15)]
    admitted = backtest.admit_candidates(candidates, max_concurrent=12)
    assert len(admitted) == 12


def test_admit_prioritizes_higher_occupation_on_contention():
    candidates = [_candidate(f"T{i}", 1, 0, 20, o_value=float(i)) for i in range(15)]
    admitted = backtest.admit_candidates(candidates, max_concurrent=12)
    admitted_tickers = {c.ticker for c in admitted}
    # highest o_value candidates are T14..T3 (12 of them); T0,T1,T2 should be dropped
    assert admitted_tickers == {f"T{i}" for i in range(3, 15)}


def test_admit_ignores_new_event_while_asset_has_open_position():
    c1 = _candidate("SPY", 1, entry_pos=0, exit_pos=20)
    c2 = _candidate("SPY", 1, entry_pos=5, exit_pos=25)  # fires while c1 still open
    admitted = backtest.admit_candidates([c1, c2], max_concurrent=12)
    assert len(admitted) == 1
    assert admitted[0] is c1


def test_admit_allows_reentry_after_exit():
    c1 = _candidate("SPY", 1, entry_pos=0, exit_pos=10)
    c2 = _candidate("SPY", 1, entry_pos=10, exit_pos=20)  # entry == prior exit: slot freed in time
    admitted = backtest.admit_candidates([c1, c2], max_concurrent=12)
    assert len(admitted) == 2


def _toy_panel_with_open(prices_close, prices_open, dates=None, ticker="TST"):
    n = len(prices_close)
    if dates is None:
        dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({ticker: prices_close}, index=dates)
    open_ = pd.DataFrame({ticker: prices_open}, index=dates)
    vol = pd.DataFrame({ticker: [1_000_000.0] * n}, index=dates)
    return data.Panel(tickers=[ticker], raw_close=close, raw_open=open_, high=close, low=close,
                       adj_close=close, adj_open=open_, volume=vol)


def test_daily_returns_hand_computed_three_day_hold():
    # 4 trading days: entry executes day 1 (open), held through day 2
    # (close-to-close), exits day 3 (open). Hand-computed leg-by-leg.
    closes = [100.0, 102.0, 105.0, 103.0]
    opens = [100.0, 101.0, 103.0, 104.0]
    panel = _toy_panel_with_open(closes, opens)
    trade = Trade(
        ticker="TST", direction=1, decision_date=panel.dates[0], entry_date=panel.dates[1],
        entry_pos=1, exit_date=panel.dates[3], exit_pos=3, exit_reason="time_stop",
        entry_price=opens[1], exit_price=opens[3], o_value=0.5, entry_adv=1e9, exit_adv=1e9,
        weight=0.20, sigma_hat=0.15, gross_return=(opens[3] / opens[1] - 1.0), cost_frac=0.001,
    )
    rets = backtest.daily_returns(panel, [trade])
    w = 0.20
    # day1 (entry, pos=1): open->close = 101 -> 102
    expected_day1 = w * (closes[1] / opens[1] - 1.0) - w * 0.001
    # day2 (held, pos=2): close[1]->close[2] = 102 -> 105
    expected_day2 = w * (closes[2] / closes[1] - 1.0)
    # day3 (exit, pos=3): close[2]->open[3] = 105 -> 104
    expected_day3 = w * (opens[3] / closes[2] - 1.0)
    assert abs(rets.iloc[0]) < 1e-12
    assert abs(rets.iloc[1] - expected_day1) < 1e-12
    assert abs(rets.iloc[2] - expected_day2) < 1e-12
    assert abs(rets.iloc[3] - expected_day3) < 1e-12


def test_solve_k_for_target_vol_converges_when_gross_cap_binds():
    """Regression test: a single-shot linear rescale (k_final = k0 *
    target/vol(k0)) overshoots when the 200% gross cap binds, because the
    k->vol relationship stops being linear once positions get haircut.
    _solve_k_for_target_vol must iterate to convergence instead. Construct a
    scenario with moderate cap pressure (staggered entries, realistic vol,
    so several -- not all, not just one -- positions get haircut) and show
    the iterative solve lands materially closer to the target than the
    single-shot linear rescale it replaced."""
    n = 550
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(5)
    closes = pd.DataFrame(
        {f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))) for i in range(20)}, index=dates
    )
    vol = pd.DataFrame(1_000_000.0, index=dates, columns=closes.columns)
    panel = data.Panel(tickers=list(closes.columns), raw_close=closes, raw_open=closes, high=closes,
                        low=closes, adj_close=closes, adj_open=closes, volume=vol)
    # Staggered, long-overlapping windows -> the cap binds for a sustained
    # stretch across many (not all) names, closer to the real Donchian-twin
    # dynamics that exposed the original single-shot bug.
    admitted = [_candidate(f"T{i}", 1, entry_pos=20 + 5 * i, exit_pos=350 + 5 * i) for i in range(20)]

    target_vol = 0.08
    k_final, _ = backtest._solve_k_for_target_vol(
        panel, admitted, is_start=dates[0], is_end=dates[-1], target_vol=target_vol, gross_cap=config.GROSS_CAP,
    )
    achieved_vol_iterative = backtest.annualized_vol(
        backtest.daily_returns(panel, backtest.assign_weights(panel, admitted, k_final, gross_cap=config.GROSS_CAP))
        .loc[dates[0]:dates[-1]]
    )

    # The single-shot approach the fix replaced: one linear rescale from k0.
    trades0 = backtest.assign_weights(panel, admitted, target_vol, gross_cap=config.GROSS_CAP)
    vol_at_k0 = backtest.annualized_vol(backtest.daily_returns(panel, trades0).loc[dates[0]:dates[-1]])
    k_single_shot = target_vol * (target_vol / vol_at_k0)
    achieved_vol_single_shot = backtest.annualized_vol(
        backtest.daily_returns(panel, backtest.assign_weights(panel, admitted, k_single_shot, gross_cap=config.GROSS_CAP))
        .loc[dates[0]:dates[-1]]
    )

    err_iterative = abs(achieved_vol_iterative - target_vol) / target_vol
    err_single_shot = abs(achieved_vol_single_shot - target_vol) / target_vol
    assert err_iterative < 0.05          # iterative solve lands within 5% of target
    assert err_iterative < err_single_shot  # and strictly beats the single-shot approach it replaced


def test_assign_weights_gross_cap_haircut():
    # Many simultaneous candidates with huge raw weight (tiny sigma_hat) must
    # be haircut so total gross never exceeds config.GROSS_CAP.
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = pd.DataFrame({f"T{i}": np.full(n, 100.0) + np.cumsum(np.random.default_rng(i).normal(0, 0.01, n))
                            for i in range(20)}, index=dates)
    opens = closes.copy()
    vol = pd.DataFrame(1_000_000.0, index=dates, columns=closes.columns)
    panel = data.Panel(tickers=list(closes.columns), raw_close=closes, raw_open=opens, high=closes, low=closes,
                        adj_close=closes, adj_open=opens, volume=vol)
    candidates = [_candidate(f"T{i}", 1, entry_pos=100, exit_pos=110) for i in range(20)]
    trades = backtest.assign_weights(panel, candidates, k=10.0, gross_cap=config.GROSS_CAP)  # huge k
    gross = sum(abs(t.weight) for t in trades)
    assert gross <= config.GROSS_CAP + 1e-9
