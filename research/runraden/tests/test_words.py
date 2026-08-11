import numpy as np
import pandas as pd

from words import sign_char, weekly_table, build_asset_week_panel, word_sign_vector


def test_sign_char_zero_maps_down():
    assert sign_char(0.0) == "-"
    assert sign_char(1e-9) == "+"
    assert sign_char(-1e-9) == "-"
    assert sign_char(0.01) == "+"
    assert sign_char(-0.01) == "-"


def _make_daily_series(dates, prices):
    return pd.Series(prices, index=pd.to_datetime(dates))


def test_weekly_table_five_day_week():
    # Mon 2024-01-08 .. Fri 2024-01-12, plus a seed day before.
    dates = ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
    prices = [100, 101, 100, 102, 101, 103]  # returns for the week: +,-,+,-,+
    s = _make_daily_series(dates, prices)
    wt = weekly_table(s)
    week_rows = wt[wt["n_days"] == 5]
    assert len(week_rows) == 1
    row = week_rows.iloc[0]
    assert row["table_id"] == "5d"
    assert row["word"] == ("+", "-", "+", "-", "+")


def test_weekly_table_four_day_week_gets_16_cell_table():
    # Tue-Fri only (Monday holiday)
    dates = ["2024-01-05", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
    prices = [100, 101, 102, 103, 104]
    s = _make_daily_series(dates, prices)
    wt = weekly_table(s)
    row = wt[wt["n_days"] == 4].iloc[0]
    assert row["table_id"] == "4d"
    assert row["word"] == ("+", "+", "+", "+")


def test_weekly_table_short_week_is_flat():
    dates = ["2024-01-05", "2024-01-08", "2024-01-09"]  # only 2 trading days that week
    prices = [100, 101, 99]
    s = _make_daily_series(dates, prices)
    wt = weekly_table(s)
    row = wt[wt["n_days"] == 2].iloc[0]
    assert row["table_id"] is None
    assert row["word"] is None


def test_weekly_table_week_return_is_order_invariant():
    dates = ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
    prices_a = [100, 101, 100, 102, 101, 103]
    s_a = _make_daily_series(dates, prices_a)
    wt_a = weekly_table(s_a)
    week_return_a = wt_a[wt_a["n_days"] == 5].iloc[0]["week_return"]

    # Same daily returns, different order -> same compounded week return, different word.
    rets = np.array(prices_a[1:]) / np.array(prices_a[:-1]) - 1.0
    shuffled = rets[::-1]
    prices_b = [100.0]
    for r in shuffled:
        prices_b.append(prices_b[-1] * (1 + r))
    s_b = _make_daily_series(dates, prices_b)
    wt_b = weekly_table(s_b)
    week_return_b = wt_b[wt_b["n_days"] == 5].iloc[0]["week_return"]

    assert np.isclose(week_return_a, week_return_b)


def test_word_sign_vector():
    v = word_sign_vector(("+", "-", "+", "+", "-"))
    assert list(v) == [1.0, -1.0, 1.0, 1.0, -1.0]


def test_build_asset_week_panel_drops_gapped_weeks():
    # Two assets, one with a normal weekly cadence, one with a big data gap
    # that should not be silently compounded into a single "next week" return.
    dates_ok = pd.date_range("2020-01-06", periods=20, freq="B")
    prices_ok = pd.Series(
        np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, len(dates_ok))) * 100,
        index=dates_ok,
    )

    dates_gap = list(pd.date_range("2020-01-06", periods=10, freq="B")) + \
        list(pd.date_range("2020-03-02", periods=10, freq="B"))  # ~7 week gap
    prices_gap = pd.Series(100 + np.arange(len(dates_gap)), index=pd.DatetimeIndex(dates_gap))

    panel = build_asset_week_panel({"OK": prices_ok, "GAP": prices_gap})
    gap_rows = panel[panel["asset"] == "GAP"]
    # The row whose t_signal is just before the gap must be dropped (next_first_date
    # would be ~7 weeks later otherwise).
    max_gap_days = (gap_rows["next_first_date"] - gap_rows["t_signal"]).dt.days.max()
    assert max_gap_days <= 4 or gap_rows.empty
