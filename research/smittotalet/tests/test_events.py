import numpy as np
import pandas as pd

from .. import events


def test_event_threshold_is_pit_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, len(idx))}, index=idx)
    thresh = events.event_threshold(returns, q=95, lookback=252)
    # threshold at t must be computable from data strictly before t: perturbing
    # returns.loc[t] must not change threshold.loc[t]
    perturbed = returns.copy()
    t = idx[300]
    perturbed.loc[t, "A"] = 999.0
    thresh_perturbed = events.event_threshold(perturbed, q=95, lookback=252)
    assert np.isclose(thresh.loc[t, "A"], thresh_perturbed.loc[t, "A"], equal_nan=True)


def test_event_matrix_extreme_value_always_flagged():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, len(idx))}, index=idx)
    returns.iloc[350] = 5.0  # huge outlier, must exceed any reasonable percentile
    ev = events.event_matrix(returns, q=95, lookback=252)
    assert ev.iloc[350]["A"] == 1.0


def test_daily_count_sums_across_assets():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    ev = pd.DataFrame({"A": [1, 0, 1], "B": [1, 1, 0]}, index=idx[:3])
    x_t = events.daily_count(ev)
    assert list(x_t) == [2, 1, 1]


def test_build_returns_matrix_and_counts_consistently():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 5)), index=idx,
                            columns=list("ABCDE"))
    ev, x_t = events.build(returns, q=90)
    assert (x_t.dropna() == ev.sum(axis=1, skipna=True).loc[x_t.dropna().index]).all()
