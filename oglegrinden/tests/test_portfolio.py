import numpy as np
import pandas as pd
import pytest

from oglegrinden.portfolio import winsorize_mad, cap_and_renormalize, formation_weights


def test_winsorize_mad_clips_outliers():
    s = pd.Series([1.0, 2.0, 3.0, 2.5, 100.0, 1.5, 2.2])
    w = winsorize_mad(s, k=3.0)
    med = s.median()
    mad = (s - med).abs().median()
    assert w.max() <= med + 3 * mad + 1e-9
    assert w.min() >= med - 3 * mad - 1e-9
    # non-outlier values are untouched
    assert w.iloc[0] == 1.0


def test_winsorize_mad_zero_mad_is_noop():
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    w = winsorize_mad(s, k=3.0)
    assert (w == s).all()


def test_cap_and_renormalize_respects_cap_and_preserves_gross():
    # 10 names, cap 15% -> max capacity 150%, comfortably above this
    # gross target, so redistribution should reach the original gross
    # exactly while respecting the cap.
    w = pd.Series([0.4, -0.2, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    gross_target = w.abs().sum()
    capped = cap_and_renormalize(w, cap=0.15)
    assert (capped.abs() <= 0.15 + 1e-9).all()
    assert np.isclose(capped.abs().sum(), gross_target, atol=1e-6)


def test_cap_and_renormalize_noop_when_already_within_cap():
    w = pd.Series([0.1, -0.1, 0.05, -0.05])
    capped = cap_and_renormalize(w, cap=0.15)
    assert np.allclose(capped.values, w.values)


def test_cap_and_renormalize_infeasible_gross_does_not_violate_cap():
    # 4 names, cap 15% -> max reachable gross = 60% < 100% target.
    w = pd.Series([0.25, -0.25, 0.25, -0.25])
    capped = cap_and_renormalize(w, cap=0.15)
    assert (capped.abs() <= 0.15 + 1e-9).all()
    assert capped.abs().sum() <= 0.60 + 1e-6


def test_formation_weights_are_dollar_neutral_and_gross_100pct_when_uncapped():
    rng = np.random.default_rng(0)
    f = pd.Series(rng.normal(0, 0.02, 20), index=[f"T{i}" for i in range(20)])
    w = formation_weights(f, cap=0.15)
    assert np.isclose(w.sum(), 0.0, atol=1e-6)
    assert np.isclose(w.abs().sum(), 1.0, atol=1e-6)
    assert (w.abs() <= 0.15 + 1e-9).all()


def test_formation_weights_are_reversal_sign_correct():
    # T0 had the worst return (biggest loser) -> should get the largest
    # positive (buy) weight; T_last had the best return (biggest winner)
    # -> should get the most negative (sell) weight.
    f = pd.Series(np.linspace(-0.05, 0.05, 10), index=[f"T{i}" for i in range(10)])
    w = formation_weights(f, cap=0.5)
    assert w.iloc[0] == w.max()
    assert w.iloc[-1] == w.min()
    assert w.iloc[0] > 0
    assert w.iloc[-1] < 0


def test_formation_weights_handles_nan_and_empty():
    f = pd.Series([np.nan, np.nan])
    w = formation_weights(f)
    assert w.empty

    f2 = pd.Series([0.01, np.nan, -0.02, 0.005])
    w2 = formation_weights(f2)
    assert w2.isna().sum() == 0
    assert set(w2.index) == {0, 2, 3}
