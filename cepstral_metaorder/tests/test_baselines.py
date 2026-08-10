import numpy as np
import pandas as pd

from cepstral_metaorder import baselines


def test_twin_is_alive_flags_degenerate_all_zero_direction():
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    frame = pd.DataFrame({"S_bar": np.random.default_rng(0).normal(size=30), "D": np.zeros(30)}, index=dates)
    checks = baselines.twin_is_alive(frame)
    assert checks["alive"] is False
    assert checks["direction_balance_ok"] is False


def test_twin_is_alive_flags_degenerate_constant_score():
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    rng = np.random.default_rng(1)
    d = rng.choice([-1.0, 1.0], size=30)
    frame = pd.DataFrame({"S_bar": np.full(30, 2.5), "D": d}, index=dates)
    checks = baselines.twin_is_alive(frame)
    assert checks["alive"] is False
    assert checks["score_dispersion_ok"] is False


def test_twin_is_alive_flags_low_coverage():
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    rng = np.random.default_rng(2)
    s = rng.normal(size=30)
    d = rng.choice([-1.0, 1.0], size=30)
    s[:20] = np.nan
    frame = pd.DataFrame({"S_bar": s, "D": d}, index=dates)
    checks = baselines.twin_is_alive(frame)
    assert checks["alive"] is False
    assert checks["coverage_ok"] is False


def test_twin_is_alive_passes_a_healthy_signal():
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "S_bar": rng.normal(size=60),
        "D": rng.choice([-1.0, -0.5, 0.5, 1.0], size=60),
    }, index=dates)
    checks = baselines.twin_is_alive(frame)
    assert checks["alive"] is True


def test_reversal_twin_fades_the_prior_move():
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    df = pd.DataFrame({"date": dates, "ret_5d": np.linspace(-0.1, 0.1, 10)})
    out = baselines.reversal_twin_raw({"A": df})["A"]
    # positive prior 5d return -> reversal twin direction should be negative, and vice versa
    up_days = df["ret_5d"] > 0
    assert (out.loc[dates[up_days.values], "D"] < 0).all()
    down_days = df["ret_5d"] < 0
    assert (out.loc[dates[down_days.values], "D"] > 0).all()
