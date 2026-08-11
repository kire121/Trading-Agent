import numpy as np
import pandas as pd

from .. import signal


def test_omori_kernel_sums_to_one():
    w = signal.omori_kernel(lags=10, p=0.5758457103777312, c=1.0)
    assert w.shape == (10,)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.diff(w) < 0)  # monotone decaying


def test_branching_intensity_zero_when_no_events():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    x_t = pd.Series(0.0, index=idx)
    w = signal.omori_kernel()
    lam = signal.branching_intensity(x_t, w)
    assert (lam.dropna() == 0.0).all()


def test_r_hat_equals_one_when_x_matches_lambda_exactly():
    """If X_t is generated to exactly match its own expected Lambda_t on
    average, R_hat should hover near 1 (subcritical/critical boundary)."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    x_t = pd.Series(rng.poisson(3, size=len(idx)).astype(float), index=idx)
    w = signal.omori_kernel()
    lam = signal.branching_intensity(x_t, w)
    r_hat = signal.reproduction_number(x_t, lam, tau=21)
    valid = r_hat.dropna()
    assert len(valid) > 0
    # i.i.d. Poisson X has no branching structure -> R_hat should be near 1
    assert 0.5 < valid.median() < 2.0


def test_r_hat_rises_after_a_burst():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    x_t = pd.Series(1.0, index=idx)
    # inject a burst: 10 elevated days, so recent X > implied Lambda from a flat past
    x_t.iloc[100:110] = 10.0
    w = signal.omori_kernel()
    lam = signal.branching_intensity(x_t, w)
    r_hat = signal.reproduction_number(x_t, lam, tau=10)
    pre_burst = r_hat.iloc[90]
    post_burst = r_hat.iloc[109]
    assert post_burst > pre_burst


def test_tilt_clips_to_bounds():
    r_hat = pd.Series([0.01, 0.5, 1.0, 2.0, 100.0])
    g = signal.tilt(r_hat, kappa=1, low=0.3, high=1.3)
    assert g.min() >= 0.3
    assert g.max() <= 1.3


def test_tilt_monotone_decreasing_in_r_hat():
    r_hat = pd.Series(np.linspace(0.2, 5.0, 50))
    g = signal.tilt(r_hat, kappa=1)
    assert (g.diff().dropna() <= 1e-9).all()


def test_build_full_pipeline_runs():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(1)
    x_t = pd.Series(rng.poisson(2, size=len(idx)).astype(float), index=idx)
    r_hat, g_t, lam_t = signal.build(x_t, q=95, tau=21, kappa=1)
    assert r_hat.shape == x_t.shape
    assert g_t.dropna().between(0.3, 1.3).all()
