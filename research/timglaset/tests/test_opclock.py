"""Mandatory opclock test cases 1-6 per docs/timglaset_forregistrering.md
§13. Must be fully green before any real data is read (rule: "alla gröna
före första riktiga datainläsning"). Cases 7-9 (end-to-end synthetic
generator, liveness-assertion regression, DSR-tiling reproduction) live in
tests/test_integration.py since they require twins.py/ladder.py.
"""
import numpy as np
import pandas as pd
import pytest

from research.timglaset import opclock


def _mkdf(arr, start="2000-01-03", cols=None):
    n_t = arr.shape[0]
    n_c = arr.shape[1] if arr.ndim > 1 else 1
    idx = pd.bdate_range(start, periods=n_t)
    cols = cols or [f"c{i}" for i in range(n_c)]
    return pd.DataFrame(arr.reshape(n_t, n_c), index=idx, columns=cols)


# ---------------------------------------------------------------------------
# 1. tau == 1 -> M identical (atol 1e-12) with closed calendar-EWMA sum.
# ---------------------------------------------------------------------------
def test_1_tau_one_matches_closed_form_calendar_ewma():
    rng = np.random.default_rng(0)
    n = 80
    r = rng.normal(0, 0.01, size=(n, 1))
    r[5, 0] = np.nan  # exercise the "NaN return -> freeze" path too
    returns = _mkdf(r)
    tau = opclock.calendar_twin_tau(returns)
    halflife = 20.0
    kappa = np.log(2.0) / halflife

    M, V, T = opclock.compute_op_ewma(returns, tau, halflife)

    # Closed-form reference: sequential recursion in plain Python, treating
    # tau==1 always, NaN-return days frozen (no decay, not accumulated).
    m_ref, v_ref, t_ref = 0.0, 0.0, 0.0
    m_series, v_series, t_series = [], [], []
    for rt in r[:, 0]:
        if np.isnan(rt):
            pass  # freeze
        else:
            m_ref = rt + np.exp(-kappa) * m_ref
            v_ref = rt * rt + np.exp(-2 * kappa) * v_ref
            t_ref = t_ref + 1.0
        m_series.append(m_ref)
        v_series.append(v_ref)
        t_series.append(t_ref)

    np.testing.assert_allclose(M.iloc[:, 0].to_numpy(), np.array(m_series), atol=1e-12)
    np.testing.assert_allclose(V.iloc[:, 0].to_numpy(), np.array(v_series), atol=1e-12)
    np.testing.assert_allclose(T.iloc[:, 0].to_numpy(), np.array(t_series), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Scale invariance: volume x10 globally -> tau unchanged after burn-in.
# ---------------------------------------------------------------------------
def test_2_tau_scale_invariance():
    rng = np.random.default_rng(1)
    n = 400
    vol = rng.lognormal(mean=10, sigma=0.3, size=(n, 3))
    volume = _mkdf(vol, cols=["a", "b", "c"])
    tau1 = opclock.compute_tau(volume, window=252, cap=5.0)
    tau10 = opclock.compute_tau(volume * 10.0, window=252, cap=5.0)
    # Post burn-in (index >= window), values must match exactly regardless
    # of scale; pre-burn-in both sides are the same 1.0 fallback anyway.
    pd.testing.assert_frame_equal(tau1, tau10, check_exact=False, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Impulse response: single r != 0 at s -> M_t = r_s * exp(-kappa*(T_t-T_s)) exactly.
# ---------------------------------------------------------------------------
def test_3_impulse_response_exact():
    rng = np.random.default_rng(2)
    n = 120
    s = 30
    r = np.zeros((n, 1))
    r[s, 0] = 0.0345
    returns = _mkdf(r)
    # Arbitrary (non-degenerate) tau path, independent of returns.
    tau_vals = rng.uniform(0.3, 2.5, size=(n, 1))
    tau = _mkdf(tau_vals)
    halflife = 15.0
    kappa = np.log(2.0) / halflife

    M, V, T = opclock.compute_op_ewma(returns, tau, halflife)
    T_arr = T.iloc[:, 0].to_numpy()
    M_arr = M.iloc[:, 0].to_numpy()

    assert np.allclose(M_arr[:s], 0.0, atol=1e-12)  # nothing before the impulse
    expected = r[s, 0] * np.exp(-kappa * (T_arr[s:] - T_arr[s]))
    np.testing.assert_allclose(M_arr[s:], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. PIT-assert: truncate input at arbitrary t, recompute -> prefix identical.
# ---------------------------------------------------------------------------
def test_4_pit_no_lookahead():
    rng = np.random.default_rng(3)
    n = 500
    vol = rng.lognormal(10, 0.3, size=(n, 2))
    ret = rng.normal(0, 0.01, size=(n, 2))
    volume = _mkdf(vol, cols=["x", "y"])
    returns = _mkdf(ret, cols=["x", "y"])

    tau_full = opclock.compute_tau(volume, window=252, cap=5.0)
    M_full, V_full, T_full = opclock.compute_op_ewma(returns, tau_full, 63.0)
    z_full = opclock.compute_signal(M_full, V_full, T_full, 63.0, "tanh")

    cutoff = 350
    volume_trunc = volume.iloc[:cutoff]
    returns_trunc = returns.iloc[:cutoff]
    tau_trunc = opclock.compute_tau(volume_trunc, window=252, cap=5.0)
    M_t, V_t, T_t = opclock.compute_op_ewma(returns_trunc, tau_trunc, 63.0)
    z_trunc = opclock.compute_signal(M_t, V_t, T_t, 63.0, "tanh")

    pd.testing.assert_frame_equal(tau_full.iloc[:cutoff], tau_trunc, atol=1e-12)
    pd.testing.assert_frame_equal(M_full.iloc[:cutoff], M_t, atol=1e-12)
    pd.testing.assert_frame_equal(V_full.iloc[:cutoff], V_t, atol=1e-12)
    pd.testing.assert_frame_equal(T_full.iloc[:cutoff], T_t, atol=1e-12)
    pd.testing.assert_frame_equal(z_full.iloc[:cutoff], z_trunc, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. NaN-/zero-volume day -> tau=1 fallback path (spec §13 for NaN, §14
#    verbatim "nollvolymdagar -> tau=1" for zero), no NaN propagation in M/V.
# ---------------------------------------------------------------------------
def test_5_nan_and_zero_volume_no_propagation():
    rng = np.random.default_rng(4)
    n = 400
    vol = rng.lognormal(10, 0.3, size=(n, 1))
    vol[300, 0] = np.nan
    vol[310, 0] = 0.0
    volume = _mkdf(vol)
    ret = rng.normal(0, 0.01, size=(n, 1))
    returns = _mkdf(ret)

    tau = opclock.compute_tau(volume, window=252, cap=5.0)
    assert tau.iloc[300, 0] == pytest.approx(1.0)  # NaN volume -> fallback 1.0
    assert tau.iloc[310, 0] == pytest.approx(1.0)  # zero volume -> spec §14 fallback 1.0
    assert not tau.isna().any().any()

    M, V, T = opclock.compute_op_ewma(returns, tau, 63.0)
    assert not M.isna().any().any()
    assert not V.isna().any().any()
    assert not T.isna().any().any()
    assert np.isfinite(M.to_numpy()).all()
    assert np.isfinite(V.to_numpy()).all()


def test_5b_zero_volume_is_isolated_not_propagating():
    """A zero-volume day is a real (non-null) numerator value, so unlike a
    NaN it must NOT poison the rolling-median window for ~`window` following
    days -- only the single zero-volume day itself falls back to tau=1."""
    rng = np.random.default_rng(5)
    n = 500
    vol = rng.lognormal(10, 0.3, size=(n, 1))
    vol[400, 0] = 0.0
    volume = _mkdf(vol)
    tau = opclock.compute_tau(volume, window=252, cap=5.0)
    assert tau.iloc[400, 0] == pytest.approx(1.0)
    # Neighboring, unaffected days (post burn-in, away from index 400) must
    # not be forced to the fallback value -- confirms no propagation. With
    # continuous lognormal volume, an exact ratio of 1.0 has probability 0.
    for i in (395, 396, 397, 398, 399, 401, 402, 403, 404, 405):
        assert tau.iloc[i, 0] != pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. z-variance ~= 1 (+-10%) under iid simulation regardless of synthetic
#    clock speed (clock-invariant normalization).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tau_mean,tau_sigma", [(1.0, 0.05), (0.4, 0.15), (2.5, 0.8)])
def test_6_z_variance_clock_invariant(tau_mean, tau_sigma):
    rng = np.random.default_rng(hash((tau_mean, tau_sigma)) % (2 ** 32))
    n, n_c = 6000, 25
    ret = rng.normal(0, 0.01, size=(n, n_c))
    returns = _mkdf(ret, cols=[f"t{i}" for i in range(n_c)])
    tau_vals = np.clip(rng.normal(tau_mean, tau_sigma, size=(n, n_c)), 0.01, None)
    tau = _mkdf(tau_vals, cols=[f"t{i}" for i in range(n_c)])

    halflife = 63.0
    M, V, T = opclock.compute_op_ewma(returns, tau, halflife)
    z = opclock.compute_raw_z(M, V, T, halflife)
    z_valid = z.to_numpy()
    z_valid = z_valid[~np.isnan(z_valid)]
    assert len(z_valid) > 1000
    std_z = np.std(z_valid)
    assert 0.9 <= std_z <= 1.1, f"std(z)={std_z} not within +-10% of 1 for tau~({tau_mean},{tau_sigma})"
