"""Sanity checks that the three irreversibility estimators actually measure
time-reversal asymmetry, and are not degenerate proxies for volatility or
skewness (the pre-registered "expected weakness" of this lab).

Run with: python3 -m pytest irreversibility_lab/tests/test_estimators.py -v
"""

import numpy as np
import pytest

from irreversibility_lab.estimators import (
    hvg_irreversibility,
    ordinal_irreversibility,
    psi_irreversibility,
)

N = 2000
SEED = 7


def _reversible_gaussian_iid(n=N, sigma=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma, n)


def _reversible_gaussian_ar1(n=N, phi=0.3, sigma=1.0, seed=SEED):
    """Linear Gaussian AR(1) -- provably time-reversible in the strict
    statistical sense (its time-reversed process is also AR(1) with the
    same parameters)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _irreversible_tar_process(n=N, phi1=0.6, phi2=-0.2, sigma=1.0, seed=SEED):
    """Threshold-autoregressive (SETAR) process with asymmetric AR
    coefficients above/below zero -- the canonical nonlinear time-irreversible
    test process from the time-reversibility testing literature (Ramsey &
    Rothman 1996; Diks, van Houwelingen, Takens & DeGoede 1995). Provably
    irreversible whenever phi1 != phi2, with no reliance on jump kurtosis or
    a leverage-style vol/return coupling.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        phi = phi1 if x[t - 1] > 0 else phi2
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _irreversible_leverage_process(n=N, seed=SEED):
    """GARCH-like process with a leverage effect (negative returns raise
    future volatility more than positive ones) -- the classic confound this
    lab worries about (Zumbach effect). Used to confirm detection, not to
    prove the signal is more than this."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    vol = np.full(n, 1.0)
    for t in range(1, n):
        shock = -0.15 * min(r[t - 1], 0) + 0.03 * max(r[t - 1], 0)
        vol[t] = max(0.2, 0.9 * vol[t - 1] + shock + 0.1)
        r[t] = vol[t] * rng.normal()
    return r


def _shuffled_null(x, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    x = np.asarray(x).copy()
    rng.shuffle(x)
    return x


ESTIMATORS = {
    "hvg": hvg_irreversibility,
    "ordinal": ordinal_irreversibility,
    "psi": lambda w: abs(psi_irreversibility(w)),
}


@pytest.mark.parametrize("name,func", ESTIMATORS.items())
def test_iid_noise_is_near_reversible(name, func):
    x = _reversible_gaussian_iid()
    val = func(x)
    null_vals = [func(_shuffled_null(x, seed=s)) for s in range(5)]
    # iid noise should sit within the shuffle-null spread, not far above it.
    assert val <= np.mean(null_vals) + 3 * (np.std(null_vals) + 1e-9), (
        f"{name}: iid Gaussian noise looks spuriously irreversible "
        f"(val={val}, null_mean={np.mean(null_vals)}, null_std={np.std(null_vals)})"
    )


@pytest.mark.parametrize("name,func", ESTIMATORS.items())
def test_gaussian_ar1_is_near_reversible(name, func):
    x = _reversible_gaussian_ar1()
    val = func(x)
    null_vals = [func(_shuffled_null(x, seed=s)) for s in range(5)]
    assert val <= np.mean(null_vals) + 3 * (np.std(null_vals) + 1e-9), (
        f"{name}: Gaussian AR(1) (provably time-reversible) looks spuriously "
        f"irreversible (val={val}, null_mean={np.mean(null_vals)})"
    )


@pytest.mark.parametrize("name,func", ESTIMATORS.items())
def test_tar_process_is_detected_as_irreversible(name, func):
    irr_vals = [func(_irreversible_tar_process(seed=s)) for s in range(5)]
    rev_vals = [func(_reversible_gaussian_iid(seed=s)) for s in range(5)]
    irr, rev = np.mean(irr_vals), np.mean(rev_vals)
    assert irr > rev, (
        f"{name}: failed to flag the TAR (threshold-AR) process "
        f"as more irreversible than iid noise (irr={irr}, rev={rev})"
    )


@pytest.mark.parametrize("name,func", ESTIMATORS.items())
def test_leverage_process_is_detected_as_irreversible(name, func):
    """This is expected to pass -- it demonstrates the known confound
    (leverage effect creates irreversibility "for free"), which is exactly
    why validation.py must orthogonalize against vol/skew before trusting
    the signal.
    """
    irr = func(_irreversible_leverage_process())
    rev = func(_reversible_gaussian_iid())
    assert irr > rev, (
        f"{name}: failed to flag the leverage-effect process as irreversible "
        f"(irr={irr}, rev={rev})"
    )


@pytest.mark.parametrize("name,func", ESTIMATORS.items())
def test_time_reversal_symmetry_of_the_estimator(name, func):
    """Reversing the entire input series should not change the *magnitude*
    of a KL-style irreversibility estimate (irreversibility of a series and
    its mirror image should match); for psi, sign should flip, magnitude
    should match.
    """
    x = _irreversible_tar_process()
    val_fwd = func(x)
    val_bwd = func(x[::-1].copy())
    assert val_fwd == pytest.approx(val_bwd, rel=0.15, abs=0.05), (
        f"{name}: forward vs reversed-series irreversibility magnitude "
        f"mismatch (fwd={val_fwd}, bwd={val_bwd})"
    )


def test_hvg_scale_and_shift_invariance():
    """HVG irreversibility depends only on the ordinal structure of the
    series, so it must be invariant to affine transforms (scale/shift) --
    otherwise it would just be re-deriving a volatility level."""
    x = _irreversible_tar_process()
    base = hvg_irreversibility(x)
    scaled = hvg_irreversibility(x * 5.0 + 3.0)
    assert base == pytest.approx(scaled, rel=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
