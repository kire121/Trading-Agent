"""Tests for the core topological signal.

These are the tests that matter most for the hypothesis: does total H1
persistence actually distinguish a single-factor correlation cloud (should
be ~flat/simplex-like, no loops) from a frustrated-cycle correlation cloud
(A~B~C~D~A but A not~C, which cannot be embedded without a hole)?
"""

import numpy as np
import pandas as pd
import pytest

from oglegrinden.topology import compute_topology, correlation_distance, absorption_ratio


def _single_factor_returns(n_assets=20, t=250, seed=42, beta_low=0.3, beta_high=0.9):
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(t)
    betas = rng.uniform(beta_low, beta_high, n_assets)
    data = {
        f"A{i}": betas[i] * factor + np.sqrt(1 - betas[i] ** 2) * rng.standard_normal(t)
        for i in range(n_assets)
    }
    return pd.DataFrame(data)


def _frustrated_cycle_returns(assets_per_hub=5, t=250, seed=42, beta=0.8):
    rng = np.random.default_rng(seed)
    z_ab, z_bc, z_cd, z_da = (rng.standard_normal(t) for _ in range(4))
    hubs = {
        "A": (z_da + z_ab) / np.sqrt(2),
        "B": (z_ab + z_bc) / np.sqrt(2),
        "C": (z_bc + z_cd) / np.sqrt(2),
        "D": (z_cd + z_da) / np.sqrt(2),
    }
    cols = {}
    for name, hub in hubs.items():
        for i in range(assets_per_hub):
            cols[f"{name}{i}"] = beta * hub + np.sqrt(1 - beta ** 2) * rng.standard_normal(t)
    return pd.DataFrame(cols)


class TestCorrelationDistance:
    def test_metric_properties(self):
        returns = _single_factor_returns(n_assets=8, t=100)
        dist_df, corr = correlation_distance(returns)
        dist = dist_df.values
        assert np.allclose(np.diag(dist), 0.0, atol=1e-8)
        assert np.allclose(dist, dist.T)
        assert np.all(dist >= 0)
        # d_ij = sqrt(2*(1-rho_ij)) exactly
        expected = np.sqrt(np.clip(2 * (1 - corr.values), 0, None))
        assert np.allclose(dist, expected)

    def test_rejects_missing_data(self):
        returns = _single_factor_returns(n_assets=5, t=50)
        returns.iloc[3, 2] = np.nan
        with pytest.raises(ValueError):
            correlation_distance(returns)


class TestTotalH1Persistence:
    def test_single_factor_world_has_no_persistent_loops(self):
        """A single common factor puts points on a near-1D arc ordered by
        factor loading; the VR complex should fill in without ever
        enclosing a hole, so total H1 persistence should be at (or very
        near) zero."""
        returns = _single_factor_returns(n_assets=20, t=250, seed=42)
        snap = compute_topology(returns)
        assert snap.total_h1_persistence < 0.05
        assert snap.n_h1_features <= 1

    def test_frustrated_cycle_produces_persistent_loops(self):
        """A~B, B~C, C~D, D~A but A not~C and B not~D cannot be embedded
        without a hole -> the VR complex must carry persistent H1 classes."""
        returns = _frustrated_cycle_returns(assets_per_hub=5, t=250, seed=42)
        snap = compute_topology(returns)
        assert snap.total_h1_persistence > 0.15
        assert snap.n_h1_features >= 1

    def test_frustrated_exceeds_single_factor_across_seeds(self):
        """The qualitative separation should hold across random seeds, not
        just be an artifact of one draw."""
        wins = 0
        trials = 8
        for seed in range(trials):
            single = compute_topology(_single_factor_returns(n_assets=16, t=200, seed=seed))
            frustrated = compute_topology(
                _frustrated_cycle_returns(assets_per_hub=4, t=200, seed=seed)
            )
            if frustrated.total_h1_persistence > single.total_h1_persistence:
                wins += 1
        assert wins >= trials - 1  # allow at most one noisy draw to flip

    def test_more_points_scale_reasonably_fast(self):
        """Spec claims ripser on ~24 points is millisecond-scale; sanity
        check it doesn't blow up for a universe-sized cloud."""
        import time

        returns = _single_factor_returns(n_assets=24, t=120, seed=1)
        start = time.perf_counter()
        compute_topology(returns)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0  # generous CI-safe bound, spec expects ms-scale


class TestAbsorptionRatio:
    def test_strong_single_factor_has_high_absorption(self):
        returns = _single_factor_returns(n_assets=20, t=250, seed=7, beta_low=0.85, beta_high=0.95)
        _, corr = correlation_distance(returns)
        ar = absorption_ratio(corr, top_fraction=0.2)
        assert ar > 0.5

    def test_near_independent_assets_have_low_absorption(self):
        rng = np.random.default_rng(7)
        returns = pd.DataFrame(rng.standard_normal((250, 20)))
        _, corr = correlation_distance(returns)
        ar = absorption_ratio(corr, top_fraction=0.2)
        # top 20% (4 of 20) eigenvalues of a near-identity correlation
        # matrix should explain only modestly more than their 1/5 share
        assert ar < 0.4
