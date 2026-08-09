"""Topology of the correlation cloud.

Core idea: embed the universe as a point cloud with the correlation
distance d_ij = sqrt(2 * (1 - rho_ij)) (this is exactly Euclidean distance
between standardized return vectors, up to a scale factor, so it is a
genuine metric -- non-negative, symmetric, zero on the diagonal, and
satisfies the triangle inequality whenever rho is a valid correlation
matrix). Run Vietoris-Rips persistent homology (maxdim=1) on that distance
matrix. In a single-factor world all pairwise correlations are driven by
one common factor, points sit close to a low-dimensional (near-simplex or
ordered-arc) configuration, and no persistent 1-cycles form. Persistent
H1 classes ("loops") require frustrated correlation cycles: A~B~C~D~A but
A not~ C, which is exactly the relative-value structure that cross-
sectional reversal is a bet on.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from ripser import ripser


@dataclass
class TopologySnapshot:
    """Result of one persistent-homology computation on a correlation cloud."""

    n_points: int
    tickers: list
    corr: pd.DataFrame
    dist: np.ndarray
    h0_diagram: np.ndarray  # (birth, death) pairs, dim 0
    h1_diagram: np.ndarray  # (birth, death) pairs, dim 1
    total_h1_persistence: float  # L_t = sum_k (death_k - birth_k) over finite H1 bars
    n_h1_features: int
    rho_bar: float  # mean off-diagonal correlation
    absorption_ratio: float  # top-k eigenvalue share of total variance


def correlation_distance(returns_window: pd.DataFrame, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Pairwise correlation-distance matrix d_ij = sqrt(2*(1-rho_ij)).

    `returns_window` is a (T days x N assets) frame of daily returns with
    no missing values (caller is responsible for restricting to the
    point-in-time-eligible universe and dropping any asset with gaps in the
    window -- silently imputing/forward-filling return data would bias the
    correlation structure).
    """
    if returns_window.isna().any().any():
        raise ValueError(
            "correlation_distance: returns_window contains NaNs; caller must "
            "restrict to assets with complete history over the window"
        )
    corr = returns_window.corr(method="pearson")
    # Numerical noise can push (1 - rho) fractionally negative on the
    # diagonal or for rho fractionally > 1 between near-duplicate columns;
    # clip before the sqrt.
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr.values), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    # Symmetrize away any floating-point asymmetry from corr() itself.
    dist = (dist + dist.T) / 2.0
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns), corr


def absorption_ratio(corr: pd.DataFrame, top_fraction: float = 0.2) -> float:
    """Kritzman et al. (2011) absorption ratio: share of total variance
    explained by the top `top_fraction` of eigenvalues of the correlation
    matrix. High values indicate a compressed, low-dimensional (single-
    factor-like) correlation structure -- the topological null case.
    """
    n = corr.shape[0]
    k = max(1, int(np.ceil(top_fraction * n)))
    eigvals = np.linalg.eigvalsh(corr.values)
    eigvals = np.clip(eigvals, 0.0, None)  # correlation matrices are PSD in theory
    eigvals = np.sort(eigvals)[::-1]
    total = eigvals.sum()
    if total <= 0:
        return float("nan")
    return float(eigvals[:k].sum() / total)


def residualize_against(returns_window: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """Per-column OLS residuals of `returns_window` regressed on
    `benchmark_returns` (with intercept), computed in closed form
    (beta = cov(x, bench) / var(bench)). Used for the declared
    "SPY-residual correlation" variant: build the topology on
    market-neutralized returns instead of raw returns.
    """
    bench = benchmark_returns.reindex(returns_window.index)
    if bench.isna().any():
        raise ValueError("residualize_against: benchmark has gaps over the window")
    bench_c = bench - bench.mean()
    bench_var = float((bench_c ** 2).mean())
    if bench_var == 0:
        raise ValueError("residualize_against: benchmark has zero variance over the window")

    out = {}
    for col in returns_window.columns:
        x = returns_window[col]
        x_c = x - x.mean()
        beta = float((x_c * bench_c).mean() / bench_var)
        alpha = float(x.mean() - beta * bench.mean())
        out[col] = x - (alpha + beta * bench)
    return pd.DataFrame(out, index=returns_window.index)


def compute_topology(returns_window: pd.DataFrame, absorption_top_fraction: float = 0.2) -> TopologySnapshot:
    """Run VR persistence (maxdim=1) on the correlation-distance cloud of
    `returns_window` and summarize it into a `TopologySnapshot`.
    """
    tickers = list(returns_window.columns)
    n = len(tickers)
    if n < 3:
        raise ValueError(f"compute_topology: need >=3 points, got {n}")

    dist_df, corr = correlation_distance(returns_window)
    dist = dist_df.values

    result = ripser(dist, distance_matrix=True, maxdim=1)
    dgms = result["dgms"]
    h0 = dgms[0]
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    # Finite bars only: an infinite death (== inf) marks an essential class
    # that survives to the end of the filtration. For a finite VR complex
    # built up to maxdim=1, H1 essential classes do not occur (only H0 has
    # exactly one, for the final connected component); we filter
    # defensively in case of numerical edge cases.
    if h1.size:
        finite_mask = np.isfinite(h1[:, 1])
        h1_finite = h1[finite_mask]
    else:
        h1_finite = h1

    total_h1 = float((h1_finite[:, 1] - h1_finite[:, 0]).sum()) if h1_finite.size else 0.0

    off_diag = corr.values[~np.eye(n, dtype=bool)]
    rho_bar = float(np.mean(off_diag))
    ar = absorption_ratio(corr, top_fraction=absorption_top_fraction)

    return TopologySnapshot(
        n_points=n,
        tickers=tickers,
        corr=corr,
        dist=dist,
        h0_diagram=h0,
        h1_diagram=h1_finite,
        total_h1_persistence=total_h1,
        n_h1_features=int(h1_finite.shape[0]),
        rho_bar=rho_bar,
        absorption_ratio=ar,
    )
