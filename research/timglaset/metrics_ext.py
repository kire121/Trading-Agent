"""Small statistics primitives referenced by the spec but not migrated to
lib/metrics.py during the 2026-08-11 consolidation (see docs/INSTRUKTION.md
avsnitt 7's own scope statement: lib/metrics.py only covers Sharpe/Sortino/
DSR/Newey-West -- IC and PC1 share were never part of that port). Copied
with provenance rather than reinvented, per rule 2.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def pooled_rank_ic(signal_vals, forward_returns) -> float:
    """Spearman rank correlation between a signal and forward returns,
    computed on the POOLED (all ticker-week observations flattened
    together) sample -- matches spec §8 Steg 2's "poolad veckovis rank-IC".
    Provenance: research/smittotalet/metrics.py::information_coefficient,
    branch claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b
    (ported verbatim; renamed for clarity since this module already has an
    unrelated "IC" concept nowhere else)."""
    df = pd.concat([pd.Series(signal_vals), pd.Series(forward_returns)], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    rho, _ = scipy_stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho)


def pc1_share(matrix: np.ndarray) -> float:
    """Share of total variance explained by the first principal component
    of a (T, N) matrix (columns mean-centred; NaNs filled with their own
    column mean before centring, i.e. "no signal this week" contributes no
    deviation). Provenance: research/runraden/metrics.py::pc1_share, branch
    claude/runraden-vecko-ordning-vvztim, commit a4e0d53 -- ported verbatim.
    Spec §8 Steg 4 / §1.2.6: "PC1-check obligatorisk även för poolade/
    tidsseriestrategier" (Runraden mallkrav 4)."""
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    X = matrix.copy()
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    X = X - X.mean(axis=0, keepdims=True)
    col_var = X.var(axis=0)
    if np.allclose(col_var, 0):
        return np.nan
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total <= 0:
        return np.nan
    return float(eigvals.max() / total)
