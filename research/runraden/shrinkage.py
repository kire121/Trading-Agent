"""Empirical-Bayes shrinkage of per-cell residual means.

ghat(w) = lambda_w * mean(residual | word = w),  lambda_w = n_w / (n_w + kappa)

Cells with few observations shrink toward 0 (the additive model's residual
grand mean is ~0 by OLS construction with an intercept, so "toward 0" is
"toward the additive model's own baseline"). Unseen cells (n_w = 0) get
lambda_w = 0, i.e. ghat = 0, which falls out of the formula automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cell_shrinkage(words: list[tuple], residuals: np.ndarray, kappa: float) -> dict[tuple, float]:
    """Compute EB-shrunk cell means from a (words, residuals) training sample."""
    df = pd.DataFrame({"word": words, "resid": residuals})
    grp = df.groupby("word")["resid"].agg(["mean", "count"])
    lam = grp["count"] / (grp["count"] + kappa)
    ghat = lam * grp["mean"]
    return dict(zip(ghat.index, ghat.to_numpy(dtype=float)))


def lambda_weight(n_w: int, kappa: float) -> float:
    return n_w / (n_w + kappa) if (n_w + kappa) > 0 else 0.0
