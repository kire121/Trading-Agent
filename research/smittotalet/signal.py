"""Cori (EpiEstim) effective-reproduction-number estimator and the G_t tilt.

    Lambda_t = sum_{s=1..10} w_s X_{t-s}                       (frozen Omori kernel)
    R_hat_t  = (1 + sum_{u=t-tau+1..t} X_u) / (1 + sum_u Lambda_u)   (Gamma(1,1) prior)
    G_t      = clip((1/R_hat_t)^kappa, 0.3, 1.3)

All closed-form; no MCMC, no fitting at signal-construction time -- the only
fitted object (the Omori kernel exponent) is frozen in config.py, imported
from Efterskalvsklockan's own frozen output.
"""
import numpy as np
import pandas as pd

from . import config


def omori_kernel(lags: int = config.OMORI_KERNEL_LAGS, p: float = config.OMORI_KERNEL_P,
                  c: float = config.OMORI_KERNEL_C) -> np.ndarray:
    """w_s ~ (s+c)^-p for s=1..lags, normalized to sum to 1."""
    s = np.arange(1, lags + 1, dtype=float)
    raw = (s + c) ** (-p)
    return raw / raw.sum()


def branching_intensity(x_t: pd.Series, w: np.ndarray) -> pd.Series:
    """Lambda_t = sum_s w_s X_{t-s}, causal (only uses strictly past X)."""
    lags = len(w)
    lam = pd.Series(0.0, index=x_t.index)
    x_filled = x_t.fillna(0.0)
    for s, w_s in enumerate(w, start=1):
        lam = lam.add(w_s * x_filled.shift(s), fill_value=0.0)
    # Lambda undefined until `lags` full days of prior X exist.
    lam.iloc[:lags] = np.nan
    return lam


def reproduction_number(x_t: pd.Series, lam_t: pd.Series, tau: int) -> pd.Series:
    """R_hat_t, closed-form Cori posterior mean under a Gamma(1,1) prior."""
    x_sum = x_t.fillna(0.0).rolling(tau, min_periods=tau).sum()
    lam_sum = lam_t.fillna(0.0).rolling(tau, min_periods=tau).sum()
    r_hat = (1.0 + x_sum) / (1.0 + lam_sum)
    r_hat[lam_t.isna() | x_t.isna()] = np.nan
    return r_hat


def tilt(r_hat: pd.Series, kappa: float, low: float = config.G_CLIP_LOW,
         high: float = config.G_CLIP_HIGH) -> pd.Series:
    """G_t = clip((1/R_hat_t)^kappa, low, high)."""
    g = (1.0 / r_hat) ** kappa
    return g.clip(lower=low, upper=high)


def build(x_t: pd.Series, q: int, tau: int, kappa: float, p: float = config.OMORI_KERNEL_P,
          c: float = config.OMORI_KERNEL_C, lags: int = config.OMORI_KERNEL_LAGS):
    """Full pipeline: kernel -> Lambda_t -> R_hat_t -> G_t. Returns (r_hat, g_t, lam_t)."""
    w = omori_kernel(lags=lags, p=p, c=c)
    lam_t = branching_intensity(x_t, w)
    r_hat = reproduction_number(x_t, lam_t, tau)
    g_t = tilt(r_hat, kappa)
    return r_hat, g_t, lam_t
