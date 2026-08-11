"""Block-bootstrap / block-shuffle null-generating primitives.

circular_block_shuffle ported from research/dammluckan/nulls.py
(block_bootstrap_indices). Used for: (a) the dispersion null on R_hat's
excursion band ("Efterskalvsklockans check oversatt till tidsdimensionen"),
and (b) the Step-2 bootstrap CI on the net-SR increment.
"""
import numpy as np
import pandas as pd

from . import config
from . import signal as signal_mod


def block_bootstrap_indices(n_obs: int, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, n_obs, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) % n_obs for s in starts])
    return idx[:length]


def circular_block_shuffle(arr: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    idx = block_bootstrap_indices(len(arr), len(arr), block_len, rng)
    return np.asarray(arr)[idx]


def r_hat_dispersion_null(x_t: pd.Series, q: int, tau: int, kappa: float, n_draws: int = 500,
                           block_len: int = 21, seed: int = 0) -> np.ndarray:
    """Circularly block-shuffle X_t, re-run the (fixed, frozen) Cori pipeline
    on each shuffle, and collect the resulting R_hat dispersion (std over
    valid days). This is the null band R_hat's own time-variation must beat.
    """
    rng = np.random.default_rng(seed)
    x_vals = x_t.fillna(0.0).to_numpy()
    out = np.empty(n_draws)
    for i in range(n_draws):
        shuffled = circular_block_shuffle(x_vals, block_len, rng)
        x_shuf = pd.Series(shuffled, index=x_t.index)
        r_hat, _, _ = signal_mod.build(x_shuf, q=q, tau=tau, kappa=kappa)
        out[i] = r_hat.std(skipna=True)
    return out


def block_bootstrap_sharpe_ci(returns: pd.Series, n_draws: int = 1000, block_len: int = 21,
                               seed: int = 0, alpha: float = 0.10):
    """Two-sided (1-alpha) block-bootstrap CI on the annualized Sharpe of `returns`."""
    from . import metrics
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    n = len(r)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        idx = block_bootstrap_indices(n, n, block_len, rng)
        draws[i] = metrics.sharpe(pd.Series(r[idx]))
    draws = draws[np.isfinite(draws)]
    lo = np.quantile(draws, alpha / 2)
    hi = np.quantile(draws, 1 - alpha / 2)
    return lo, hi, draws
