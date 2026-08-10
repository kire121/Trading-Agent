"""
Dammluckan -- block-bootstrap null primitives.

A single circular block-bootstrap resampler of daily log returns, used for:
  (a) calibrating the band constant c (signal.calibrate_band_constant)
  (b) calibrating per-asset theta_i (signal.calibrate_theta)
  (c) the Steg-1 estimator-level IC null test (robustness.ic_null_test)
  (d) null-twin #3, "block-permuted returns per asset -> the full rule set on
      synthetic paths" (battery.block_permuted_returns_null)

Block length is config.BLOCK_LENGTH (~1 trading month), matching the
convention Formdriften's own estimator null used for the same purpose.
"""
import numpy as np
import pandas as pd

from . import config


def block_bootstrap_indices(n_obs: int, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block-bootstrap index array of len `length`, drawn from a
    series of length n_obs (wraps around so every start point is usable)."""
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, n_obs, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block) % n_obs) for s in starts])[:length]
    return idx


def block_bootstrap_returns(returns: np.ndarray, length: int, rng: np.random.Generator,
                             block: int = config.BLOCK_LENGTH) -> np.ndarray:
    """Circular block-bootstrap resample of a 1D return array to `length`."""
    idx = block_bootstrap_indices(len(returns), length, block, rng)
    return returns[idx]


def synthetic_price_path(returns: np.ndarray, length: int, rng: np.random.Generator,
                          start_price: float = 100.0, block: int = config.BLOCK_LENGTH) -> np.ndarray:
    """Reconstruct a synthetic raw-close path from a block-bootstrapped log-return draw."""
    r = block_bootstrap_returns(returns, length, rng, block=block)
    log_path = np.log(start_price) + np.concatenate([[0.0], np.cumsum(r)])
    return np.exp(log_path)


def _asset_log_returns(panel, ticker) -> np.ndarray:
    s = panel.raw_close[ticker].dropna()
    r = np.log(s / s.shift(1)).dropna().to_numpy()
    return r


def per_asset_null_occupation(panel, ticker: str, n: int, c: float, side: str,
                               n_draws: int, seed: int, path_length: int = None) -> np.ndarray:
    """
    Pooled null distribution of O+ (side='high') or O- (side='low') for one
    asset: n_draws synthetic block-bootstrap price paths of the asset's own
    IS return history, each scored with the real (n, c) occupation formula,
    pooled (post warm-up) into one array.
    """
    from . import signal  # local import: signal.py imports nulls.py

    r = _asset_log_returns(panel, ticker)
    if len(r) <= n + config.VOL_LOOKBACK:
        return np.array([])
    length = path_length or len(r) + 1
    rng = np.random.default_rng(seed)
    pooled = []
    occ_fn = signal._occupation_high_1d if side == "high" else signal._occupation_low_1d
    for draw in range(n_draws):
        path = synthetic_price_path(r, length - 1, rng, block=config.BLOCK_LENGTH)
        log_r = np.diff(np.log(path))
        band_vol = pd.Series(log_r).rolling(
            config.VOL_LOOKBACK, min_periods=config.VOL_MIN_PERIODS
        ).std(ddof=1).shift(1).to_numpy()
        band_vol = np.concatenate([[np.nan], band_vol])   # align to price path length
        occ = occ_fn(path, band_vol, n, c)
        pooled.append(occ[~np.isnan(occ)])
    return np.concatenate(pooled) if pooled else np.array([])


def pooled_null_occupation(panel, n: int, c: float, side: str, n_draws: int, seed: int) -> np.ndarray:
    """Same as per_asset_null_occupation but pooled across every ticker in
    the panel (used to calibrate the shared band constant c)."""
    pooled = []
    for i, ticker in enumerate(panel.tickers):
        vals = per_asset_null_occupation(panel, ticker, n, c, side, n_draws, seed=seed + 7919 * i)
        pooled.append(vals)
    return np.concatenate(pooled) if pooled else np.array([])
