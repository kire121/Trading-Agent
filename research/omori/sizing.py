"""Position sizing: vol-target base size times the Omori tilt, never a gate
(every event that fires is traded, at some size)."""
import numpy as np

from research.omori import config, signal


def raw_target_weight(direction, p_tilde, p_star, sigma_hat_daily,
                       vol_target=config.VOL_TARGET_DAILY):
    """w_i (fraction of NAV, signed) BEFORE the portfolio-level gross cap is
    applied: tilt * (vol_target / sigma_hat_i). sigma_hat_i is the
    instrument's own trailing daily-return vol (config.VOL_TARGET_LOOKBACK,
    causal, as of entry) -- this is what makes the position's own ex-ante
    daily risk contribution equal `vol_target` at tilt magnitude 1.0."""
    tilt = signal.traded_signal(direction, p_tilde, p_star)
    if not np.isfinite(sigma_hat_daily) or sigma_hat_daily <= 0:
        return 0.0
    return tilt * (vol_target / sigma_hat_daily)


def apply_gross_cap(weights, gross_cap=config.GROSS_CAP):
    """Scale a dict/array of signed position weights down proportionally
    (never up) so that sum(|w_i|) <= gross_cap.

    NOTE: `backtest.run_backtest` does NOT call this -- it implements a
    different, headroom-based policy inline (a NEW entry is sized down to
    fit whatever gross-cap room remains; EXISTING positions are never
    rescaled to make room, consistent with sizing being fixed at entry, see
    backtest.py's module docstring). This function documents/tests the
    alternative "rescale everyone proportionally" policy in isolation; it
    is not dead by mistake, just not the one the production backtest uses."""
    weights = np.asarray(weights, dtype=float)
    gross = np.abs(weights).sum()
    if gross <= gross_cap or gross == 0:
        return weights
    return weights * (gross_cap / gross)
