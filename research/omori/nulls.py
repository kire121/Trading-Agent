"""Two null-hypothesis gates that must pass before the backtest itself is
trusted at all:

  1. Estimator null: is there genuine cross-event dispersion in p_hat, or
     would within-event-shuffled (decay-order-destroyed) postvolume produce
     just as much apparent spread? "En klocka med en enda tid ar ingen
     klocka" -- the brief's own framing of what this test protects against.
  2. Step-1 IC tests, run on the SIGNED quantities only (never on bare,
     unsigned p_hat/p_tilde against an undirected return -- see signal.py's
     traded_signal docstring): rank-IC(p_tilde, realized half-life) and
     signed-IC(Z, forward return over the position's own adaptive horizon),
     both block-permutation tested.
"""
import numpy as np
from scipy import stats as scipy_stats

from research.omori import config, signal


def circular_block_shuffle(arr, block_len, rng):
    n = len(arr)
    if n <= 1:
        return arr.copy()
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    out = []
    for s in starts:
        idx = (np.arange(s, s + block_len)) % n
        out.append(arr[idx])
    return np.concatenate(out)[:n]


def estimator_null_test(terminal_fits_list, n_draws=config.N_ESTIMATOR_NULL_DRAWS,
                         subsample_n=config.ESTIMATOR_NULL_SUBSAMPLE,
                         block_len=config.ESTIMATOR_NULL_BLOCK, seed=0):
    identified = [f for f in terminal_fits_list if f["identified"] and f["e_path"] is not None]
    rng = np.random.default_rng(seed)
    if len(identified) > subsample_n:
        sel_idx = rng.choice(len(identified), size=subsample_n, replace=False)
        subsample = [identified[i] for i in sel_idx]
    else:
        subsample = identified

    real_p_hats = np.array([f["p_hat"] for f in subsample])
    real_dispersion = float(np.std(real_p_hats, ddof=1))

    null_dispersions = np.empty(n_draws)
    for d in range(n_draws):
        shuffled_p_hats = []
        for f in subsample:
            e_path = f["e_path"]
            shuffled = circular_block_shuffle(e_path, block_len, rng)
            fit = signal.fit_omori(len(shuffled), shuffled)
            if fit.identified:
                shuffled_p_hats.append(fit.p_hat)
        null_dispersions[d] = np.std(shuffled_p_hats, ddof=1) if len(shuffled_p_hats) > 2 else np.nan

    null_p95 = float(np.nanpercentile(null_dispersions, 95))
    return {
        "real_dispersion": real_dispersion,
        "null_p95_dispersion": null_p95,
        "n_events_used": len(subsample),
        "n_draws": n_draws,
        "passed": bool(real_dispersion > null_p95),
    }


def block_permutation_ic(x, y, dates, n_draws=config.IC_PERMUTATION_DRAWS,
                          block_len=config.BLOCK_LENGTH, seed=0):
    """Spearman rank-IC between x and y with a block-permutation null:
    events are ordered by `dates`, split into contiguous blocks of
    `block_len`, and the block ORDER (not within-block pairing) of y is
    shuffled relative to x -- preserving short-range serial/cross-sectional
    dependence while destroying the x-y association. Two-sided p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(np.asarray(dates)[mask])
    x, y = x[order], y[order]
    n = len(x)
    if n < 20:
        return {"ic": np.nan, "p_value": np.nan, "n": n}

    observed_ic = float(scipy_stats.spearmanr(x, y).correlation)

    n_blocks = int(np.ceil(n / block_len))
    block_ids = np.repeat(np.arange(n_blocks), block_len)[:n]
    rng = np.random.default_rng(seed)
    null_ics = np.empty(n_draws)
    for d in range(n_draws):
        perm_order = rng.permutation(n_blocks)
        y_perm = np.concatenate([y[block_ids == b] for b in perm_order])[:n]
        null_ics[d] = scipy_stats.spearmanr(x, y_perm).correlation

    p_value = float(np.mean(np.abs(null_ics) >= abs(observed_ic)))
    return {"ic": observed_ic, "p_value": p_value, "n": n, "n_draws": n_draws}


def rank_ic_p_tilde_halflife(battery_df, n_draws=config.IC_PERMUTATION_DRAWS, seed=0):
    hl = battery_df.dropna(subset=["p_tilde", "half_life"])
    dates = hl["t0_idx"].values
    return block_permutation_ic(hl["p_tilde"].values, hl["half_life"].values, dates, n_draws, seed=seed)


def signed_ic_z_forward_return(closed_events_df, p_star, n_draws=config.IC_PERMUTATION_DRAWS, seed=1):
    df = closed_events_df
    z = np.array([signal.traded_signal(d, p, p_star) for d, p in zip(df["direction"], df["p_tilde_entry"])])
    fwd = df["gross_return"].values
    dates = df["t0_idx"].values
    return block_permutation_ic(z, fwd, dates, n_draws, seed=seed)
