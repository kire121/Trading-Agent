"""Null-hypothesis baselines: N1 (within-week permutation, full refit) and
N2 (circular block bootstrap).

N1 -- the primary null for "does word ORDER carry information": for every
(asset, week) independently, randomly permute the order of that week's own
daily-return signs. This preserves the week's multiset of signs (hence its
own compounded week_return, and therefore every row's regression *target*,
which depends on the *following* week) while destroying which cell of the
32/16-cell table the week lands in. The full pipeline (walk-forward OLS +
EB shrinkage + OOF scoring) is then refit from scratch on the permuted
words, and the same OOF statistic used on real data is recomputed. Doing
this many times gives the null distribution K1a compares against (its p95).

N2 -- circular block bootstrap over weeks: draws overlapping blocks of
consecutive weeks (wrapping around the end of the sample) and concatenates
them into a resampled series of the original length. Used (a) as a
column-independent common-factor null for K1b (each asset's ghat column is
block-resampled independently, decoupling any genuine same-week
co-movement while preserving each asset's own serial dependence), and (b)
as a general block-bootstrap CI tool elsewhere in the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from additive_model import WalkForwardAdditiveModel


def permute_words_within_week(words: list[tuple], rng: np.random.Generator) -> list[tuple]:
    """Independently permute the letters of every word (row) in `words`."""
    if not words:
        return []
    arr = np.array([list(w) for w in words])
    permuted = rng.permuted(arr, axis=1)
    return [tuple(row) for row in permuted]


def n1_draw(df_table: pd.DataFrame, word_len: int, kappa: float, burn_in_years: int,
            rng: np.random.Generator) -> pd.DataFrame:
    """One N1 permutation draw: full pipeline refit on within-week-permuted words."""
    permuted = df_table.copy()
    permuted["word"] = permute_words_within_week(df_table["word"].tolist(), rng)
    model = WalkForwardAdditiveModel(word_len, kappa=kappa, burn_in_years=burn_in_years)
    model.fit_walkforward(permuted)
    scored = model.score(permuted)
    permuted = permuted.join(scored)
    return permuted


def n1_null_distribution(df_table: pd.DataFrame, word_len: int, kappa: float, burn_in_years: int,
                          stat_fn, n_draws: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stats = np.full(n_draws, np.nan)
    for d in range(n_draws):
        permuted = n1_draw(df_table, word_len, kappa, burn_in_years, rng)
        stats[d] = stat_fn(permuted)
    return stats


def n1_pooled_null_distribution(tables: list[tuple[pd.DataFrame, int]], kappa: float,
                                 burn_in_years: int, stat_fn_combined, n_draws: int = 200,
                                 seed: int = 0) -> np.ndarray:
    """Pooled N1 null across several tables (e.g. the 5d and 4d word tables)
    permuted together within the same draw, so `stat_fn_combined` (which
    receives the concatenation of all permuted+rescored tables) reflects the
    same "pooled over the whole panel" statistic used on real data."""
    rng = np.random.default_rng(seed)
    stats = np.full(n_draws, np.nan)
    for d in range(n_draws):
        parts = []
        for df_table, word_len in tables:
            if df_table.empty:
                continue
            parts.append(n1_draw(df_table, word_len, kappa, burn_in_years, rng))
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        stats[d] = stat_fn_combined(combined)
    return stats


def circular_block_bootstrap_1d(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Resample a 1-D array of length T into a new length-T array using
    circular (wrap-around) overlapping blocks."""
    T = len(x)
    if T == 0:
        return x.copy()
    block_size = max(1, min(block_size, T))
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.integers(0, T, size=n_blocks)
    out = []
    for s in starts:
        idx = (np.arange(block_size) + s) % T
        out.append(x[idx])
    return np.concatenate(out)[:T]


def circular_block_bootstrap_columns(mat: np.ndarray, block_size: int,
                                      rng: np.random.Generator) -> np.ndarray:
    """Block-bootstrap each column of a (T, N) matrix independently (breaks
    cross-sectional/same-row alignment while preserving each column's own
    serial dependence structure)."""
    T, N = mat.shape
    out = np.empty_like(mat)
    for j in range(N):
        out[:, j] = circular_block_bootstrap_1d(mat[:, j], block_size, rng)
    return out


def circular_block_bootstrap_rows(mat: np.ndarray, block_size: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """Block-bootstrap the row (time) axis of a (T, N) matrix, keeping each
    row's cross-section intact (used where cross-sectional structure should
    be preserved but the time-ordering should be scrambled)."""
    T, N = mat.shape
    if T == 0:
        return mat.copy()
    block_size = max(1, min(block_size, T))
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.integers(0, T, size=n_blocks)
    row_idx = []
    for s in starts:
        row_idx.extend(((np.arange(block_size) + s) % T).tolist())
    row_idx = np.array(row_idx[:T])
    return mat[row_idx, :]
