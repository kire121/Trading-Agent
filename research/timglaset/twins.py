"""T0-T3 null/benchmark twins (spec §7) and the T2 shuffle-clock mechanism.

T0/T1/T3 are thin compositions of opclock.py + baseengine.py (see spec's
own "Kalendertvilling: compute_op_ewma(returns, tau=ones_like, ...) --
samma kodväg, ingen parallellimplementation" and "Variansklocka:
compute_tau(m5, ...)"). T2's within-instrument block permutation is NEW
code: exhaustive search (docs/INSTRUKTION.md avsnitt 7) confirms no
"block-permutation without replacement" primitive was migrated to lib/ --
the mechanically similar-sounding lib.bootstrap functions are all block
BOOTSTRAP (overlapping blocks, random starts, WITH replacement), which
INSTRUKTION.md explicitly flags as non-interchangeable with permutation.
The one genuine "permute block ORDER, no replacement" example in the whole
11-branch corpus is the block-reassembly core inside
research/omori/nulls.py::block_permutation_ic, branch
claude/omori-exit-strategy-du0qvg, commit 3553ede -- generalized below
or (a) per-column independence ("within-instrument", omori's version
pools a single 1-D array) and (b) shuffling tau itself rather than a
target series, per spec §7 T2 / §13 "Shuffleklocka".

Liveness assertions (unconditional, spec §7/§1.2.7) reuse
lib.twins.twin_is_alive verbatim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.twins import twin_is_alive

from . import baseengine
from . import config
from . import opclock


def block_permute_columns(df: pd.DataFrame, block_len: int, rng: np.random.Generator) -> pd.DataFrame:
    """Within-column (within-instrument) block-order permutation, no
    replacement: split into contiguous blocks of `block_len` (last block
    possibly shorter), permute the block ORDER independently per column,
    reassemble. Exactly preserves each column's own empirical marginal
    distribution -- mechanically distinct from block bootstrap (see module
    docstring)."""
    n = len(df)
    n_blocks = int(np.ceil(n / block_len))
    block_ids = np.repeat(np.arange(n_blocks), block_len)[:n]
    arr = df.to_numpy(dtype=float)
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        perm_order = rng.permutation(n_blocks)
        col = arr[:, j]
        pieces = [col[block_ids == b] for b in perm_order]
        out[:, j] = np.concatenate(pieces)[:n]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ---------------------------------------------------------------------------
# T1: calendar twin -- identical code path, tau == 1.
# ---------------------------------------------------------------------------
def build_t1(panel, returns: pd.DataFrame, cell: config.GridCell, is_start, is_end,
             apply_costs: bool = True):
    tau_cal = opclock.calendar_twin_tau(returns)
    M, V, T = opclock.compute_op_ewma(returns, tau_cal, cell.hl_op)
    weights, rets, k = baseengine.build_clock_book(panel, M, V, T, cell.hl_op, cell.f,
                                                     is_start, is_end, apply_costs)
    return {"M": M, "V": V, "T": T, "weights": weights, "returns": rets, "k": k}


# ---------------------------------------------------------------------------
# Primary op-clock signal -- real, split-adjusted volume.
# ---------------------------------------------------------------------------
def build_primary(panel, returns: pd.DataFrame, volume: pd.DataFrame, cell: config.GridCell,
                   is_start, is_end, apply_costs: bool = True):
    tau_op = opclock.compute_tau(volume, window=config.NORMALIZER_WINDOW, cap=cell.c)
    M, V, T = opclock.compute_op_ewma(returns, tau_op, cell.hl_op)
    weights, rets, k = baseengine.build_clock_book(panel, M, V, T, cell.hl_op, cell.f,
                                                     is_start, is_end, apply_costs)
    return {"tau": tau_op, "M": M, "V": V, "T": T, "weights": weights, "returns": rets, "k": k}


# ---------------------------------------------------------------------------
# T3: variance clock -- tau derived from m5 = 5d rolling mean of r^2.
# ---------------------------------------------------------------------------
def build_t3(panel, returns: pd.DataFrame, cell: config.GridCell, is_start, is_end,
             apply_costs: bool = True):
    m5 = opclock.variance_clock_input(returns, window=config.T3_VARIANCE_WINDOW)
    tau_var = opclock.compute_tau(m5, window=config.NORMALIZER_WINDOW, cap=cell.c)
    M, V, T = opclock.compute_op_ewma(returns, tau_var, cell.hl_op)
    weights, rets, k = baseengine.build_clock_book(panel, M, V, T, cell.hl_op, cell.f,
                                                     is_start, is_end, apply_costs)
    return {"tau": tau_var, "M": M, "V": V, "T": T, "weights": weights, "returns": rets, "k": k}


# ---------------------------------------------------------------------------
# T2: shuffle clock -- real tau, block-permuted within-instrument, full
# pipeline recompute per draw. Draw-by-draw generator (not a single big
# batch) so we never hold 200 full M/V/T panels in memory at once.
# ---------------------------------------------------------------------------
def t2_draw_signal(returns: pd.DataFrame, volume: pd.DataFrame, cell: config.GridCell, seed: int):
    """One T2 draw: shuffle tau, full recompute of (M, V, T) and raw z."""
    tau_real = opclock.compute_tau(volume, window=config.NORMALIZER_WINDOW, cap=cell.c)
    rng = np.random.default_rng(seed)
    tau_shuf = block_permute_columns(tau_real, config.T2_BLOCK_LEN, rng)
    M, V, T = opclock.compute_op_ewma(returns, tau_shuf, cell.hl_op)
    z = opclock.compute_raw_z(M, V, T, cell.hl_op)
    f_z = opclock.compute_signal(M, V, T, cell.hl_op, cell.f)
    return z, f_z


def t2_draws(returns: pd.DataFrame, volume: pd.DataFrame, cell: config.GridCell,
             seeds=None):
    """Yield (seed, z, f_z) for each T2 draw, computed lazily."""
    seeds = seeds or config.T2_SEEDS
    for seed in seeds:
        z, f_z = t2_draw_signal(returns, volume, cell, seed)
        yield seed, z, f_z
