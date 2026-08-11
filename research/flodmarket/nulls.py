"""Block-PERMUTATION null (spec SS0b/SS1/SS8 T3: "s-serien blockpermuterad
inom tillgang (block 21d)... bevarar marginal+ACF+FE" -- distinct from
block-BOOTSTRAP, spec SS9 Steg5 K5.3, which resamples WITH replacement/
overlap; see lib.bootstrap for that family, reused unmodified for K5.3).

Proveniens: after reading all four candidates docs/INSTRUKTION.md SS7 lists
under the "block-PERMUTATION utan aterlaggning" family
(vindkastet/run_gate_checks.py::block_shuffle, cepstral_metaorder/
validation.py, omori/nulls.py::block_permutation_ic, runraden/nulls.py::N1),
only ONE is actually a no-replacement permutation of a fixed block
partition: cepstral_metaorder/validation.py::block_shuffle_signal /
block_shuffle_null_dispersion, branch claude/cepstral-metaorder-detection-
b8nvwb, commit 0edbc4a (`order = rng.permutation(n_blocks)`, applied
per-column/per-symbol independently). The other three draw i.i.d. random
BLOCK START POSITIONS with `rng.integers(...)` -- mechanically a moving-
block BOOTSTRAP (blocks can overlap/repeat) despite being named "shuffle",
confirmed by direct code inspection (session-logged in AVVIKELSER.md as a
precision correction to INSTRUKTION.md's own categorization). Reused/
generalized here: renamed away from S_bar-specific naming, made to operate
on a MultiIndex(ticker,date) panel with an explicit `block_len` in trading
days rather than a fixed block-count.
"""
import numpy as np
import pandas as pd


def block_permute_1d(arr: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Chops `arr` into contiguous blocks of length `block_len` (last block
    may be shorter) and reorders the BLOCKS (not their contents) uniformly
    at random, without replacement -- every original value appears exactly
    once, in a different position (unless the permutation is the
    identity). Preserves the exact marginal empirical distribution and each
    block's own internal (short-range) serial structure; destroys
    block-to-block ordering / any predictive link keyed to real calendar
    position."""
    n = len(arr)
    if n <= 1:
        return arr.copy()
    n_blocks = int(np.ceil(n / block_len))
    order = rng.permutation(n_blocks)
    blocks = [arr[b * block_len: min((b + 1) * block_len, n)] for b in range(n_blocks)]
    return np.concatenate([blocks[b] for b in order])[:n]


def block_permute_within_ticker(s: pd.Series, block_len: int, rng: np.random.Generator) -> pd.Series:
    """Applies block_permute_1d independently to each ticker's own time
    series (spec: "blockpermuterad inom tillgang" -- never permutes across
    a ticker boundary). `s` must carry a MultiIndex with a 'ticker' level."""
    if not (isinstance(s.index, pd.MultiIndex) and "ticker" in (s.index.names or [])):
        raise ValueError("block_permute_within_ticker requires a MultiIndex(ticker,date) Series")

    def _one(series: pd.Series) -> pd.Series:
        permuted = block_permute_1d(series.to_numpy(), block_len, rng)
        return pd.Series(permuted, index=series.index)

    return s.groupby(level="ticker", group_keys=False).apply(_one)


def empirical_pvalue(observed: float, null_draws) -> float:
    """Laplace-smoothed empirical p-value (+1 numerator/denominator), same
    convention as lib.bootstrap.empirical_pvalue -- reused here directly
    for consistency rather than reimplemented; kept as a thin wrapper so
    callers only need to import research.flodmarket.nulls."""
    from lib.bootstrap import empirical_pvalue as _lib_empirical_pvalue
    return _lib_empirical_pvalue(observed, null_draws)
