import numpy as np

from nulls import (
    circular_block_bootstrap_1d,
    circular_block_bootstrap_columns,
    permute_words_within_week,
)


def test_permute_words_within_week_preserves_multiset():
    rng = np.random.default_rng(0)
    words = [("+", "-", "+", "-", "+"), ("-", "-", "-", "+", "+")]
    permuted = permute_words_within_week(words, rng)
    for orig, perm in zip(words, permuted):
        assert sorted(orig) == sorted(perm)


def test_permute_words_within_week_actually_shuffles_over_many_draws():
    rng = np.random.default_rng(0)
    word = ("+", "+", "-", "-", "-")
    draws = [permute_words_within_week([word], rng)[0] for _ in range(200)]
    assert len(set(draws)) > 1  # not a no-op


def test_circular_block_bootstrap_1d_preserves_length_and_values():
    rng = np.random.default_rng(0)
    x = np.arange(20, dtype=float)
    out = circular_block_bootstrap_1d(x, block_size=5, rng=rng)
    assert len(out) == len(x)
    assert set(out.tolist()) <= set(x.tolist())


def test_circular_block_bootstrap_columns_independent_per_column():
    rng = np.random.default_rng(1)
    mat = np.arange(40, dtype=float).reshape(20, 2)
    out = circular_block_bootstrap_columns(mat, block_size=4, rng=rng)
    assert out.shape == mat.shape
    # Column 0's resampled values must still all come from column 0's original values.
    assert set(out[:, 0].tolist()) <= set(mat[:, 0].tolist())
    assert set(out[:, 1].tolist()) <= set(mat[:, 1].tolist())
