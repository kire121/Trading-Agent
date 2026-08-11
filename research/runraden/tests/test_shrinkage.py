import numpy as np

from shrinkage import cell_shrinkage, lambda_weight


def test_lambda_weight_limits():
    assert lambda_weight(0, 300) == 0.0
    assert np.isclose(lambda_weight(300, 300), 0.5)
    assert lambda_weight(10_000_000, 300) > 0.999


def test_cell_shrinkage_shrinks_low_n_more_than_high_n():
    rng = np.random.default_rng(0)
    words = ["A"] * 10 + ["B"] * 1000
    resid = np.concatenate([
        rng.normal(0.5, 0.01, 10),   # high true mean, few obs
        rng.normal(0.5, 0.01, 1000),  # same true mean, many obs
    ])
    ghat = cell_shrinkage(list(words), resid, kappa=300)
    # Both raw means are ~0.5, but the low-n cell should be shrunk much
    # further toward 0 than the high-n cell.
    assert abs(ghat["A"]) < abs(ghat["B"])
    # lambda_B = 1000/(1000+300) ~= 0.77 -> ghat_B ~= 0.77*0.5 ~= 0.385
    assert ghat["B"] > 0.35
    assert ghat["A"] < 0.5 * (10 / (10 + 300)) + 0.05  # lambda_A ~= 0.032 -> ghat_A small, with noise slack


def test_cell_shrinkage_unseen_cell_is_zero_by_construction():
    ghat = cell_shrinkage(["A", "A", "B"], np.array([1.0, 1.0, -1.0]), kappa=300)
    assert "C" not in ghat  # never observed -> not in the map; caller defaults to 0.0
