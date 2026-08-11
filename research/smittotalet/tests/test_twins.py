import numpy as np
import pandas as pd

from .. import twins


def test_quantile_map_preserves_reference_distribution():
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.uniform(0.3, 1.3, len(idx)), index=idx)
    raw = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    mapped = twins.quantile_map_to(reference, raw)
    # same empirical CDF: sorted mapped values ~= sorted reference values
    ref_sorted = np.sort(reference.reindex(mapped.dropna().index).to_numpy())
    mapped_sorted = np.sort(mapped.dropna().to_numpy())
    assert np.allclose(ref_sorted, mapped_sorted, atol=1e-6)


def test_quantile_map_preserves_raw_ordering():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    reference = pd.Series(np.linspace(0.3, 1.3, 50), index=idx)
    raw = pd.Series(np.arange(50), index=idx)  # strictly increasing
    mapped = twins.quantile_map_to(reference, raw)
    assert (mapped.diff().dropna() >= 0).all()


def test_broadcast_aggregate_series_is_mean_abs_return():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.0, 0.03, -0.01],
                             "B": [-0.01, 0.02, 0.0, -0.03, 0.01]}, index=idx)
    agg = twins.broadcast_aggregate_series(returns)
    assert np.allclose(agg.to_numpy(), returns.abs().mean(axis=1).to_numpy())
