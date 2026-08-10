import numpy as np
import pandas as pd
import pytest

from research.omori import config, data, events


def _tiny_panel(n=200, tickers=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, size=(n, len(tickers))), axis=0),
                          index=idx, columns=tickers)
    volume = pd.DataFrame(rng.integers(1_000_000, 2_000_000, size=(n, len(tickers))).astype(float),
                           index=idx, columns=tickers)
    panel = data.Panel.__new__(data.Panel)
    panel.label = "tiny"
    panel.raw = {
        "open": close.copy(), "high": close * 1.01, "low": close * 0.99,
        "close": close, "adjusted_close": close, "volume": volume,
    }
    panel.tickers = list(tickers)
    panel.index = idx
    return panel


class TestRollingStats:
    def test_z_score_is_causal_no_lookahead(self):
        """Injecting a huge FUTURE spike must not change today's z-score --
        the rolling baseline only looks backward."""
        panel = _tiny_panel(n=250)
        dv = panel.dollar_volume()
        z_before = data.rolling_median_mad_z(dv, 120)

        dv2 = dv.copy()
        dv2.iloc[-1] = dv2.iloc[-1] * 1000  # inject an enormous spike on the LAST day only
        z_after = data.rolling_median_mad_z(dv2, 120)

        # every day strictly before the injected spike must be unaffected
        pd.testing.assert_frame_equal(z_before.iloc[:-1], z_after.iloc[:-1])

    def test_z_score_nan_before_lookback_window_available(self):
        panel = _tiny_panel(n=50)
        dv = panel.dollar_volume()
        z = data.rolling_median_mad_z(dv, 120)
        assert z.iloc[:120].isna().all().all()

    def test_sigma_excludes_day_t_itself(self):
        panel = _tiny_panel(n=200)
        r = panel.simple_returns()
        sigma_before = data.rolling_return_sigma(r, 60)
        r2 = r.copy()
        r2.iloc[-1] = r2.iloc[-1] + 5.0  # huge outlier return on the last day only
        sigma_after = data.rolling_return_sigma(r2, 60)
        pd.testing.assert_frame_equal(sigma_before.iloc[:-1], sigma_after.iloc[:-1])


class TestClustering:
    def test_single_candidate_survives(self):
        assert events.cluster_same_day([("A", 5.0)], pd.DataFrame()) == ["A"]

    def test_uncorrelated_pair_both_survive(self):
        corr = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"])
        survivors = events.cluster_same_day([("A", 5.0), ("B", 6.0)], corr, threshold=0.7)
        assert set(survivors) == {"A", "B"}

    def test_correlated_pair_keeps_only_highest_z(self):
        corr = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], index=["A", "B"], columns=["A", "B"])
        survivors = events.cluster_same_day([("A", 5.0), ("B", 6.0)], corr, threshold=0.7)
        assert survivors == ["B"]

    def test_transitive_cluster_of_three(self):
        # A-B correlated, B-C correlated, A-C not directly measured (NaN) --
        # still one connected component via B, one survivor overall.
        corr = pd.DataFrame(
            [[1.0, 0.9, np.nan], [0.9, 1.0, 0.85], [np.nan, 0.85, 1.0]],
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        survivors = events.cluster_same_day([("A", 3.0), ("B", 9.0), ("C", 4.0)], corr, threshold=0.7)
        assert survivors == ["B"]

    def test_negative_correlation_below_threshold_in_magnitude_clusters(self):
        corr = pd.DataFrame([[1.0, -0.8], [-0.8, 1.0]], index=["A", "B"], columns=["A", "B"])
        survivors = events.cluster_same_day([("A", 5.0), ("B", 2.0)], corr, threshold=0.7)
        assert survivors == ["A"]  # |rho|=0.8 > 0.7 clusters even though rho is negative


class TestEventFields:
    def test_excess_volume_series_baseline_is_causal(self):
        panel = _tiny_panel(n=250)
        ef = events.EventFields(panel)
        e = ef.excess_volume_series("A")
        assert e.iloc[:120].isna().all()
        assert e.iloc[130:].notna().any()

    def test_candidates_long_columns(self):
        panel = _tiny_panel(n=250)
        ef = events.EventFields(panel)
        cands = ef.candidates_long(z_threshold=0.5, sigma_mult=0.1)  # loose thresholds -> some hits
        assert set(["date", "ticker", "date_idx", "r0", "volume_z", "direction"]) <= set(cands.columns)
        assert (cands["direction"].isin([-1.0, 1.0])).all()
