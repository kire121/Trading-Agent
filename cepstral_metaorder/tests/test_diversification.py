import numpy as np
import pandas as pd

from cepstral_metaorder import diversification as div


def test_beta_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=300, freq="B")
    market = pd.Series(rng.normal(0, 0.01, 300), index=dates)
    strat = 0.10 * market + pd.Series(rng.normal(0, 0.002, 300), index=dates)
    result = div.beta_to_market(strat, market)
    assert abs(result["beta"] - 0.10) < 0.03
    assert result["passed"] is True  # |0.10| < 0.15 threshold


def test_beta_flags_high_market_exposure():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-02", periods=300, freq="B")
    market = pd.Series(rng.normal(0, 0.01, 300), index=dates)
    strat = 0.8 * market + pd.Series(rng.normal(0, 0.002, 300), index=dates)
    result = div.beta_to_market(strat, market)
    assert result["passed"] is False


def test_tsmom_proxy_is_flat_when_no_trend():
    dates = pd.date_range("2020-01-02", periods=400, freq="B")
    flat = pd.DataFrame({"date": dates, "adjusted_close": np.full(400, 100.0)})
    proxy = div.tsmom_proxy_returns({"A": flat, "B": flat}, lookback_days=252)
    assert proxy.dropna().abs().max() < 1e-9


def test_correlation_to_tsmom_recovers_known_correlation():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2024-01-02", periods=300, freq="B")
    tsmom = pd.Series(rng.normal(0, 0.01, 300), index=dates)
    independent = pd.Series(rng.normal(0, 0.01, 300), index=dates)
    result = div.correlation_to_tsmom(independent, tsmom)
    assert abs(result["corr"]) < 0.25
    assert result["passed"] is True
