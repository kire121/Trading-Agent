import numpy as np
import pandas as pd
import pytest

from oglegrinden.data import Panel
from oglegrinden.signal import weekly_fridays
from oglegrinden.grid import build_grid_spec, run_grid
from oglegrinden.stats import deflated_sharpe_ratio


def _make_synthetic_panel(n_tickers=20, n_days=700, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    histories = {}
    tickers = [f"T{i:02d}" for i in range(n_tickers)] + ["SPY"]
    for tk in tickers:
        base = 100.0
        if tk != "SPY":
            noise = rng.normal(0, 0.01, n_days)
            dev = np.zeros(n_days)
            for i in range(1, n_days):
                dev[i] = 0.7 * dev[i - 1] + noise[i]
            trend = np.linspace(0, 0.1, n_days)
            log_px = np.log(base) + trend + dev
        else:
            log_px = np.log(base) + np.cumsum(rng.normal(0.0002, 0.01, n_days))
        close = np.exp(log_px)
        open_ = close * (1 + rng.normal(0, 0.001, n_days))
        high = np.maximum(open_, close) * 1.002
        low = np.minimum(open_, close) * 0.998
        volume = rng.uniform(5_000_000, 8_000_000, n_days)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "adjclose": close},
            index=dates,
        )
        histories[tk] = df
    return Panel(histories)


def test_build_grid_spec_has_expected_shape():
    spec = build_grid_spec()
    assert len(spec) == 30
    kinds = {v["kind"] for v in spec}
    assert kinds == {"base", "beta_hedge", "residual_corr"}
    base = [v for v in spec if v["kind"] == "base"]
    assert len(base) == 24
    assert len({v["id"] for v in spec}) == len(spec)  # unique ids


@pytest.mark.slow
def test_run_grid_end_to_end_on_synthetic_panel():
    panel = _make_synthetic_panel(n_tickers=20, n_days=700, seed=42)
    all_fridays = weekly_fridays(panel.close.index)

    grid_run = run_grid(panel, all_fridays, min_history_years=0.5)
    table = grid_run.table

    assert len(table) == 30
    assert table["n_obs"].gt(0).any()
    assert table["sharpe"].notna().all()

    # DSR should compute without error from the grid's per-period Sharpes
    trial_sharpes = (table["sharpe"] / np.sqrt(52)).values  # de-annualize back to per-period
    dsr = deflated_sharpe_ratio(
        observed_sharpe_per_period=trial_sharpes.max(),
        trial_sharpes_per_period=trial_sharpes,
        n_obs=int(table["n_obs"].max()),
    )
    assert 0.0 <= dsr["dsr"] <= 1.0
