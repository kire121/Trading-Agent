import numpy as np
import pandas as pd

from positions import apply_no_trade_band, build_positions, portfolio_vol_est, solve_common_scale


def test_apply_no_trade_band_holds_small_changes_trades_big_ones():
    held = {"A": 0.05, "B": -0.02}
    proposed = {"A": 0.052, "B": 0.10}  # A: tiny change, B: big change
    out = apply_no_trade_band(proposed, held, band=0.01)
    assert out["A"] == 0.05  # held (change 0.002 < band 0.01)
    assert out["B"] == 0.10  # traded (change 0.12 >= band 0.01)


def test_apply_no_trade_band_zero_band_always_trades():
    held = {"A": 0.05}
    proposed = {"A": 0.0500001}
    out = apply_no_trade_band(proposed, held, band=0.0)
    assert out["A"] == 0.0500001


def test_apply_no_trade_band_new_name_has_zero_prior():
    out = apply_no_trade_band({"NEW": 0.2}, {}, band=0.01)
    assert out["NEW"] == 0.2


def test_portfolio_vol_est_zero_correlation_formula():
    w = np.array([0.1, -0.2, 0.3])
    sigma = np.array([0.02, 0.02, 0.02])
    expected = np.sqrt(np.sum((w * sigma) ** 2))
    assert np.isclose(portfolio_vol_est(w, sigma), expected)


def test_solve_common_scale_hits_vol_target_when_not_gross_constrained():
    x_raw = np.array([1.0, -1.0, 0.5])
    sigma_weekly = np.array([0.02, 0.02, 0.02])
    k = solve_common_scale(x_raw, sigma_weekly, target_vol=0.08, max_gross=1000.0)
    scaled = k * x_raw
    assert np.isclose(portfolio_vol_est(scaled, sigma_weekly), 0.08, atol=1e-9)
    assert np.sum(np.abs(scaled)) <= 1000.0 + 1e-9


def test_solve_common_scale_respects_gross_cap():
    # Huge raw signal so the vol target alone would blow through the gross cap.
    x_raw = np.array([100.0, -100.0, 50.0])
    sigma_weekly = np.array([0.001, 0.001, 0.001])
    k = solve_common_scale(x_raw, sigma_weekly, target_vol=0.08, max_gross=2.0)
    scaled = k * x_raw
    assert np.sum(np.abs(scaled)) <= 2.0 + 1e-9


def _toy_scored_panel():
    dates = pd.bdate_range("2020-01-06", periods=4, freq="7D")  # 4 weekly exec dates
    rows = []
    for asset in ["A", "B"]:
        for d in dates:
            rows.append({
                "asset": asset, "t_signal": d, "execution_date": d + pd.Timedelta(days=3),
                "ghat": 0.01 if asset == "A" else -0.01, "sigma_60d": 0.01,
                "table_id": "5d", "next_week_return": 0.0,
            })
    return pd.DataFrame(rows)


def test_no_trade_band_holds_small_changes():
    panel = _toy_scored_panel()
    out = build_positions(panel, target_vol=0.08, max_gross=2.0, max_gross_per_name=0.10,
                           no_trade_band=10.0)  # huge band -> nothing should ever trade after week 1
    weights_by_week = out.pivot_table(index="execution_date", columns="asset", values="weight")
    # After the first week's initial entry, later weeks must equal the first week's weights.
    first = weights_by_week.iloc[0]
    for i in range(1, len(weights_by_week)):
        assert np.allclose(weights_by_week.iloc[i].to_numpy(), first.to_numpy())


def test_per_name_cap_is_respected():
    panel = _toy_scored_panel()
    out = build_positions(panel, target_vol=0.08, max_gross=2.0, max_gross_per_name=0.10,
                           no_trade_band=0.0)
    assert (out["weight"].abs() <= 0.10 + 1e-9).all()


def test_gross_cap_is_respected_with_many_names():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-06", periods=1, freq="7D")
    rows = []
    for i in range(50):
        rows.append({
            "asset": f"N{i}", "t_signal": dates[0], "execution_date": dates[0] + pd.Timedelta(days=3),
            "ghat": rng.normal(0, 0.02), "sigma_60d": 0.005, "table_id": "5d", "next_week_return": 0.0,
        })
    panel = pd.DataFrame(rows)
    out = build_positions(panel, target_vol=0.08, max_gross=2.0, max_gross_per_name=0.10,
                           no_trade_band=0.0)
    assert out["weight"].abs().sum() <= 2.0 + 1e-6
