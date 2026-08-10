import numpy as np
import pandas as pd

from research.omori import battery


def _synthetic_battery_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    p_hat = rng.uniform(0.3, 1.5, size=n)
    c_hat = rng.choice([0, 1, 2], size=n).astype(float)
    n_pos = rng.integers(4, 20, size=n)
    # realized half-life constructed to be genuinely informed by p_hat (via
    # the model's own tau_exit-at-theta=0.5 formula), plus noise, so the
    # redundancy screen's incremental-R^2 test has something real to detect.
    half_life = np.clip((1 + c_hat) * 0.5 ** (-1.0 / p_hat) - c_hat + rng.normal(0, 1, n), 1, 20)
    return pd.DataFrame({
        "ticker": rng.choice(["A", "B", "C"], size=n),
        "t0_idx": np.arange(n),
        "p_hat": p_hat, "c_hat": c_hat, "n_pos": n_pos, "half_life": half_life,
        "realized_vol": rng.uniform(0.005, 0.03, n), "abs_r_autocorr": rng.uniform(-0.2, 0.4, n),
        "skew": rng.normal(0, 1, n), "mean_corr": rng.uniform(-0.3, 0.7, n),
        "absorption_ratio": rng.uniform(0.2, 0.7, n), "gap_share": rng.uniform(-1, 1, n),
        "volume_z": rng.uniform(4, 10, n), "abs_r0": rng.uniform(0.01, 0.1, n),
    })


class TestRedundancyScreen:
    def test_runs_and_returns_expected_keys(self):
        df = _synthetic_battery_df()
        out = battery.redundancy_screen(df)
        assert set(["r2_phat_vs_battery", "delta_r2_halflife", "killed"]) <= set(out.keys())

    def test_kills_when_p_hat_is_literally_a_battery_column(self):
        df = _synthetic_battery_df()
        df["realized_vol"] = df["p_hat"] * 0.01  # make p_hat perfectly linearly recoverable
        out = battery.redundancy_screen(df)
        assert out["r2_phat_vs_battery"] > 0.9
        assert out["kill_phat_redundant"]
        assert out["killed"]

    def test_absorption_ratio_between_0_and_1(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2015-01-01", periods=200)
        returns = pd.DataFrame(rng.normal(0, 0.01, size=(200, 10)), index=idx,
                                columns=[f"T{i}" for i in range(10)])
        ar = battery.absorption_ratio(returns, 150)
        assert 0.0 <= ar <= 1.0

    def test_absorption_ratio_high_for_single_factor_returns(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2015-01-01", periods=200)
        factor = rng.normal(0, 0.01, size=200)
        returns = pd.DataFrame({f"T{i}": factor + rng.normal(0, 1e-5, 200) for i in range(10)}, index=idx)
        ar = battery.absorption_ratio(returns, 150)
        assert ar > 0.9  # near-perfectly collinear -> top PC explains almost everything
