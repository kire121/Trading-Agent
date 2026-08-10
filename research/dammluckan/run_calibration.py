"""
Dammluckan -- Step 0 of the pipeline: calibrate c (band constant, per grid n)
and theta_i (per-asset occupation threshold, per grid n / side / percentile),
IS-only (2004-2017), and freeze the result to output/calibration.pkl for
every other stage (backtest, grid search, OOS-1, OOS-2) to load and reuse
without ever re-touching OOS data during calibration.
"""
import os
import pickle
import time

from . import config
from . import data
from . import signal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "output", "calibration.pkl")


def run(n_draws_c=200, n_draws_theta=config.N_THETA_CALIB_DRAWS, force=False):
    if os.path.exists(OUT_PATH) and not force:
        with open(OUT_PATH, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    panel = data.load_primary_panel()
    result = {"c": {}, "theta_high": {}, "theta_low": {}, "anti_theta_high": {}, "anti_theta_low": {}}

    # Anti-twin uses the mirrored LOW percentile (100-p) of the exact same
    # null distribution -- computed in the same pass as the primary
    # percentiles below, no extra bootstrap draws needed.
    anti_pctls = [100 - p for p in config.THETA_PCTL_GRID]

    for n in config.N_GRID:
        c = signal.calibrate_band_constant(panel, n=n, n_draws=n_draws_c, seed=42)
        result["c"][n] = c
        print(f"[{time.time()-t0:7.1f}s] n={n}: calibrated c={c:.4f}")

        all_high = signal.calibrate_theta_multi(
            panel, n=n, c=c, pctls=list(config.THETA_PCTL_GRID) + anti_pctls, side="high",
            n_draws=n_draws_theta, seed=100,
        )
        result["theta_high"][n] = {p: all_high[p] for p in config.THETA_PCTL_GRID}
        result["anti_theta_high"][n] = {p: all_high[100 - p] for p in config.THETA_PCTL_GRID}
        print(f"[{time.time()-t0:7.1f}s] n={n}: theta_high calibrated "
              f"(median theta80={all_high[80].median():.4f}, anti theta20={all_high[20].median():.4f})")

        all_low = signal.calibrate_theta_multi(
            panel, n=n, c=c, pctls=list(config.THETA_PCTL_GRID) + anti_pctls, side="low",
            n_draws=n_draws_theta, seed=200,
        )
        result["theta_low"][n] = {p: all_low[p] for p in config.THETA_PCTL_GRID}
        result["anti_theta_low"][n] = {p: all_low[100 - p] for p in config.THETA_PCTL_GRID}
        print(f"[{time.time()-t0:7.1f}s] n={n}: theta_low calibrated "
              f"(median theta80={all_low[80].median():.4f}, anti theta20={all_low[20].median():.4f})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"[{time.time()-t0:7.1f}s] Saved calibration to {OUT_PATH}")
    return result


if __name__ == "__main__":
    run(force=True)
