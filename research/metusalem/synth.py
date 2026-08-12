"""Synthetic sign-panel generators for Steg 0c's machine gate (spec Sec.10).

Simulates the EPISODE STRUCTURE directly (renewal-process durations drawn
from a known Weibull/exponential law, alternating sign per episode) rather
than simulating price paths -- Steg 0c's own text is explicit that this is
a test of "hela kedjan episodextraktion->MLE", i.e. of the machinery given
a KNOWN sign panel, not of the base book's price-to-signal step (which has
its own, separate, already-tested provenance in basbok.py/smittotalet).
"""
import numpy as np
import pandas as pd


def _draw_weibull_durations(rng: np.random.Generator, k: float, lam: float, n: int) -> np.ndarray:
    u = rng.uniform(1e-12, 1.0 - 1e-12, n)
    d_cont = (-np.log(u)) ** (1.0 / k) / lam
    return np.maximum(1, np.round(d_cont)).astype(int)


def planted_weibull_panel(n_instr: int, n_weeks: int, k: float, lam_range: tuple,
                           seed: int) -> pd.DataFrame:
    """Stratified Weibull ground truth: common shape `k`, one lambda_i per
    instrument drawn uniformly from `lam_range` (per-week hazard scale)."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start="2000-01-07", periods=n_weeks, freq="W-FRI")
    cols = {}
    lam_lo, lam_hi = lam_range
    for i in range(n_instr):
        lam_i = rng.uniform(lam_lo, lam_hi)
        sign = 1
        weeks_filled = 0
        vals = np.empty(n_weeks)
        while weeks_filled < n_weeks:
            # generate a batch of durations at once, cheap and avoids a
            # python-level while-loop-of-one draws
            durations = _draw_weibull_durations(rng, k, lam_i, 64)
            for d in durations:
                d = min(int(d), n_weeks - weeks_filled)
                vals[weeks_filled: weeks_filled + d] = sign
                weeks_filled += d
                sign = -sign
                if weeks_filled >= n_weeks:
                    break
        cols[f"I{i:03d}"] = vals
    return pd.DataFrame(cols, index=idx)


def exponential_mixture_panel(n_instr: int, n_weeks: int, lam_range: tuple, seed: int) -> pd.DataFrame:
    """Null scenario for K1b: k=1 (truly memoryless) per instrument, but
    lambda_i heterogeneous across instruments -- the classic frailty-
    illusion setup (poolable decreasing hazard is a pure mixture artefact,
    per spec Sec.3's "kritisk falla")."""
    return planted_weibull_panel(n_instr, n_weeks, k=1.0, lam_range=lam_range, seed=seed)
