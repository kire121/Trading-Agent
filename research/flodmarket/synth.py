"""Seeded synthetic OHLC generator (spec SS12.4), used for two purposes,
both BEFORE any real data is fetched (spec SS9 Steg0b ordering):

  1. Uppnaelighetsband (achievability bands, K0b.1/K0b.2): ONE full-scale
     run (40 tickers x 5000 days, 390-step intraday GBM) with theta=0 (no
     planted effect) -- see bands.py.
  2. A/B-separation (estimator power/false-positive-rate demonstration):
     200 repeated smaller-scale runs -- see ab_separation.py. Full-scale
     (40 x 5000 x 390) x 200 reps is computationally intractable within a
     single research session (~15bn random draws); the reduced scale is a
     DECLARED, disclosed choice (AVVIKELSER.md) affecting ONLY this
     internal null-methodology self-check, never the real Steg 0a-5
     pipeline, which always runs on the full real 40-ticker/22-year panel.

Design (spec SS12.4, locked): per ticker, sigma_year cycles through
{5,10,20,40}%. Each day: gap O_t = C_{t-1}*exp(eps), eps ~ N(0, 0.3*sigma_dag).
Intraday: 390-step GBM, innovation mix 70% normal + 30% Student-t(4)
(rescaled to the same per-step variance as the normal leg, since a
standard t(4) has variance df/(df-2)=2). H/L are the running max/min of the
intraday path (including the open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA_YEAR_LEVELS = (0.05, 0.10, 0.20, 0.40)
INTRADAY_STEPS = 390
GAP_VOL_FRACTION = 0.3
T_DF = 4
T_MIX_FRACTION = 0.3
TRADING_DAYS_YEAR = 252


def _t4_scale_for_variance(target_var: np.ndarray) -> np.ndarray:
    """Scale factor so that scale * StudentT(4) has variance == target_var
    (Var[T4] = df/(df-2) = 2)."""
    return np.sqrt(target_var / (T_DF / (T_DF - 2)))


def simulate_panel(n_assets: int, n_days: int, seed: int, *, intraday_steps: int = INTRADAY_STEPS,
                    start_price: float = 100.0) -> dict:
    """Simulates `n_assets` independent tickers over `n_days` trading days.
    sigma_year cycles through SIGMA_YEAR_LEVELS (assets split as evenly as
    possible across the four levels). Returns
    {"O","H","L","C": DataFrame[date x asset], "sigma_year": {asset: float}}.
    """
    rng = np.random.default_rng(seed)
    assets = [f"SYN{i:02d}" for i in range(n_assets)]
    sigma_year_by_asset = {a: SIGMA_YEAR_LEVELS[i % len(SIGMA_YEAR_LEVELS)] for i, a in enumerate(assets)}

    dates = pd.bdate_range("2000-01-03", periods=n_days)
    O = np.empty((n_days, n_assets))
    H = np.empty((n_days, n_assets))
    L = np.empty((n_days, n_assets))
    C = np.empty((n_days, n_assets))

    for j, a in enumerate(assets):
        sigma_year = sigma_year_by_asset[a]
        sigma_dag = sigma_year / np.sqrt(TRADING_DAYS_YEAR)
        sigma_min = sigma_dag / np.sqrt(intraday_steps)

        gap_eps = rng.normal(0.0, GAP_VOL_FRACTION * sigma_dag, n_days)

        is_t = rng.random((n_days, intraday_steps)) < T_MIX_FRACTION
        normal_incr = rng.normal(0.0, sigma_min, (n_days, intraday_steps))
        t_scale = _t4_scale_for_variance(np.array(sigma_min ** 2))
        t_incr = rng.standard_t(T_DF, (n_days, intraday_steps)) * t_scale
        incr = np.where(is_t, t_incr, normal_incr)
        log_path = np.cumsum(incr, axis=1)  # relative to log(O_t), shape (n_days, steps)

        prev_close = start_price
        for t in range(n_days):
            o = prev_close * np.exp(gap_eps[t])
            path = o * np.exp(log_path[t])
            full_path = np.concatenate([[o], path])
            c = path[-1]
            O[t, j] = o
            H[t, j] = full_path.max()
            L[t, j] = full_path.min()
            C[t, j] = c
            prev_close = c

    frames = {
        "O": pd.DataFrame(O, index=dates, columns=assets),
        "H": pd.DataFrame(H, index=dates, columns=assets),
        "L": pd.DataFrame(L, index=dates, columns=assets),
        "C": pd.DataFrame(C, index=dates, columns=assets),
    }
    return {**frames, "sigma_year": sigma_year_by_asset, "assets": assets, "dates": dates}


def panel_to_multiindex(panel: dict) -> dict:
    """Converts the wide {O,H,L,C: DataFrame[date x asset]} dict into
    MultiIndex(ticker,date)-indexed Series, matching intrabar.load_ohlc's
    output shape (minus adjC, which is irrelevant for synthetic data: no
    corporate actions)."""
    out = {}
    for field in ("O", "H", "L", "C"):
        wide = panel[field]
        stacked = wide.stack()
        stacked.index.names = ["date", "ticker"]
        stacked = stacked.reorder_levels(["ticker", "date"]).sort_index()
        out[field] = stacked
    return out


def plant_effect(panel: dict, seed: int, theta: float, *, mixed_sign: bool = False,
                  n_episodes: int = 6, ar1_phi: float = 0.9) -> dict:
    """Adds a latent AR(1) factor z_t (phi=ar1_phi) that biases (a) the
    day's shadow asymmetry (via a small O/C nudge that shifts realized s in
    the direction of z_t), per ticker independently (shared AR(1) SHAPE via
    a per-asset-seeded draw, not a shared realization across assets --
    avoids injecting a spurious common-factor artifact into the breadth
    checks this generator also feeds).

    The matching "5d-forward-drift" (spec SS12.4: the same latent factor
    "driver bade s-bias och 5d-framatdrift") is DELIBERATELY NOT baked into
    the OHLC price path itself. An earlier version tried to inject it via a
    rolling/accumulating log-price adjustment and produced a spurious,
    WRONG-SIGNED relationship: any additive perturbation applied to
    individual days' price LEVELS necessarily shows up in a two-point
    return as a DIFFERENCE of the perturbations at the window's two
    endpoints (not as a clean function of the window's own theta*z_t), and
    an accumulating (cumsum) version creates unbounded per-asset drift over
    a 1200-5000 day panel that swamps the base GBM noise. Since the sole
    purpose of this generator's effect-planting is to validate the
    ESTIMATOR/null machinery (rolling t-stat + block-permutation null) on a
    KNOWN ground truth, not to produce a self-consistent tradeable price
    series, the forward-return target is instead exposed directly via the
    returned "z" factor: callers (ab_separation.py) construct the biased
    evaluation target as raw_forward_5d_return + theta*z_t, a transparent,
    bug-resistant construction with the intended predictive relationship
    by definition. See AVVIKELSER.md.

    mixed_sign=True: theta's SIGN alternates across n_episodes contiguous,
    roughly-equal-length segments of the sample (DECLARED resolution of
    spec's "mixed-sign-plantering ingar (Runraden-mallkrav 3)" -- see
    AVVIKELSER.md: no such numbered template exists in Runraden's own repo
    history, confirmed via repo-wide grep; research/smittotalet/README.md
    already documents this same citation-integrity gap for a different
    "Runraden mallkrav" reference and resolves it the same way, via a
    disclosed de-facto interpretation rather than a hard stop).

    Returns a NEW panel dict (does not mutate the input) with O,H,L,C's
    CLOSE nudged (s-bias only) and an extra "z" DataFrame (the latent
    factor, per asset, sign-flipped per episode when mixed_sign=True).
    """
    rng = np.random.default_rng(seed + 987_654)
    out = {k: v.copy() if hasattr(v, "copy") else v for k, v in panel.items()}
    n_days = len(panel["dates"])
    assets = panel["assets"]

    episode_bounds = np.linspace(0, n_days, n_episodes + 1).astype(int)
    episode_sign = np.empty(n_days)
    for e in range(n_episodes):
        sign = 1.0 if (not mixed_sign or e % 2 == 0) else -1.0
        episode_sign[episode_bounds[e]:episode_bounds[e + 1]] = sign

    z_by_asset = {}
    for a in assets:
        innov = rng.normal(0, 1, n_days)
        z = np.empty(n_days)
        z[0] = innov[0]
        for t in range(1, n_days):
            z[t] = ar1_phi * z[t - 1] + np.sqrt(1 - ar1_phi ** 2) * innov[t]
        z = z / (z.std() if z.std() > 0 else 1.0)  # unit variance
        z_by_asset[a] = z * episode_sign

        theta_t = theta * z_by_asset[a]

        H = out["H"][a].to_numpy()
        L = out["L"][a].to_numpy()
        C = out["C"][a].to_numpy().copy()
        R = H - L
        R_safe = np.where(R > 0, R, np.nan)

        # bias today's s by nudging C toward/away from the day's midpoint
        # within [L,H], magnitude theta_t * R (theta in s-units, s in [-1,1]).
        shift = np.clip(theta_t, -0.9, 0.9) * R_safe
        C_new = np.clip(C + shift, L, H)
        out["C"][a] = np.where(np.isnan(C_new), C, C_new)

    out["z"] = pd.DataFrame(z_by_asset, index=panel["dates"])
    out["theta"] = theta
    out["mixed_sign"] = mixed_sign
    return out
