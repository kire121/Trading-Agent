"""IS-only calibration of the two hyperparameters the brief doesn't pin:
kappa (EB shrinkage strength) and p_star (the sizing reference exponent).
Both run once, frozen, before OOS -- see config.py's KAPPA_GRID / P_STAR
docstrings for why these specific procedures were chosen.
"""
import json
import os

import numpy as np

from research.omori import config

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")


def p_realized_from_halflife(half_life, c_hat):
    """Invert tau_exit's own closed form at theta=0.5 (half-life is exactly
    that: the day the model-implied excess first halves) to get the
    exponent a model WOULD need to have produced the realized half-life:
        0.5 = ((half_life + c) / (1 + c))^-p  =>  p = -ln(0.5) / ln((half_life+c)/(1+c))
    Used only as a cross-validation TARGET for calibrating kappa -- never
    fed back into the traded signal itself."""
    half_life = np.asarray(half_life, dtype=float)
    c_hat = np.asarray(c_hat, dtype=float)
    ratio = (half_life + c_hat) / (1.0 + c_hat)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = -np.log(0.5) / np.log(ratio)
    p[~np.isfinite(p) | (p <= 0)] = np.nan
    return p


def calibrate_kappa(battery_df, kappa_grid=config.KAPPA_GRID, save=True):
    """Leave-one-instrument-out: for every event, the instrument prior used
    is the population median EXCLUDING that instrument's own events (i.e.
    treating it as if it had no IS history, exactly the situation every OOS
    ticker will actually be in). Selects the kappa minimizing MSE between
    p_tilde (built from that LOO prior) and the half-life-implied realized
    exponent."""
    df = battery_df.dropna(subset=["half_life", "c_hat", "p_hat", "n_pos"]).copy()
    df["p_realized"] = p_realized_from_halflife(df["half_life"].values, df["c_hat"].values)
    df = df.dropna(subset=["p_realized"])

    global_all = df["p_hat"].values

    def loo_prior(ticker):
        others = df.loc[df["ticker"] != ticker, "p_hat"].values
        return float(np.median(others)) if len(others) else float(np.median(global_all))

    loo_priors = {t: loo_prior(t) for t in df["ticker"].unique()}
    n_pos = df["n_pos"].values if "n_pos" in df.columns else np.full(len(df), config.MIN_POSITIVE_EXCESS_DAYS)
    p_hat = df["p_hat"].values
    prior_arr = df["ticker"].map(loo_priors).values
    p_realized = df["p_realized"].values

    best_kappa, best_mse = None, np.inf
    mse_by_kappa = {}
    for kappa in kappa_grid:
        p_tilde_loo = (n_pos * p_hat + kappa * prior_arr) / (n_pos + kappa)
        mse = float(np.mean((p_tilde_loo - p_realized) ** 2))
        mse_by_kappa[kappa] = mse
        if mse < best_mse:
            best_mse, best_kappa = mse, kappa

    out = {"kappa": best_kappa, "mse_by_kappa": mse_by_kappa, "n_events": int(len(df))}
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "kappa.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out


def calibrate_p_star(closed_events_df, save=True):
    """p_star = frozen IS-panel median of entry-day (tau=1) p_tilde across
    all traded IS events. Entry-day p_tilde is mechanically always the
    (frozen) instrument prior -- see backtest.py's module docstring -- so
    this is equivalent to the event-count-weighted median instrument prior
    among traded instruments, but is computed directly from realized trades
    to also reflect the gross-cap admission process."""
    p_star = float(closed_events_df["p_tilde_entry"].median())
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "p_star.json"), "w") as f:
            json.dump({"p_star": p_star, "n_events": int(len(closed_events_df))}, f, indent=2)
    return p_star


def load_kappa():
    with open(os.path.join(OUTPUT_DIR, "kappa.json")) as f:
        return json.load(f)["kappa"]


def load_p_star():
    with open(os.path.join(OUTPUT_DIR, "p_star.json")) as f:
        return json.load(f)["p_star"]
