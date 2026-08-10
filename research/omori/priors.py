"""Instrument-prior (p_bar_i) and global-prior (p_bar_global) estimation --
run ONCE on the IS panel, frozen, then reused unchanged for both the IS
backtest and the OOS run (see config.py's note on "instrument prior p_bar_i
skattad enbart pa IS-panelen": since none of the OOS tickers have any IS
history, their p_bar_i necessarily collapses to p_bar_global by
construction, which is exactly the intended behavior, not a special case).

This is a one-time, non-causal-within-IS calibration step (like Dammluckan's
band-constant calibration) -- IS is spent freely on design; only the
downstream Omori CURVE FIT (signal.fit_omori, per event, per day) has to be
causal, not this population-level prior.
"""
import json
import os

import numpy as np

from research.omori import config, events, signal

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")


def terminal_fits(panel, z_threshold=config.Z_VOLUME_THRESHOLD,
                   sigma_mult=config.R0_SIGMA_THRESHOLD,
                   cap=config.TAU_EXIT_CAP_DEFAULT):
    """One terminal (tau = min(cap, days available)) Omori fit per raw
    candidate event (ignoring the open-instrument/clustering trading rules
    -- this is a population-level descriptive pass, not a trading
    simulation). Returns a list of dicts with ticker/date/p_hat/identified."""
    ef = events.EventFields(panel)
    cands = ef.candidates_long(z_threshold, sigma_mult)
    n = len(panel.index)
    out = []
    for _, row in cands.iterrows():
        ticker, t0_idx = row["ticker"], int(row["date_idx"])
        series = ef.excess_volume_series(ticker)
        max_tau = min(cap, n - 1 - t0_idx)
        if max_tau < config.MIN_POSITIVE_EXCESS_DAYS:
            out.append({"ticker": ticker, "date": row["date"], "t0_idx": t0_idx, "p_hat": np.nan,
                        "c_hat": np.nan, "n_pos": 0, "e_path": None, "identified": False})
            continue
        e_path = series.values[t0_idx + 1: t0_idx + 1 + max_tau]
        fit = signal.fit_omori(max_tau, e_path)
        out.append({
            "ticker": ticker, "date": row["date"], "t0_idx": t0_idx,
            "p_hat": fit.p_hat if fit.identified else np.nan,
            "c_hat": fit.c_hat if fit.identified else np.nan,
            "n_pos": fit.n_pos, "e_path": e_path if fit.identified else None,
            "identified": fit.identified,
        })
    return out


def calibrate_priors(panel, save=True):
    fits = terminal_fits(panel)
    return calibrate_priors_from_fits(panel, fits, save=save)


def calibrate_priors_from_fits(panel, fits, save=True):
    identified = [f for f in fits if f["identified"]]
    if not identified:
        raise RuntimeError("No identified IS events -- cannot calibrate priors.")
    global_p = float(np.median([f["p_hat"] for f in identified]))

    per_instrument = {}
    for ticker in panel.tickers:
        ps = [f["p_hat"] for f in identified if f["ticker"] == ticker]
        per_instrument[ticker] = float(np.mean(ps)) if ps else global_p

    out = {
        "global": global_p,
        "per_instrument": per_instrument,
        "n_identified": len(identified),
        "n_candidates": len(fits),
    }
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "priors.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out


def load_priors():
    with open(os.path.join(OUTPUT_DIR, "priors.json")) as f:
        return json.load(f)


def prior_for(ticker, priors):
    return priors["per_instrument"].get(ticker, priors["global"])
