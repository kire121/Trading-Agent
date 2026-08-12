"""Metusalem's statistical core: episode extraction, stratified Weibull/exponential
hazard MLE, cluster bootstrap, pooled Nelson-Aalen diagnostic, the oracle's residual-
life indata, and the tilt-weight construction.

Full API per the locked pre-registration Sec.12 -- "fullstandig -- inga designbeslut
kvar". No design choice below is a NEW statistical decision: every formula, bound and
tie-break is transcribed directly from Sec.12 or is a mechanical consequence of it
(see the docstring of each function for where it maps to the spec text). Where the
spec's function signature left a level-of-scaling ambiguity (tilt_weights: raw vs.
already-vol-scaled w_bas), the choice made is documented there and in AVVIKELSER.md
-- it is a wiring/reuse-fidelity decision (which representation of w_bas avoids
double-applying the gross cap), not a new statistical design decision.

No existing survival/hazard module was found anywhere in the repo (grep across lib/,
main, and all branches for "weibull", "hazard", "survival", "censor", and the exact
function names below all came back empty except this repo's own history of writing
this file) -- see AVVIKELSER.md sec.1 for the confirmed absence. Built fresh per
Sec.12, as the spec's own Sec.1 kyrkogardskontroll anticipated ("Ingen dod familj
ateruppfinns" -- survival analysis has no prior grave in this repo).

Dependencies: numpy, pandas, scipy.optimize/stats -- no lifelines (per Sec.6,
"Overlevnadsformlerna specas fullt i S12 => inget beroende av lifelines kravs").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize as sopt


# ---------------------------------------------------------------------------
# 1. Episode extraction (Sec.12.1, Sec.5 "Alder")
# ---------------------------------------------------------------------------

def extract_episodes(s_panel: pd.DataFrame) -> pd.DataFrame:
    """Extracts trend episodes per instrument from a +-1/NaN direction panel.

    A flip between week t-1 and t (both non-NaN, different sign) starts a new
    episode at t. `event`=1 if the episode ends via an observed flip (the very
    next non-NaN week has a different sign), 0 if it is right-censored (panel
    end of the given `s_panel`, or a NaN gap immediately follows). The FIRST
    episode of an instrument, and any episode that resumes immediately after a
    NaN gap, is `left_censored`=1 (true start unknown) -- excluded from
    estimation (weibull/exp MLE) downstream, but still recorded here with its
    OBSERVED partial duration `d` and its own `event` flag.

    `s_panel` is expected to already be sliced to whatever estimation window
    the caller wants (e.g. IS-A up to 2017-12-31) -- panel-end censoring is
    always just "last row of the given frame", so IS-A-boundary censoring and
    true-panel-end censoring are the same code path by construction.

    Returns a DataFrame with columns: instrument, start_v, slut_v, d, event,
    left_censored -- one row per episode, `start_v`/`slut_v` are the week
    labels (index values) of `s_panel`.
    """
    rows = []
    weeks = s_panel.index
    for col in s_panel.columns:
        s = s_panel[col]
        start_idx = None
        left_censored = None  # left-censored status of the CURRENTLY open episode
        prev_val = None  # last non-NaN value seen (None until the first one)
        after_gap = False  # True if the episode about to open follows a NaN gap

        def _close(end_i, event_flag):
            rows.append({
                "instrument": col,
                "start_v": weeks[start_idx],
                "slut_v": weeks[end_i],
                "d": end_i - start_idx + 1,
                "event": event_flag,
                "left_censored": left_censored,
            })

        for i, val in enumerate(s.to_numpy()):
            is_nan = val != val  # NaN check without pandas overhead
            if is_nan:
                if start_idx is not None:
                    _close(i - 1, 0)  # NaN-lucka -> right-censored
                    start_idx = None
                after_gap = True
                prev_val = None
                continue

            if start_idx is None:
                # opening a new episode at i
                start_idx = i
                left_censored = 1 if (prev_val is None) else 1 if after_gap else 0
                after_gap = False
            elif val != prev_val:
                _close(i - 1, 1)  # flip -> event
                start_idx = i
                left_censored = 0  # started via an observed flip: known start
            prev_val = val

        if start_idx is not None:
            _close(len(s) - 1, 0)  # panel end -> right-censored

    return pd.DataFrame(rows, columns=["instrument", "start_v", "slut_v", "d",
                                        "event", "left_censored"])


def age_panel(s_panel: pd.DataFrame) -> pd.DataFrame:
    """Per (week, instrument) observed episode age: a=1 the first week of an
    OBSERVED episode, +1/week thereafter; NaN while left-censored (unknown
    true age) or while s itself is NaN (Sec.12.2, Sec.5 "Alder")."""
    episodes = extract_episodes(s_panel)
    out = pd.DataFrame(np.nan, index=s_panel.index, columns=s_panel.columns)
    week_pos = {w: i for i, w in enumerate(s_panel.index)}
    for row in episodes.itertuples(index=False):
        if row.left_censored:
            continue
        s_i, e_i = week_pos[row.start_v], week_pos[row.slut_v]
        out.iloc[s_i:e_i + 1, out.columns.get_loc(row.instrument)] = np.arange(1, e_i - s_i + 2)
    return out


# ---------------------------------------------------------------------------
# 3/4. Stratified Weibull / exponential MLE (Sec.12.3-12.4)
# ---------------------------------------------------------------------------

def _profile_lambda(k: float, d_tilde: np.ndarray, event: np.ndarray, instr_codes: np.ndarray,
                     n_instr: int) -> np.ndarray:
    """lambda_hat_i(k) = (m_i / sum_j d_tilde_ij^k)^(1/k), m_i = #events in i."""
    m_i = np.bincount(instr_codes[event == 1], minlength=n_instr).astype(float)
    denom = np.bincount(instr_codes, weights=d_tilde ** k, minlength=n_instr)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(m_i > 0, (m_i / np.where(denom > 0, denom, np.nan)) ** (1.0 / k), 0.0)
    return lam


def _stratified_loglik(k: float, d_tilde: np.ndarray, event: np.ndarray, instr_codes: np.ndarray,
                        n_instr: int) -> tuple:
    """Total log-likelihood at shape k, profiling out each stratum's lambda_i.
    Returns (loglik, lambda_by_stratum)."""
    lam = _profile_lambda(k, d_tilde, event, instr_codes, n_instr)
    lam_j = lam[instr_codes]
    base = -(lam_j * d_tilde) ** k
    event_mask = event == 1
    ll = np.sum(base)
    n_events = int(event_mask.sum())
    if n_events:
        lam_ev = lam_j[event_mask]
        ll += n_events * np.log(k)
        with np.errstate(divide="ignore"):
            log_lam = np.where(lam_ev > 0, np.log(lam_ev), -np.inf)
        ll += k * np.sum(log_lam)
        ll += (k - 1.0) * np.sum(np.log(d_tilde[event_mask]))
    return float(ll), lam


def _prep(d, event, instr):
    d = np.asarray(d, dtype=float)
    event = np.asarray(event, dtype=int)
    instr = np.asarray(instr)
    uniq, instr_codes = np.unique(instr, return_inverse=True)
    d_tilde = d - 0.5
    return d_tilde, event, instr_codes, uniq


def weibull_stratified_mle(d, event, instr, k_bounds: tuple = (0.2, 3.0)) -> tuple:
    """Stratified Weibull MLE: common shape k across all instruments, one
    profile scale lambda_i per instrument (Sec.12.3). Continuity correction
    d_tilde = d - 0.5. k optimized by bounded 1-D minimization (scipy,
    Brent-derived bounded search) over k_bounds.

    Returns (k_hat, {instrument: lambda_hat}, loglik, aic) with
    aic = 2*(n_instr + 1) - 2*loglik (n_instr scale params + 1 shared shape)."""
    d_tilde, event, instr_codes, uniq = _prep(d, event, instr)
    n_instr = len(uniq)

    def neg_ll(k):
        ll, _ = _stratified_loglik(k, d_tilde, event, instr_codes, n_instr)
        return -ll

    res = sopt.minimize_scalar(neg_ll, bounds=k_bounds, method="bounded",
                                options={"xatol": 1e-8})
    k_hat = float(res.x)
    ll, lam = _stratified_loglik(k_hat, d_tilde, event, instr_codes, n_instr)
    aic = 2.0 * (n_instr + 1) - 2.0 * ll
    lam_dict = {inst: float(lam_i) for inst, lam_i in zip(uniq, lam)}
    return k_hat, lam_dict, ll, aic


def exp_stratified_mle(d, event, instr) -> tuple:
    """Stratified exponential MLE (k fixed at 1, no shared shape parameter):
    the frailty-control null model (Sec.12.4). aic = 2*n_instr - 2*loglik.

    Returns ({instrument: lambda_hat}, loglik, aic)."""
    d_tilde, event, instr_codes, uniq = _prep(d, event, instr)
    n_instr = len(uniq)
    ll, lam = _stratified_loglik(1.0, d_tilde, event, instr_codes, n_instr)
    aic = 2.0 * n_instr - 2.0 * ll
    lam_dict = {inst: float(lam_i) for inst, lam_i in zip(uniq, lam)}
    return lam_dict, ll, aic


# ---------------------------------------------------------------------------
# 5. Cluster (instrument) bootstrap CI for k (Sec.12.5)
# ---------------------------------------------------------------------------

def cluster_bootstrap_k(episoder: pd.DataFrame, B: int, seed: int,
                         k_bounds: tuple = (0.2, 3.0)) -> dict:
    """Resamples INSTRUMENTS with replacement (B draws), refits
    weibull_stratified_mle on each resampled set of episodes (left-censored
    episodes excluded, matching the estimation population), and returns a
    percentile CI for k. Duplicate instrument draws are relabeled as distinct
    synthetic strata so the profile-lambda step treats each draw as its own
    stratum (preserves the intended within-instrument dependence structure of
    a cluster bootstrap: episodes move together with their instrument)."""
    est = episoder[episoder["left_censored"] == 0]
    instruments = est["instrument"].unique()
    rng = np.random.default_rng(seed)
    k_boot = np.empty(B)
    for b in range(B):
        draw = rng.choice(instruments, size=len(instruments), replace=True)
        d_parts, event_parts, instr_parts = [], [], []
        for synth_id, inst in enumerate(draw):
            sub = est[est["instrument"] == inst]
            d_parts.append(sub["d"].to_numpy())
            event_parts.append(sub["event"].to_numpy())
            instr_parts.append(np.full(len(sub), synth_id))
        d_b = np.concatenate(d_parts)
        event_b = np.concatenate(event_parts)
        instr_b = np.concatenate(instr_parts)
        k_hat, _, _, _ = weibull_stratified_mle(d_b, event_b, instr_b, k_bounds=k_bounds)
        k_boot[b] = k_hat
    return {
        "k_boot": k_boot,
        "ci_lower_95": float(np.percentile(k_boot, 2.5)),
        "ci_upper_95": float(np.percentile(k_boot, 97.5)),
    }


# ---------------------------------------------------------------------------
# 6. Pooled Nelson-Aalen (diagnostic only, Sec.12.6)
# ---------------------------------------------------------------------------

def nelson_aalen_pooled(d, event) -> tuple:
    """Pooled (not stratified) discrete-time Nelson-Aalen hazard: at each
    distinct observed integer age a, h_hat(a) = d_a / n_a where d_a = #events
    with duration exactly a, n_a = #episodes (event or censored) with
    duration >= a (at risk). Binomial SE = sqrt(d_a*(n_a-d_a)/n_a^3).
    Diagnostic only -- not decision-bearing (Sec.10 Steg 1). Caller controls
    which population is passed in (left-censored episodes are conventionally
    excluded, matching the MLE population, but this function itself performs
    no filtering).

    Returns (a, h_hat, n_a, se) as four aligned 1-D numpy arrays, sorted by a."""
    d = np.asarray(d, dtype=float)
    event = np.asarray(event, dtype=int)
    ages = np.arange(1, int(d.max()) + 1) if len(d) else np.array([], dtype=int)
    n_a = np.array([(d >= a).sum() for a in ages], dtype=float)
    d_a = np.array([((d == a) & (event == 1)).sum() for a in ages], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_hat = np.where(n_a > 0, d_a / n_a, np.nan)
        se = np.where(n_a > 0, np.sqrt(d_a * (n_a - d_a) / n_a ** 3), np.nan)
    return ages, h_hat, n_a, se


# ---------------------------------------------------------------------------
# 7. Residual life -- oracle indata (Sec.12.7)
# ---------------------------------------------------------------------------

def residual_life(episoder: pd.DataFrame, freq: str = "W-FRI") -> dict:
    """Weeks remaining until the episode's observed end, for every week
    actually within a COMPLETED episode (event==1 -- the end is observed,
    regardless of whether the start was left-censored; see AVVIKELSER.md for
    why left-censored-but-completed episodes are still included here, unlike
    in the MLE population). Censored episodes (event==0) contribute NaN for
    every one of their weeks (m=1 downstream, per Sec.5 "okand alder => m=1"
    applied analogously to unknown residual life).

    `episoder`'s start_v/slut_v are assumed to be week-end labels on a
    regular weekly cadence (`freq`, default Friday-anchored -- Sec.5's
    REBALANCE_WEEKDAY convention); the full week sequence within an episode
    is reconstructed via pd.date_range/period_range on that cadence, so no
    separate reference to the source panel is needed -- matching the spec's
    single-argument signature exactly.

    Returns a dict keyed by (instrument, week_label) -> residual weeks
    (float, NaN where undefined)."""
    out = {}
    for row in episoder.itertuples(index=False):
        if isinstance(row.start_v, pd.Period):
            weeks = pd.period_range(row.start_v, row.slut_v, freq=freq)
        else:
            weeks = pd.date_range(row.start_v, row.slut_v, freq=freq)
        if row.event == 1:
            n = len(weeks)
            for offset, w in enumerate(weeks):
                out[(row.instrument, w)] = float(n - 1 - offset)
        else:
            for w in weeks:
                out[(row.instrument, w)] = float("nan")
    return out


# ---------------------------------------------------------------------------
# 8. Tilt weights (Sec.12.8, Sec.5 "Tilt")
# ---------------------------------------------------------------------------

def tilt_weights(w_bas: pd.DataFrame, ages: pd.DataFrame, kappa: float) -> pd.DataFrame:
    """w_tilt = m (*) w_bas, per Sec.5:
      P_{i,t} = meanrank(a_{i,t}) / (n_known_t + 1) among instruments with
                known age at t (unknown age excluded from the ranking);
      m_{i,t} = 1 + kappa*(2*P_{i,t} - 1), unknown age => m=1;
      w_tilt  = m (*) w_bas  (elementwise, same index/columns).

    `w_bas` MUST be the RAW, pre-vol-target-scaling weekly base position
    (e.g. basbok.weekly_rebalanced_position's output) -- NOT the base book's
    own FINAL gross-capped/vol-targeted weights. Feeding the final weights
    in would apply the 200% gross cap twice (once inside the base book's own
    k-solve, once again in the caller's re-solve of w_tilt through "exakt
    samma volmalslosare"), silently distorting the tilt relative to a clean
    single-application construction. This is a reuse-fidelity wiring choice,
    not a new statistical decision -- see AVVIKELSER.md.

    Returns w_tilt, UNSCALED (the caller re-runs the same vol-target solver
    on this raw tilted signal to reach 10% ex-ante, per Sec.5)."""
    known = ages.notna()
    n_known = known.sum(axis=1)
    ranks = ages.rank(axis=1, method="average", na_option="keep")
    p = ranks.div(n_known + 1.0, axis=0)
    m = 1.0 + kappa * (2.0 * p - 1.0)
    m = m.where(known, 1.0)
    return m.mul(w_bas)
