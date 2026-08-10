"""Omori aftershock-relaxation fit, empirical-Bayes shrinkage, and the
traded signal Z.

Excess volume e(tau), tau = trading days since the event day t0, is defined
as the RELATIVE excess of dollar volume over the same rolling-120d causal
median baseline used for event detection, evaluated forward in time:

    e(tau) = (dollar_volume(t0+tau) - baseline(t0+tau)) / baseline(t0+tau)

which is unitless and can be negative (no excess that day). The Omori model
e(tau) = K*(tau+c)^-p is fit in log-log space (log e(s) ~ log(s+c)) by
Huber-robust regression, using only days s where e(s) > 0 (log requires a
positive argument) -- this is exactly what "n_e" (positive-excess-day
count) counts, and it is what the >=4-day / full-shrinkage rule gates on.
"""
import numpy as np
import statsmodels.api as sm

from research.omori import config


class FitResult:
    __slots__ = ("p_hat", "k_hat", "c_hat", "n_pos", "identified")

    def __init__(self, p_hat, k_hat, c_hat, n_pos, identified):
        self.p_hat = p_hat
        self.k_hat = k_hat
        self.c_hat = c_hat
        self.n_pos = n_pos
        self.identified = identified


def _huber_fit_one_c(s_vals, e_vals, c):
    """Fit log(e) ~ log(s+c) with Huber-robust regression for one candidate
    c. Returns (slope, intercept, robust_pseudo_r2) or None if degenerate
    (fewer than 2 distinct x values, or singular design)."""
    x = np.log(s_vals + c)
    y = np.log(e_vals)
    if len(np.unique(x)) < 2:
        return None
    X = sm.add_constant(x)
    try:
        fit = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    except Exception:
        return None
    intercept, slope = fit.params
    w = fit.weights
    resid = fit.resid
    wsse = np.sum(w * resid ** 2)
    ybar = np.average(y, weights=w)
    wsst = np.sum(w * (y - ybar) ** 2)
    pseudo_r2 = 1.0 - wsse / wsst if wsst > 1e-12 else 0.0
    return slope, intercept, pseudo_r2


def fit_omori(tau_current, e_path, c_grid=config.C_PROFILE_GRID,
              min_pos_days=config.MIN_POSITIVE_EXCESS_DAYS):
    """Causal Omori fit as of `tau_current` (trading days since event),
    given e_path (1-indexed array-like, e_path[s-1] = e(s) for s=1..tau_current,
    NaN where unavailable). c is profiled over `c_grid`, selecting the
    candidate with the highest robust pseudo-R^2. Returns a FitResult;
    `identified=False` (full shrinkage) when there are fewer than
    `min_pos_days` positive-excess days, or no candidate c produces a valid
    decaying (p>0) fit -- the per-event identifiability gate ("Vridmoment"
    -style degeneracy guard: an unidentifiable slope is surfaced as
    unidentified, never silently coerced into a number)."""
    e_path = np.asarray(e_path[:tau_current], dtype=float)
    s_vals = np.arange(1, tau_current + 1, dtype=float)
    pos_mask = np.isfinite(e_path) & (e_path > 0)
    n_pos = int(pos_mask.sum())
    if n_pos < min_pos_days:
        return FitResult(p_hat=np.nan, k_hat=np.nan, c_hat=np.nan, n_pos=n_pos, identified=False)

    s_pos, e_pos = s_vals[pos_mask], e_path[pos_mask]
    best = None
    for c in c_grid:
        res = _huber_fit_one_c(s_pos, e_pos, c)
        if res is None:
            continue
        slope, intercept, r2 = res
        p_hat = -slope
        if not np.isfinite(p_hat) or p_hat <= 0:
            continue  # not a decaying law under this c -> unidentified for this c
        if best is None or r2 > best[3]:
            best = (c, p_hat, intercept, r2)

    if best is None:
        return FitResult(p_hat=np.nan, k_hat=np.nan, c_hat=np.nan, n_pos=n_pos, identified=False)

    c_hat, p_hat, log_k_hat, _ = best
    return FitResult(p_hat=p_hat, k_hat=np.exp(log_k_hat), c_hat=float(c_hat), n_pos=n_pos, identified=True)


def shrink(fit: FitResult, prior_p, kappa=config.KAPPA_DEFAULT):
    """Empirical-Bayes posterior exponent:
        p_tilde = (n_e * p_hat + kappa * prior_p) / (n_e + kappa)
    Full shrinkage (p_tilde = prior_p) when the fit is unidentified, i.e.
    n_e is treated as 0."""
    if not fit.identified:
        return float(prior_p)
    n_e = fit.n_pos
    return (n_e * fit.p_hat + kappa * prior_p) / (n_e + kappa)


def g_of_p(p_tilde, p_star):
    """The traded signal's decreasing function of p_tilde: full-size tilt
    (1.0) when decay is at or slower than the reference exponent p_star,
    shrinking as decay speeds up past it. This IS the sizing tilt w's
    magnitude -- see sizing.py -- so that "steg-1-IC korrs pa Z, inte pa
    osignerad komponent" (step-1 IC always runs on the actual signed traded
    quantity, never on unsigned p_hat/p_tilde alone)."""
    if not np.isfinite(p_tilde) or p_tilde <= 0 or p_star is None:
        return 0.0
    return min(1.0, p_star / p_tilde)


def traded_signal(direction, p_tilde, p_star):
    """Z = sign(r0) * g(p_tilde) -- the signed, handled quantity. Identical
    in magnitude to the sizing tilt w (see sizing.py); IC tests below and in
    battery.py must always be run on this signed Z, never on bare p_tilde."""
    return direction * g_of_p(p_tilde, p_star)


def tau_exit(fit_c_hat, p_tilde, theta=config.THETA_DEFAULT,
             floor=config.TAU_EXIT_FLOOR, cap=config.TAU_EXIT_CAP_DEFAULT):
    """tau_exit = (1+c_hat) * theta^(-1/p_tilde) - c_hat, derived by taking
    the model's own implied excess level at tau=1 as "initial level" and
    solving K(tau+c)^-p = theta * K(1+c)^-p for tau. Clipped to [floor, cap]."""
    if not np.isfinite(p_tilde) or p_tilde <= 0 or not np.isfinite(fit_c_hat):
        return float(cap)
    # Work in log-space and short-circuit before exponentiating: very small
    # p_tilde (slow decay) implies an astronomically large raw tau_exit,
    # which clips to `cap` regardless -- checking in log-space avoids an
    # OverflowError for p_tilde values seen in practice (fits as small as
    # ~1e-5 occur when the excess-volume path is nearly flat).
    log_raw_upper_bound = np.log(1.0 + fit_c_hat) + (-1.0 / p_tilde) * np.log(theta)
    if log_raw_upper_bound > np.log(cap + abs(fit_c_hat) + 1.0):
        return float(cap)
    raw = (1.0 + fit_c_hat) * theta ** (-1.0 / p_tilde) - fit_c_hat
    return float(np.clip(raw, floor, cap))
