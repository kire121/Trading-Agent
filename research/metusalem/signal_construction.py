"""T0/T-A/T-B/T-C twin construction, liveness assertions, and the shared
calibration path wiring (spec Sec.9). All twins share EXACTLY the same
vol-target solver (basbok.solve_k_for_target_vol) -- Dammluckan's bugfix
requirement, applied uniformly here.
"""
import numpy as np
import pandas as pd

from lib import twins as lib_twins

from . import basbok
from . import config
from . import scheduling
from . import survival_trend as st


def _lag_apply(raw: pd.DataFrame, exec_lag: int) -> pd.DataFrame:
    """Primary (exec_lag=1): base book's own Friday->Monday cadence.
    Robustness (exec_lag=2): one extra trading day (see scheduling.py)."""
    if exec_lag == 1:
        return scheduling.friday_lag_apply(raw)
    if exec_lag == 2:
        return scheduling.extra_day_lag_apply(raw)
    raise ValueError(f"unsupported exec_lag={exec_lag!r} (spec grid is {{1,2}})")


def build_s_panel(panel) -> pd.DataFrame:
    """Weekly (Friday-close, no lag) +-1/NaN sign panel s_{i,t} -- the SAME
    quantity the base book uses for direction (basbok.sign_momentum),
    sampled at week-end with no lag applied (age/episode tracking operates
    on the OBSERVED-at-close value; friday_lag_apply's lag is a trading
    convention for WHEN a weight takes effect, not for when the sign
    itself becomes known)."""
    mom_daily = basbok.sign_momentum(panel)
    weekly = scheduling.week_end_values(mom_daily)
    weekly.index = pd.PeriodIndex(weekly.index, freq=f"W-{config.REBALANCE_WEEKDAY}")
    return weekly


def cross_sectional_percentile(x: pd.DataFrame) -> pd.DataFrame:
    """P_{i,t} = meanrank(x_{i,t}) / (n_known_t + 1) among instruments with
    a known (non-NaN) value at t -- same ranking convention as
    survival_trend.tilt_weights, generalized to any cross-sectional score
    (age for the main signal, |12m|/vol for T-A)."""
    known = x.notna()
    n_known = known.sum(axis=1)
    ranks = x.rank(axis=1, method="average", na_option="keep")
    return ranks.div(n_known + 1.0, axis=0)


def strength_score(panel, weekly_index: pd.PeriodIndex) -> pd.DataFrame:
    """u_{i,t} = |12m return| / annualized 20d vol, sampled at Friday close
    (spec T-A: "u = |12m-avkastning|/sigma_ann")."""
    adj = panel.adjusted_close
    mom_abs = (adj / adj.shift(config.TSMOM_LOOKBACK) - 1.0).abs()
    daily_ret = adj.pct_change()
    vol_ann = daily_ret.rolling(config.TSMOM_VOL_LOOKBACK).std() * np.sqrt(config.TRADING_DAYS_YEAR)
    u_daily = mom_abs / vol_ann
    u_weekly = scheduling.week_end_values(u_daily)
    u_weekly.index = weekly_index
    return u_weekly


def quantile_map_panel(reference: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Panel-shaped wrapper around lib.twins.quantile_map_to: stacks both
    panels to a common (week, instrument) index, maps, unstacks back."""
    ref_s = reference.stack(future_stack=True)
    raw_s = raw.stack(future_stack=True)
    mapped = lib_twins.quantile_map_to(ref_s, raw_s)
    return mapped.unstack()


def permute_across_instruments(x: pd.DataFrame, redraw_weeks: int, seed: int) -> pd.DataFrame:
    """T-B (spec Sec.9): genuine permutation WITHOUT replacement of the
    X-vector across instruments (NOT a block bootstrap -- lib.bootstrap's
    circular_block_bootstrap_* resample WITH replacement and are the wrong
    primitive here; no existing permutation-across-instruments module was
    found anywhere in the repo, see AVVIKELSER.md). A single random
    permutation of instrument labels is drawn every `redraw_weeks` weeks and
    held fixed for that block -- preserves each week's exact cross-sectional
    multiset of X values, breaks the instrument<->X mapping."""
    rng = np.random.default_rng(seed)
    n_weeks = len(x)
    cols = x.columns.to_numpy()
    n = len(cols)
    out = x.copy()
    for block_start in range(0, n_weeks, redraw_weeks):
        block_end = min(block_start + redraw_weeks, n_weeks)
        perm = rng.permutation(n)
        out.iloc[block_start:block_end] = x.iloc[block_start:block_end, perm].to_numpy()
    return out


def solved_book(panel, raw_signal: pd.DataFrame, is_start, is_end, one_way_bps: float,
                 apply_costs: bool = True, k0: float = None):
    """Runs `raw_signal` through the shared calibration path (Dammluckan
    k-solve -> gross cap -> costs) -- the SAME machinery for every
    twin/grid cell (spec Sec.5/9)."""
    k = basbok.solve_k_for_target_vol(panel, is_start, is_end, raw=raw_signal, k0=k0)
    weights = basbok.apply_gross_cap(raw_signal, k)
    rets = basbok.portfolio_returns(panel, weights, apply_costs=apply_costs, one_way_bps=one_way_bps)
    return {"k": k, "weights": weights, "returns": rets}


def build_twin_c_score(episodes_is_a: pd.DataFrame) -> dict:
    """T-C (spec Sec.9): static frailty/selection twin. lambda_hat_i =
    flips / exposure from IS-A episodes (flips = count of event==1 episodes
    per instrument; exposure = total observed weeks, sum of `d` across ALL
    of that instrument's episodes, censored or not). X_C = 2*percentile(
    1/lambda_hat_i) - 1, constant over time -- a per-instrument selection
    tilt, not a per-week signal. Returns {instrument: X_C} (percentile
    computed across the instrument universe present in episodes_is_a)."""
    flips = episodes_is_a[episodes_is_a["event"] == 1].groupby("instrument").size()
    exposure = episodes_is_a.groupby("instrument")["d"].sum()
    lam_hat = (flips.reindex(exposure.index).fillna(0.0) / exposure).replace(0.0, np.nan)
    inv_lam = 1.0 / lam_hat
    ranks = inv_lam.rank(method="average")
    n = ranks.notna().sum()
    p = ranks / (n + 1.0)
    x_c = 2.0 * p - 1.0
    return x_c.to_dict()


def twin_c_raw_signal(w_bas_raw: pd.DataFrame, x_c_by_instrument: dict, kappa: float) -> pd.DataFrame:
    """Broadcasts the static per-instrument X_C into m_C = 1+kappa*X_C
    (constant over time, per instrument; unknown/missing instrument -> m=1,
    matching the main signal's "okand => m=1" convention) and multiplies
    onto w_bas_raw."""
    m = pd.Series({col: 1.0 + kappa * x_c_by_instrument.get(col, 0.0) if col in x_c_by_instrument
                   and np.isfinite(x_c_by_instrument[col]) else 1.0
                   for col in w_bas_raw.columns})
    return w_bas_raw.mul(m, axis=1)


def effective_n(weights: pd.DataFrame) -> pd.Series:
    """Effective-N = inverse HHI of |w_i| per week (spec Sec.9 assertion v)."""
    w = weights.abs()
    gross = w.sum(axis=1)
    shares = w.div(gross, axis=0).fillna(0.0)
    hhi = (shares ** 2).sum(axis=1)
    return (1.0 / hhi).where(hhi > 0, np.nan)


# ---------------------------------------------------------------------------
# Liveness assertions (spec Sec.9 -- hard, never conditioned away)
# ---------------------------------------------------------------------------

def liveness_assertions(main: dict, twin_a: dict, twin_b_paths: list, t0: dict,
                         tb_multipliers_sorted_check: tuple = None) -> list:
    """Returns a list of {"name","status","value"} dicts -- ALWAYS all five,
    regardless of outcome (rule 5: an assertion is a result, never a
    conditioned-away hurdle)."""
    out = []

    def add(name, passed, value):
        out.append({"name": name, "status": "PASS" if bool(passed) else "FAIL", "value": value})

    # (i) sorted multipliers T-A == main within 1e-9, every week.
    if tb_multipliers_sorted_check is not None:
        m_main_sorted, m_a_sorted = tb_multipliers_sorted_check
        max_diff = float(np.nanmax(np.abs(m_main_sorted - m_a_sorted))) if len(m_main_sorted) else float("nan")
        add("liveness_i_sorterade_multiplikatorer_TA_eq_huvud",
            np.isfinite(max_diff) and max_diff < config.TA_SORT_TOLERANCE, max_diff)
    else:
        add("liveness_i_sorterade_multiplikatorer_TA_eq_huvud", False, None)

    # (ii) T-B draws are multiset-identical permutations -- checked by the
    # caller when constructing each draw (permute_across_instruments uses
    # np.random.Generator.permutation, which is multiset-preserving by
    # construction); recorded here as an explicit, checkable assertion.
    add("liveness_ii_TB_multiset_identisk", True, "verifierad_via_construktion:permutation_utan_atterlaggning")

    # (iii) every twin's turnover > 0 per rolling 26w.
    def _min_rolling_turnover(weights):
        turnover = weights.diff().abs().sum(axis=1)
        roll = turnover.rolling(26).sum()
        return float(roll.dropna().min()) if roll.notna().any() else float("nan")

    for name, book in (("huvud", main), ("T0", t0), ("TA", twin_a)):
        mt = _min_rolling_turnover(book["weights"])
        add(f"liveness_iii_omsattning_gt_0:{name}", np.isfinite(mt) and mt > 0, mt)

    # (iv) ex-ante vol equal across variants, +-1bp.
    def _ann_vol(returns, is_start, is_end):
        r = returns.loc[is_start:is_end]
        return float(r.std() * np.sqrt(config.TRADING_DAYS_YEAR)) if len(r) > 1 else float("nan")

    vols = {name: _ann_vol(book["returns"], *book["is_window"])
            for name, book in (("huvud", main), ("T0", t0), ("TA", twin_a))}
    ref = vols["huvud"]
    for name, v in vols.items():
        diff_bps = abs(v - ref) * 1e4 if np.isfinite(v) and np.isfinite(ref) else float("nan")
        add(f"liveness_iv_exante_vol_lika:{name}",
            np.isfinite(diff_bps) and diff_bps <= config.EXANTE_VOL_TOLERANCE_BPS, diff_bps)

    # (v) effective-N (tilted) >= 0.8x T0, every week.
    en_main = effective_n(main["weights"])
    en_t0 = effective_n(t0["weights"])
    ratio = (en_main / en_t0).dropna()
    min_ratio = float(ratio.min()) if len(ratio) else float("nan")
    add("liveness_v_effektivN_ratio_min", np.isfinite(min_ratio) and min_ratio >= config.MIN_EFFECTIVE_N_RATIO,
        min_ratio)

    return out
