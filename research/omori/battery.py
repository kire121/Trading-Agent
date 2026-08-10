"""Redundancy screening: is p_hat (or the realized post-event drift
half-life) already explained by a battery of known factors? Two diagnostic
regressions, both event-level, both run BEFORE the backtest is trusted:

  1. p_hat ~ battery + event covariates.  Kill if R^2 >= REDUNDANCY_R2_KILL
     -- p_hat would then be redundant with (e.g.) GARCH-style vol
     persistence in disguise (expected weakness #2 in the brief).
  2. realized_half_life ~ battery + event covariates [+ p_hat].  Kill if
     adding p_hat contributes ~zero incremental R^2 over the battery alone
     -- p_hat would then carry no information about how long the post-event
     drift actually persists, which is the whole point of using it as an
     exit clock.

Battery: trailing realized vol, |r|-autocorrelation, skew, mean pairwise
correlation to the rest of the panel, and the panel's absorption ratio
(Kritzman et al.: variance share of the top-K PCs of the trailing return
correlation matrix) -- plus event covariates volume_z, |r0|, and the
overnight-gap share of the event-day return.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from research.omori import config, data, events, priors, signal

BATTERY_LOOKBACK = 60          # DECLARED: matches RETURN_VOL_LOOKBACK convention
ABSORPTION_N_COMPONENTS_FRAC = 0.2   # DECLARED: top 20% of eligible instruments'
                                       # components, Kritzman's usual convention


def _abs_r_autocorr(returns_window):
    a = np.abs(returns_window)
    if len(a) < 10 or a.std() == 0:
        return np.nan
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def absorption_ratio(returns_panel, t_idx, lookback=BATTERY_LOOKBACK):
    start = max(0, t_idx - lookback)
    window = returns_panel.iloc[start:t_idx]
    cols = window.columns[window.notna().sum() >= max(20, lookback // 3)]
    window = window[cols].dropna(axis=0, how="any")
    if window.shape[1] < 5 or window.shape[0] < 20:
        return np.nan
    corr = window.corr().values
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.sort(eigvals)[::-1]
    k = max(1, int(round(ABSORPTION_N_COMPONENTS_FRAC * len(eigvals))))
    return float(eigvals[:k].sum() / eigvals.sum())


def battery_row(ef: "events.EventFields", ticker, t0_idx, lookback=BATTERY_LOOKBACK):
    returns = ef.returns
    r_window = returns[ticker].iloc[max(0, t0_idx - lookback):t0_idx].dropna().values
    realized_vol = float(np.std(r_window, ddof=1)) if len(r_window) > 5 else np.nan
    skew = float(pd.Series(r_window).skew()) if len(r_window) > 5 else np.nan
    abs_autocorr = _abs_r_autocorr(r_window)

    corr_mat = data.trailing_pairwise_corr(returns, lookback, t0_idx)
    if ticker in corr_mat.index:
        others = corr_mat.loc[ticker].drop(labels=[ticker], errors="ignore")
        mean_corr = float(others.mean()) if others.notna().any() else np.nan
    else:
        mean_corr = np.nan

    abs_ratio = absorption_ratio(returns, t0_idx, lookback)

    open_ = ef.panel.raw["open"][ticker].iloc[t0_idx]
    close_ = ef.panel.raw["close"][ticker].iloc[t0_idx]
    prev_close = ef.panel.raw["close"][ticker].iloc[t0_idx - 1] if t0_idx > 0 else np.nan
    r0 = ef.returns[ticker].iloc[t0_idx]
    if np.isfinite(open_) and np.isfinite(prev_close) and prev_close != 0 and np.isfinite(r0) and r0 != 0:
        gap_ret = open_ / prev_close - 1.0
        gap_share = float(gap_ret / r0)
    else:
        gap_share = np.nan

    return {
        "realized_vol": realized_vol, "abs_r_autocorr": abs_autocorr, "skew": skew,
        "mean_corr": mean_corr, "absorption_ratio": abs_ratio, "gap_share": gap_share,
        "volume_z": ef.volume_z[ticker].iloc[t0_idx], "abs_r0": abs(r0) if np.isfinite(r0) else np.nan,
    }


BATTERY_COLS = ["realized_vol", "abs_r_autocorr", "skew", "mean_corr", "absorption_ratio",
                "gap_share", "volume_z", "abs_r0"]


def build_battery_frame(panel, ef, terminal_fits_list, priors_dict, kappa=config.KAPPA_DEFAULT,
                         cap=config.TAU_EXIT_CAP_DEFAULT):
    """terminal_fits_list: output of priors.terminal_fits(panel) -- one
    terminal Omori fit per raw candidate event. Adds battery covariates,
    the EB-shrunk p_tilde, and the realized post-event drift half-life to
    each identified event."""
    rows = []
    adj = panel.adj_close
    n = len(panel.index)
    for f in terminal_fits_list:
        if not f["identified"]:
            continue
        ticker, t0_idx = f["ticker"], f["t0_idx"]
        direction = float(np.sign(ef.returns[ticker].iloc[t0_idx]))
        max_tau = min(cap, n - 1 - t0_idx)
        if max_tau < 5:
            continue
        p0 = adj[ticker].iloc[t0_idx]
        path = adj[ticker].iloc[t0_idx + 1: t0_idx + 1 + max_tau].values
        if not np.isfinite(p0) or p0 == 0 or np.any(~np.isfinite(path)):
            continue
        cfr = direction * (path / p0 - 1.0)
        terminal_drift = cfr[-1]
        half_life = np.nan
        if terminal_drift > 0:
            target = 0.5 * terminal_drift
            hits = np.where(cfr >= target)[0]
            if len(hits):
                half_life = float(hits[0] + 1)

        fit = signal.FitResult(p_hat=f["p_hat"], k_hat=np.nan, c_hat=f["c_hat"],
                                n_pos=f["n_pos"], identified=True)
        prior_p = priors.prior_for(ticker, priors_dict)
        p_tilde = signal.shrink(fit, prior_p, kappa)

        b = battery_row(ef, ticker, t0_idx)
        b.update({"ticker": ticker, "t0_idx": t0_idx, "p_hat": f["p_hat"], "c_hat": f["c_hat"],
                   "n_pos": f["n_pos"], "p_tilde": p_tilde,
                   "terminal_drift": terminal_drift, "half_life": half_life})
        rows.append(b)
    return pd.DataFrame(rows)


def _ols_r2(y, X):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 20:
        return np.nan, mask.sum()
    Xc = sm.add_constant(X[mask])
    fit = sm.OLS(y[mask], Xc).fit()
    return float(fit.rsquared), int(mask.sum())


def redundancy_screen(battery_df):
    """Returns dict with p_hat~battery R^2 and the half_life incremental R^2
    from adding p_hat, plus the kill verdicts."""
    X_battery = battery_df[BATTERY_COLS].values.astype(float)

    r2_phat, n_phat = _ols_r2(battery_df["p_hat"].values.astype(float), X_battery)

    hl_mask = battery_df["half_life"].notna()
    hl = battery_df.loc[hl_mask]
    y_hl = hl["half_life"].values.astype(float)
    X_base = hl[BATTERY_COLS].values.astype(float)
    X_aug = hl[BATTERY_COLS + ["p_hat"]].values.astype(float)
    r2_base, n_base = _ols_r2(y_hl, X_base)
    r2_aug, n_aug = _ols_r2(y_hl, X_aug)
    delta_r2 = (r2_aug - r2_base) if np.isfinite(r2_aug) and np.isfinite(r2_base) else np.nan

    kill_phat = np.isfinite(r2_phat) and r2_phat >= config.REDUNDANCY_R2_KILL
    kill_incremental = np.isfinite(delta_r2) and delta_r2 <= 0.0

    return {
        "r2_phat_vs_battery": r2_phat, "n_phat": n_phat,
        "r2_halflife_baseline": r2_base, "r2_halflife_augmented": r2_aug,
        "delta_r2_halflife": delta_r2, "n_halflife": n_base,
        "kill_phat_redundant": bool(kill_phat), "kill_zero_incremental": bool(kill_incremental),
        "killed": bool(kill_phat or kill_incremental),
    }
