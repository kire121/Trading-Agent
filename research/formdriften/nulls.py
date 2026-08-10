"""
The null-hypothesis battery. Every test here is a way for the strategy to
die; none of it is optional decoration. Order follows the spec:
redundancy screen first (gates everything else), then the estimator null
(mandatory fast-exit), block permutation, and the turnover-matched noise
floor.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap

from . import metrics
from .portfolio import BacktestConfig, run_backtest
from .signal import rolling_d_l
from .twins import TWIN_NAMES, build_twin_panels


# ---------------------------------------------------------------------------
# Redundancy screen (twin suite) -- run before everything else
# ---------------------------------------------------------------------------

def redundancy_screen(returns_panel, d_l_panel, eligible_panel, curr_window=126, prev_window=126):
    """Pooled R^2 of D_L on the four twins, and a Fama-MacBeth incremental-IC
    test of whether D_L still predicts next-month return once the twins are
    controlled for. Kill if pooled R^2 > 0.5 or incremental t-stat < 2.
    """
    eval_dates = list(d_l_panel.index)
    twin_panels = build_twin_panels(returns_panel, eval_dates, curr_window, prev_window)

    # --- pooled regression: D_L ~ const + twins, across all (date, asset) obs ---
    rows = []
    for dt in eval_dates:
        for col in returns_panel.columns:
            if not eligible_panel.loc[dt, col]:
                continue
            dl = d_l_panel.loc[dt, col]
            twin_vals = [twin_panels[name].loc[dt, col] for name in TWIN_NAMES]
            if pd.isna(dl) or any(pd.isna(v) for v in twin_vals):
                continue
            rows.append([dl] + twin_vals + [dt, col])
    pooled = pd.DataFrame(rows, columns=["d_l"] + TWIN_NAMES + ["date", "ticker"])

    if len(pooled) < 30:
        return {"pooled_r2": np.nan, "fm_incremental_tstat": np.nan, "n_obs": len(pooled),
                "pooled": pooled, "twin_panels": twin_panels}

    X = sm.add_constant(pooled[TWIN_NAMES].values)
    y = pooled["d_l"].values
    ols = sm.OLS(y, X).fit()
    pooled_r2 = ols.rsquared

    # --- Fama-MacBeth: monthly cross-sectional regression of next-period
    # return on standardized [D_L, twins], collect D_L's coefficient path.
    # Forward return is computed directly from the price panel between
    # consecutive rebalance dates. ---
    coefs = []
    for i in range(len(eval_dates) - 1):
        dt, dt_next = eval_dates[i], eval_dates[i + 1]
        elig = eligible_panel.loc[dt]
        names = elig.index[elig.fillna(False)]
        dl = d_l_panel.loc[dt, names]
        tw = pd.DataFrame({name: twin_panels[name].loc[dt, names] for name in TWIN_NAMES})
        # forward return from this rebalance to the next, per asset
        px_now = (1 + returns_panel[names]).cumprod()
        try:
            fwd = px_now.loc[dt_next] / px_now.loc[dt] - 1
        except KeyError:
            continue
        df = pd.concat([dl.rename("d_l"), tw, fwd.rename("fwd")], axis=1).dropna()
        if len(df) < 8:
            continue
        z = (df[["d_l"] + TWIN_NAMES] - df[["d_l"] + TWIN_NAMES].mean()) / df[["d_l"] + TWIN_NAMES].std(ddof=0)
        z = z.fillna(0.0)
        Xi = sm.add_constant(z.values)
        yi = df["fwd"].values
        try:
            m = sm.OLS(yi, Xi).fit()
            coefs.append(m.params[1])  # coefficient on d_l (column order: const, d_l, twins...)
        except Exception:
            continue

    coefs = np.array(coefs)
    if len(coefs) > 5 and coefs.std(ddof=1) > 0:
        fm_tstat = coefs.mean() / (coefs.std(ddof=1) / np.sqrt(len(coefs)))
    else:
        fm_tstat = np.nan

    return {
        "pooled_r2": pooled_r2,
        "fm_incremental_tstat": fm_tstat,
        "fm_mean_coef": float(coefs.mean()) if len(coefs) else np.nan,
        "fm_n_months": len(coefs),
        "n_obs": len(pooled),
        "pooled": pooled,
        "twin_panels": twin_panels,
    }


def twin_portfolio_sharpes(returns_panel, twin_panels, eligible_panel, cfg: BacktestConfig, adv_panel):
    """Run the identical portfolio-construction machinery on each twin factor
    (in place of D_L) so the OT signal's Sharpe can be compared with margin."""
    out = {}
    eval_dates = list(next(iter(twin_panels.values())).index)
    for name in TWIN_NAMES:
        res = run_backtest(returns_panel.fillna(0), twin_panels[name], eligible_panel, adv_panel, cfg,
                            rebalance_dates=eval_dates)
        out[name] = {"sharpe": metrics.sharpe(res["net_returns"]), "returns": res["net_returns"]}
    return out


# ---------------------------------------------------------------------------
# (1) Estimator null: stationary bootstrap, mandatory fast-exit gate
# ---------------------------------------------------------------------------

def estimator_null(returns_panel, curr_window=126, prev_window=126, n_reps=25, block_size=20,
                    eval_every=21, n_synthetic_cross_sections=2000, seed=0):
    rng = np.random.default_rng(seed)
    min_t = curr_window + prev_window
    null_pool = {}
    persistence_null = {}

    for col in returns_panel.columns:
        r = returns_panel[col].fillna(0.0).values
        n = len(r)
        positions = list(range(min_t, n, eval_every))
        if len(positions) < 10:
            continue
        bs = StationaryBootstrap(block_size, r, seed=int(rng.integers(0, 2**31 - 1)))
        pool, pers = [], []
        for data, _ in bs.bootstrap(n_reps):
            rep = data[0]
            d_l_rep = rolling_d_l(rep, curr_window, prev_window, at_positions=positions)
            valid = d_l_rep[~np.isnan(d_l_rep)]
            if len(valid) > 0:
                pool.append(valid)
            if len(valid) > 5:
                c = np.corrcoef(valid[:-1], valid[1:])[0, 1]
                if np.isfinite(c):
                    pers.append(c)
        null_pool[col] = np.concatenate(pool) if pool else np.array([])
        persistence_null[col] = np.array(pers)

    assets = [a for a in null_pool if len(null_pool[a]) > 0]
    disp_null = []
    for _ in range(n_synthetic_cross_sections):
        draw = [rng.choice(null_pool[a]) for a in assets]
        if len(draw) >= 5:
            disp_null.append(np.std(draw))
    return {
        "dispersion_null": np.array(disp_null),
        "persistence_null": persistence_null,
        "null_pool": null_pool,
    }


def evaluate_estimator_null(d_l_panel, eligible_panel, null_result):
    """Compare real cross-sectional dispersion & per-asset persistence against
    the stationary-bootstrap null. Require both to exceed the null's 95th pct."""
    real_disp = []
    for dt in d_l_panel.index:
        elig = eligible_panel.loc[dt]
        vals = d_l_panel.loc[dt, elig.index[elig.fillna(False)]].dropna()
        if len(vals) >= 5:
            real_disp.append(vals.std())
    real_disp = np.array(real_disp)

    real_persist = {}
    for col in d_l_panel.columns:
        s = d_l_panel[col].dropna()
        if len(s) > 5:
            c = s.autocorr(lag=1)
            if np.isfinite(c):
                real_persist[col] = c

    disp_null = null_result["dispersion_null"]
    disp_thresh = np.percentile(disp_null, 95) if len(disp_null) else np.nan
    disp_pass = np.nanmean(real_disp) > disp_thresh if len(real_disp) else False

    persist_pass_count, persist_total = 0, 0
    per_asset_verdict = {}
    for col, real_p in real_persist.items():
        null_p = null_result["persistence_null"].get(col, np.array([]))
        if len(null_p) < 5:
            continue
        thresh = np.percentile(null_p, 95)
        passed = real_p > thresh
        per_asset_verdict[col] = {"real": real_p, "null_p95": thresh, "pass": bool(passed)}
        persist_total += 1
        persist_pass_count += int(passed)

    return {
        "real_dispersion_mean": float(np.nanmean(real_disp)) if len(real_disp) else np.nan,
        "null_dispersion_p95": float(disp_thresh),
        "dispersion_pass": bool(disp_pass),
        "persist_pass_fraction": persist_pass_count / persist_total if persist_total else np.nan,
        "per_asset_persistence": per_asset_verdict,
        "PASS": bool(disp_pass) and (persist_pass_count / persist_total > 0.5 if persist_total else False),
    }


# ---------------------------------------------------------------------------
# (2) Block permutation
# ---------------------------------------------------------------------------

def block_permutation_test(returns_panel, d_l_panel, eligible_panel, adv_panel, cfg: BacktestConfig,
                            n_perms=150, block_months=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = list(d_l_panel.index)
    n = len(dates)

    real_res = run_backtest(returns_panel.fillna(0), d_l_panel, eligible_panel, adv_panel, cfg,
                             rebalance_dates=dates)
    real_sharpe = metrics.sharpe(real_res["net_returns"])

    n_blocks = int(np.ceil(n / block_months))
    perm_sharpes = np.empty(n_perms)
    for k in range(n_perms):
        block_order = rng.permutation(n_blocks)
        new_order = []
        for b in block_order:
            start = b * block_months
            new_order.extend(range(start, min(start + block_months, n)))
        new_order = new_order[:n]
        shuffled_d_l = d_l_panel.iloc[new_order].copy()
        shuffled_d_l.index = d_l_panel.index
        shuffled_elig = eligible_panel.iloc[new_order].copy()
        shuffled_elig.index = eligible_panel.index
        res = run_backtest(returns_panel.fillna(0), shuffled_d_l, shuffled_elig, adv_panel, cfg,
                            rebalance_dates=dates)
        perm_sharpes[k] = metrics.sharpe(res["net_returns"])

    pval = (np.sum(perm_sharpes >= real_sharpe) + 1) / (n_perms + 1)
    return {"real_sharpe": real_sharpe, "perm_sharpes": perm_sharpes, "pvalue": float(pval)}


# ---------------------------------------------------------------------------
# (4) Noise floor: identical construction mechanism fed random rankings
# ---------------------------------------------------------------------------

def noise_floor_test(returns_panel, d_l_panel, eligible_panel, adv_panel, cfg: BacktestConfig,
                      n_reps=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = list(d_l_panel.index)

    real_res = run_backtest(returns_panel.fillna(0), d_l_panel, eligible_panel, adv_panel, cfg,
                             rebalance_dates=dates)
    real_sharpe = metrics.sharpe(real_res["net_returns"])
    real_turnover = real_res["turnover"]["turnover"].mean()

    null_sharpes = np.empty(n_reps)
    null_turnovers = np.empty(n_reps)
    for k in range(n_reps):
        noise = pd.DataFrame(
            rng.normal(size=d_l_panel.shape), index=d_l_panel.index, columns=d_l_panel.columns
        )
        noise = noise.where(d_l_panel.notna())  # same NaN/eligibility footprint as real signal
        res = run_backtest(returns_panel.fillna(0), noise, eligible_panel, adv_panel, cfg,
                            rebalance_dates=dates)
        null_sharpes[k] = metrics.sharpe(res["net_returns"])
        null_turnovers[k] = res["turnover"]["turnover"].mean()

    pval = (np.sum(null_sharpes >= real_sharpe) + 1) / (n_reps + 1)
    return {
        "real_sharpe": real_sharpe,
        "real_turnover": real_turnover,
        "null_sharpes": null_sharpes,
        "null_turnover_mean": float(np.mean(null_turnovers)),
        "pvalue": float(pval),
    }
