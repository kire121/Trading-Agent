"""
Vindkastet -- Step B: pre-registered Steg-1 gate checks on the primary universe.

  1. v*-stability week-to-week |cos| vs. a block-shuffled noise floor.
  2. median |cos(v*, PC1)| (failure mode #1 check).
  3. Redundancy screen: R^2(m_t, eta_K ~ battery), Jaccard(trigger days, top-5% vol days).
  4. Conditional vs unconditional forecast IC, with a block-shuffle null p-value.

Requires output/panels_cache.pkl from run_grid.py (uses the K=3, ridge=0.1 baseline cell,
plus fresh (K=2, ridge=0.1) triggers for the Jaccard check against the best grid cell).
"""
import pickle

import numpy as np
import pandas as pd

import core


def block_shuffle(returns, block=10, seed=0):
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    shuffled = returns.values[idx]
    return pd.DataFrame(shuffled, index=returns.index, columns=returns.columns)


def week_to_week_cos(panel):
    weekdiag = panel["week_diag_by_friday"]
    fridays = sorted(weekdiag.keys())
    cos_vals = []
    for i in range(1, len(fridays)):
        d0, d1 = weekdiag[fridays[i - 1]], weekdiag[fridays[i]]
        if d0["assets"] == d1["assets"]:
            cos_vals.append(abs(np.dot(d0["v_star"], d1["v_star"])))
    return np.array(cos_vals)


def pc1_alignment(panel, returns):
    Z = panel["Z"]
    weekdiag = panel["week_diag_by_friday"]
    fridays = sorted(weekdiag.keys())
    idx = returns.index
    pc1_cos = []
    for f in fridays:
        d = weekdiag[f]
        assets = d["assets"]
        i = idx.get_loc(f)
        block = Z[assets].iloc[i - 249: i + 1].values
        cov = np.cov(block, rowvar=False)
        w, v = np.linalg.eigh(cov)
        pc1 = v[:, -1]
        pc1_cos.append(abs(np.dot(d["v_star"], pc1)))
    return np.array(pc1_cos)


def redundancy_battery(panel, returns):
    cov_t = panel["cov_t"]
    days = sorted(cov_t.keys())
    realized_vol, mean_corr, absorption = [], [], []
    for d in days:
        Sig = cov_t[d]
        sd = np.sqrt(np.diag(Sig))
        corr = Sig / np.outer(sd, sd)
        n = corr.shape[0]
        off = corr[~np.eye(n, dtype=bool)]
        mean_corr.append(off.mean())
        realized_vol.append(sd.mean())
        w = np.sort(np.linalg.eigvalsh(Sig))[::-1]
        absorption.append(w[:4].sum() / w.sum())
    battery = pd.DataFrame({"realized_vol": realized_vol, "mean_corr": mean_corr,
                             "absorption": absorption}, index=days)
    ew_ret = returns.mean(axis=1)
    battery["skew"] = ew_ret.rolling(60).skew().reindex(battery.index)
    battery["absr_autocorr"] = ew_ret.abs().rolling(60).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False).reindex(battery.index)
    return battery.dropna()


def r_squared(y, X_df):
    X = np.column_stack([X_df.values, np.ones(len(X_df))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def conditional_ic(panel, returns, K):
    Z = panel["Z"]
    fdir = panel["forecast_dir"]
    idx = returns.index
    loc = {d: i for i, d in enumerate(idx)}
    forecast_scalar, realized_proj, align_vals = [], [], []
    for d in sorted(fdir.keys()):
        assets, fwd, proj = fdir[d]
        i = loc[d]
        if i + K >= len(idx):
            continue
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm < 1e-10:
            continue
        block = Z[assets].iloc[i + 1: i + 1 + K].values
        if np.isnan(block).any():
            continue
        forecast_scalar.append(proj * fwd_norm)
        realized_proj.append(block.sum(axis=0) @ (fwd / fwd_norm))
        align_vals.append(panel["alignment"].loc[d])
    return np.array(forecast_scalar), np.array(realized_proj), np.array(align_vals)


def main():
    prices = pd.read_csv("data/prices_primary.csv", index_col=0, parse_dates=True)
    returns = core.log_returns(prices)
    with open("output/panels_cache.pkl", "rb") as f:
        panels_cache = pickle.load(f)
    panel = panels_cache[(3, 0.1)]

    print("=== 1. v*-stability vs block-shuffled noise floor ===")
    cos_realized = week_to_week_cos(panel)
    null_stats = []
    for seed in range(6):
        r_shuf = block_shuffle(returns, block=10, seed=seed)
        panel_shuf = core.build_signal_panel(r_shuf, K=3, ridge_param=0.1)
        null_stats.append(week_to_week_cos(panel_shuf))
    null_pooled = np.concatenate(null_stats)
    print(f"realized median |cos|={np.median(cos_realized):.4f} mean={cos_realized.mean():.4f}")
    print(f"null     median |cos|={np.median(null_pooled):.4f} mean={null_pooled.mean():.4f}")
    np.save("output/cos_realized.npy", cos_realized)
    np.save("output/null_cos_stability.npy", null_pooled)

    print("\n=== 2. median cos(v*, PC1) ===")
    pc1_cos = pc1_alignment(panel, returns)
    print(f"median={np.median(pc1_cos):.4f} mean={pc1_cos.mean():.4f}")
    np.save("output/pc1_cos.npy", pc1_cos)

    print("\n=== 3. Redundancy screen ===")
    battery = redundancy_battery(panel, returns)
    m_t = panel["m_t"].reindex(battery.index)
    eta = panel["eta_K"].reindex(battery.index)
    mask = m_t.notna() & eta.notna()
    r2_m = r_squared(m_t[mask].values, battery[mask])
    r2_eta = r_squared(eta[mask].values, battery[mask])
    print(f"R^2(m_t ~ battery)={r2_m:.4f}  R^2(eta_K ~ battery)={r2_eta:.4f}  (kill if > 0.5)")
    trig = core.trigger_series(panels_cache[(2, 0.1)], quantile=0.90, align_thresh=0.50, eta_thresh=2.0)
    trig_days = set(trig[trig].index)
    vol_days = set(battery["realized_vol"][battery["realized_vol"] > battery["realized_vol"].quantile(0.95)].index)
    jacc = len(trig_days & vol_days) / len(trig_days | vol_days) if (trig_days | vol_days) else 0
    print(f"Jaccard(trigger days, top-5% vol days) = {jacc:.4f}  (kill if > 0.8)")
    battery.to_csv("output/battery.csv")

    print("\n=== 4. Conditional vs unconditional IC ===")
    fs, rp, av = conditional_ic(panel, returns, K=3)
    ic_uncond = np.corrcoef(fs, rp)[0, 1]
    print(f"n={len(fs)}  unconditional IC={ic_uncond:.4f}")
    for pct in [0.5, 0.75, 0.9, 0.95, 0.99]:
        thr = np.quantile(av, pct)
        mask = av >= thr
        ic_c = np.corrcoef(fs[mask], rp[mask])[0, 1]
        print(f"  conditional IC (align top {(1 - pct) * 100:.0f}%, n={mask.sum()}): {ic_c:.4f}")
    rng = np.random.default_rng(0)
    n = len(fs)
    null_ics = []
    for _ in range(500):
        n_blocks = int(np.ceil(n / 10))
        starts = rng.integers(0, max(n - 10, 1), size=n_blocks)
        perm_idx = np.clip(np.concatenate([np.arange(s, s + 10) for s in starts])[:n], 0, n - 1)
        null_ics.append(np.corrcoef(fs, rp[perm_idx])[0, 1])
    null_ics = np.array(null_ics)
    pval = (np.sum(np.abs(null_ics) >= abs(ic_uncond)) + 1) / (len(null_ics) + 1)
    print(f"block-shuffle null IC: mean={null_ics.mean():.4f} std={null_ics.std():.4f}  two-sided p={pval:.3f}")
    np.save("output/forecast_scalar.npy", fs)
    np.save("output/realized_proj.npy", rp)
    np.save("output/align_vals_ic.npy", av)


if __name__ == "__main__":
    main()
