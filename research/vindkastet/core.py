"""
Vindkastet -- core propagator engine.

z_t = A z_{t-1} + eps_t   (ridge-regularized VAR(1) on EWMA-vol-standardized
log returns), re-estimated every Friday close on a rolling 250-trading-day
window. From A we take the SVD of A^K to get the maximally-amplified shock
direction v* (top right singular vector), the direction energy lands along
u* (top left singular vector) and the transient-growth ratio
eta_K = sigma_max(A^K) / rho(A)^K.

All vol/covariance estimates used to standardize day-t returns are built
strictly from data <= t-1 (RiskMetrics-style EWMA recursion), so nothing here
uses information unavailable at the close of day t.
"""
import numpy as np
import pandas as pd

LAMBDA = 0.97
WINDOW = 250


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def ewma_moments(returns: pd.DataFrame, lam: float = LAMBDA):
    """
    Returns (sigma, cov) panels where sigma.loc[t] / cov.loc[t] are the
    EWMA vol vector / covariance matrix available for use AT THE CLOSE of
    day t, i.e. built only from returns dated <= t-1.

    cov is returned as a dict[Timestamp] -> (N,N) ndarray (a DataFrame of
    matrices doesn't exist natively).
    """
    r = returns.values
    T, N = r.shape
    idx = returns.index
    cols = returns.columns

    sigma = np.full((T, N), np.nan)
    cov_t = {}

    # seed with simple variance/covariance of the first valid 20 obs
    first_valid = np.where(~np.isnan(r).any(axis=1))[0]
    if len(first_valid) < 21:
        raise ValueError("Not enough overlapping history to seed EWMA moments")
    seed_start = first_valid[0]
    seed_end = seed_start + 20
    seed_block = r[seed_start:seed_end]
    Sigma = np.cov(seed_block, rowvar=False)
    if N == 1:
        Sigma = np.array([[Sigma]])

    for t in range(seed_end, T):
        # Sigma currently represents info available through t-1 (i.e. it was
        # last updated using r[t-1]); record it as the estimate usable AT t.
        sigma[t] = np.sqrt(np.diag(Sigma))
        cov_t[idx[t]] = Sigma.copy()
        r_prev = r[t]  # will become "t-1" for the NEXT iteration's use
        if not np.isnan(r_prev).any():
            Sigma = lam * Sigma + (1 - lam) * np.outer(r_prev, r_prev)

    sigma_df = pd.DataFrame(sigma, index=idx, columns=cols)
    return sigma_df, cov_t


def standardize(returns: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    z = returns / sigma
    return z


def ridge_var1_fit(Z: np.ndarray, ridge_param: float):
    """
    Fit z_t = A z_{t-1} + eps via ridge regression on a (T, N) block of
    standardized returns Z (rows = consecutive trading days, no gaps).

    ridge_param is a SCALE-FREE regularization fraction: the effective ridge
    penalty is ridge_param * trace(X'X)/N, i.e. ridge_param=0.1 shrinks each
    equation's coefficients by an amount comparable to 10% of the average
    eigenvalue of the design's Gram matrix. This keeps the ridge grid
    {0.05, 0.1} meaningful regardless of the absolute scale of z.
    """
    X = Z[:-1]
    Y = Z[1:]
    N = X.shape[1]
    XtX = X.T @ X
    alpha = ridge_param * np.trace(XtX) / N
    A_T = np.linalg.solve(XtX + alpha * np.eye(N), X.T @ Y)
    A = A_T.T
    return A


def propagator_diagnostics(A: np.ndarray, K: int):
    eigvals = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigvals))
    AK = np.linalg.matrix_power(A, K)
    U, S, Vt = np.linalg.svd(AK)
    v_star = Vt[0]
    u_star = U[:, 0]
    sigma_max = S[0]
    eta_K = sigma_max / (rho ** K) if rho > 1e-8 else np.inf
    return {
        "A": A, "rho": rho, "AK": AK, "v_star": v_star, "u_star": u_star,
        "sigma_max": sigma_max, "eta_K": eta_K, "eigvals": eigvals,
    }


def week_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: True on the last trading day of each ISO week (our proxy for 'Friday close')."""
    iso = index.isocalendar()
    week_id = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    is_last = week_id.values[:-1] != week_id.values[1:]
    return np.append(is_last, True)


def build_signal_panel(returns: pd.DataFrame, K: int, ridge_param: float,
                        window: int = WINDOW, lam: float = LAMBDA,
                        min_assets: int = 12):
    """
    Walk the full history and, at every Friday (week-end) close with a full
    trailing `window` of complete-case standardized returns, refit A and
    derive v*, u*, eta_K. Those quantities are then held fixed and used to
    score every trading day in the following week (no re-use of future
    weeks' A). Returns a dict of per-day arrays plus per-week metadata.
    """
    sigma, cov_t = ewma_moments(returns, lam=lam)
    Z = standardize(returns, sigma)

    idx = returns.index
    N = returns.shape[1]
    is_week_end = week_end_mask(idx)

    m_t = pd.Series(np.nan, index=idx)
    alignment = pd.Series(np.nan, index=idx)
    eta_series = pd.Series(np.nan, index=idx)
    rho_series = pd.Series(np.nan, index=idx)
    n_assets_series = pd.Series(np.nan, index=idx)
    v_star_of_day = {}
    forecast_dir = {}  # t -> sum_{k=1}^K A^k v* (per active day, from the week's A)
    week_diag_by_friday = {}

    current = None  # diagnostics dict from most recently completed Friday fit
    current_assets = None

    for t in range(len(idx)):
        if is_week_end[t] and t >= window - 1:
            block = Z.iloc[t - window + 1: t + 1]
            complete_cols = block.columns[~block.isna().any(axis=0)]
            if len(complete_cols) >= min_assets and not block[complete_cols].isna().any().any():
                Zb = block[complete_cols].values
                if not np.isnan(Zb).any():
                    A = ridge_var1_fit(Zb, ridge_param)
                    diag = propagator_diagnostics(A, K)
                    diag["assets"] = list(complete_cols)
                    if diag["rho"] < 1.0:
                        current = diag
                        current_assets = list(complete_cols)
                        week_diag_by_friday[idx[t]] = diag

        if current is not None:
            assets = current_assets
            zt = Z.iloc[t][assets].values
            if not np.isnan(zt).any():
                v = current["v_star"]
                proj = float(zt @ v)
                m_t.iloc[t] = abs(proj)
                znorm = np.linalg.norm(zt)
                alignment.iloc[t] = abs(proj) / znorm if znorm > 1e-12 else np.nan
                eta_series.iloc[t] = current["eta_K"]
                rho_series.iloc[t] = current["rho"]
                n_assets_series.iloc[t] = len(assets)
                v_star_of_day[idx[t]] = (assets, v, current["u_star"])
                fwd = np.zeros(len(assets))
                Ak = np.eye(len(assets))
                for k in range(1, K + 1):
                    Ak = Ak @ current["A"]
                    fwd = fwd + Ak @ v
                forecast_dir[idx[t]] = (assets, fwd, proj)

    return {
        "sigma": sigma, "cov_t": cov_t, "Z": Z,
        "m_t": m_t, "alignment": alignment, "eta_K": eta_series, "rho": rho_series,
        "n_assets": n_assets_series, "v_star_of_day": v_star_of_day,
        "forecast_dir": forecast_dir, "week_diag_by_friday": week_diag_by_friday,
    }


def trigger_series(panel: dict, quantile: float = 0.95, align_thresh: float = 0.65,
                    eta_thresh: float = 2.0, pctl_window: int = 250) -> pd.Series:
    m_t = panel["m_t"]
    pctl = m_t.rolling(pctl_window, min_periods=pctl_window).apply(
        lambda x: (x.iloc[-1] > np.quantile(x.iloc[:-1].dropna(), quantile)) if len(x.iloc[:-1].dropna()) > 0 else 0.0,
        raw=False,
    )
    m_cond = pctl.fillna(0).astype(bool)
    align_cond = (panel["alignment"] > align_thresh).fillna(False)
    eta_cond = (panel["eta_K"] > eta_thresh).fillna(False)
    trig = m_cond & align_cond & eta_cond
    return trig
