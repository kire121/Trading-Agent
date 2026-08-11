"""PIT expanding-window additive day-position model + EB-shrunk word effect.

E[z_{t+1} | w] = alpha + sum_d beta_d * s_d + g(w)

Fit pooled across the whole panel (all assets together). Coefficients are
re-estimated on an expanding window at each month-end refit date (after a
burn-in period), and applied walk-forward (point-in-time: a row is scored
using the latest refit whose training window closed before that row's
target could have been realised) -- this produces genuinely out-of-fold
(OOF) predictions for K1a/K1c/K2 without any lookahead.

Two independent instances of this model are fit -- one for 5-day weeks
(32-cell table) and one for 4-day weeks (16-cell table) -- per the spec's
"4-dagarsveckor far egen 16-cellstabell" rule. They never share data.

Coefficient estimation inside the walk-forward loop uses a plain closed-form
OLS (numpy lstsq) for speed, since this loop runs thousands of times inside
the N1 permutation null. Cluster-robust (by ISO week) standard errors are
only needed for descriptive/reporting purposes and are computed separately,
once, via `final_ols_with_cluster_se` on the real (unpermuted) full sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from shrinkage import cell_shrinkage

MIN_TRAIN_OBS_MULTIPLIER = 5  # require >= 5 obs per regressor before a refit is usable


def design_matrix(words: list[tuple], word_len: int) -> np.ndarray:
    n = len(words)
    X = np.ones((n, word_len + 1))
    for j in range(word_len):
        X[:, j + 1] = [1.0 if w[j] == "+" else -1.0 for w in words]
    return X


def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


_ols_beta = ols_beta  # internal alias used throughout this module


def month_end_refit_dates(start: pd.Timestamp, end: pd.Timestamp, burn_in_years: int,
                           freq: str = config.REFIT_FREQ) -> pd.DatetimeIndex:
    burn_in_cutoff = start + pd.DateOffset(years=burn_in_years)
    if burn_in_cutoff > end:
        return pd.DatetimeIndex([])
    return pd.date_range(burn_in_cutoff, end, freq=freq)


class WalkForwardAdditiveModel:
    """Expanding-window additive + EB-shrunk word-cell model for one word length."""

    def __init__(self, word_len: int, kappa: float = 300.0, burn_in_years: int = 3,
                 refit_freq: str = config.REFIT_FREQ):
        self.word_len = word_len
        self.kappa = kappa
        self.refit_freq = refit_freq
        self.burn_in_years = burn_in_years
        self.refit_dates: list[pd.Timestamp] = []
        self.betas: list[np.ndarray] = []
        self.cell_maps: list[dict] = []
        self.n_train: list[int] = []

    def fit_walkforward(self, df: pd.DataFrame) -> "WalkForwardAdditiveModel":
        """df must have columns: t_signal, t_target_end, word, z_next (word length == self.word_len)."""
        df = df.dropna(subset=["z_next"]).sort_values("t_target_end").reset_index(drop=True)
        self.refit_dates, self.betas, self.cell_maps, self.n_train = [], [], [], []
        if df.empty:
            return self

        start, end = df["t_target_end"].min(), df["t_target_end"].max()
        candidate_dates = month_end_refit_dates(start, end, self.burn_in_years, freq=self.refit_freq)
        if len(candidate_dates) == 0:
            return self

        words = df["word"].tolist()
        X_full = design_matrix(words, self.word_len)
        y_full = df["z_next"].to_numpy(dtype=float)
        target_end = df["t_target_end"].to_numpy()
        min_obs = MIN_TRAIN_OBS_MULTIPLIER * (self.word_len + 1)

        for R in candidate_dates:
            mask = target_end <= np.datetime64(R)
            n = int(mask.sum())
            if n < min_obs:
                continue
            Xw, yw = X_full[mask], y_full[mask]
            beta = _ols_beta(Xw, yw)
            resid = yw - Xw @ beta
            words_w = [words[i] for i in np.nonzero(mask)[0]]
            cell_map = cell_shrinkage(words_w, resid, self.kappa)
            self.refit_dates.append(pd.Timestamp(R))
            self.betas.append(beta)
            self.cell_maps.append(cell_map)
            self.n_train.append(n)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame(index=df.index) with columns additive_pred, ghat, full_pred.

        NaN for rows scored before the first refit (burn-in) or with no
        matching table (caller's responsibility to pre-filter by word_len).
        """
        out = pd.DataFrame(
            {"additive_pred": np.nan, "ghat": np.nan, "full_pred": np.nan},
            index=df.index,
        )
        if not self.refit_dates or df.empty:
            return out

        refit_df = pd.DataFrame({
            "refit_date": self.refit_dates,
            "ridx": np.arange(len(self.refit_dates)),
        }).sort_values("refit_date")

        order = df[["t_signal"]].sort_values("t_signal")
        merged = pd.merge_asof(
            order, refit_df, left_on="t_signal", right_on="refit_date", direction="backward",
        )
        merged.index = order.index
        ridx = merged["ridx"].reindex(df.index)

        words = df["word"].tolist()
        X = design_matrix(words, self.word_len)

        additive_pred = np.full(len(df), np.nan)
        ghat = np.full(len(df), np.nan)
        pos = np.arange(len(df))
        for r_i in ridx.dropna().unique():
            r_i = int(r_i)
            sel = (ridx.to_numpy() == r_i)
            beta = self.betas[r_i]
            additive_pred[sel] = X[sel] @ beta
            cell_map = self.cell_maps[r_i]
            g_sel = np.array([cell_map.get(words[p], 0.0) for p in pos[sel]])
            ghat[sel] = g_sel

        out["additive_pred"] = additive_pred
        out["ghat"] = ghat
        out["full_pred"] = additive_pred + ghat
        return out


def final_ols_with_cluster_se(df: pd.DataFrame, word_len: int):
    """One-off statsmodels OLS with week-clustered SE on the full realised sample.

    For descriptive reporting of the day-position coefficients only -- not
    used anywhere in the walk-forward / kill-criteria machinery.
    """
    import statsmodels.api as sm

    df = df.dropna(subset=["z_next"]).copy()
    X = design_matrix(df["word"].tolist(), word_len)
    y = df["z_next"].to_numpy(dtype=float)
    iso = df["t_signal"].dt.isocalendar()
    cluster_id = (iso["year"].astype(str) + "-W" + iso["week"].astype(str)).to_numpy()
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_id})
    return res
