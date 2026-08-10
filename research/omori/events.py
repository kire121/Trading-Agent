"""Event candidate detection (vectorized, stateless) and same-day clustering
(stateful, applied inside backtest.py's simulation loop since it also needs
to know which instruments currently hold an open position).

Event condition: dollar-volume-z (rolling 120d median/MAD, causal) >= z_thr
AND |r0| >= sigma_mult * sigma_hat_60 (causal trailing realized vol of daily
adjusted-close returns).
"""
import numpy as np
import pandas as pd

from research.omori import config, data


class EventFields:
    """Precomputed per-(date, ticker) fields needed for detection, fitting,
    and clustering, for one Panel."""

    def __init__(self, panel: data.Panel):
        self.panel = panel
        dv = panel.dollar_volume()
        self.volume_z = data.rolling_median_mad_z(dv, config.DOLLAR_VOLUME_LOOKBACK)
        self.dv_baseline = dv.shift(1).rolling(
            config.DOLLAR_VOLUME_LOOKBACK, min_periods=config.DOLLAR_VOLUME_LOOKBACK
        ).median()
        self.dollar_volume = dv
        self.returns = panel.simple_returns()
        self.sigma60 = data.rolling_return_sigma(self.returns, config.RETURN_VOL_LOOKBACK)
        self.adv = data.rolling_adv(dv, config.ADV_COST_LOOKBACK)
        self.direction = np.sign(self.returns)

    def excess_volume_series(self, ticker):
        """e(tau) for every calendar day of `ticker`'s history: relative
        excess of dollar volume over the causal rolling-120d median
        baseline. NaN where the baseline is undefined."""
        dv = self.dollar_volume[ticker]
        base = self.dv_baseline[ticker]
        return (dv - base) / base.replace(0.0, np.nan)

    def candidate_mask(self, z_threshold=config.Z_VOLUME_THRESHOLD,
                        sigma_mult=config.R0_SIGMA_THRESHOLD):
        return (self.volume_z >= z_threshold) & (self.returns.abs() >= sigma_mult * self.sigma60)

    def candidates_long(self, z_threshold=config.Z_VOLUME_THRESHOLD,
                         sigma_mult=config.R0_SIGMA_THRESHOLD):
        """Long-format table of ALL raw candidate (date, ticker) events,
        ignoring the open-instrument and same-day-clustering de-dup rules
        (those require simulation state and are applied in backtest.py)."""
        mask = self.candidate_mask(z_threshold, sigma_mult)
        rows = []
        idx_pos = {d: i for i, d in enumerate(self.panel.index)}
        for ticker in mask.columns:
            hits = mask.index[mask[ticker].fillna(False)]
            for d in hits:
                rows.append({
                    "date": d,
                    "ticker": ticker,
                    "date_idx": idx_pos[d],
                    "r0": self.returns.loc[d, ticker],
                    "volume_z": self.volume_z.loc[d, ticker],
                    "direction": float(self.direction.loc[d, ticker]),
                })
        cols = ["date", "ticker", "date_idx", "r0", "volume_z", "direction"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows)[cols].sort_values(["date", "ticker"]).reset_index(drop=True)


def cluster_same_day(candidates, corr_matrix, threshold=config.CLUSTER_CORR_THRESHOLD):
    """candidates: list of (ticker, z) tuples for a single day, already
    filtered to exclude instruments with an open position. corr_matrix:
    trailing pairwise |corr| DataFrame (tickers x tickers) as of that day.
    Returns the list of surviving tickers: connected components (edge =
    |corr| > threshold) are collapsed to their highest-z member."""
    if len(candidates) <= 1:
        return [t for t, _ in candidates]

    tickers = [t for t, _ in candidates]
    z_of = dict(candidates)
    parent = {t: t for t in tickers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            if a not in corr_matrix.index or b not in corr_matrix.columns:
                continue
            rho = corr_matrix.loc[a, b]
            if pd.notna(rho) and abs(rho) > threshold:
                union(a, b)

    clusters = {}
    for t in tickers:
        clusters.setdefault(find(t), []).append(t)

    survivors = []
    for members in clusters.values():
        survivors.append(max(members, key=lambda t: z_of[t]))
    return survivors
