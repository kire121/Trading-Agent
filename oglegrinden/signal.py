"""Weekly signal pipeline: raw topology/correlation series -> smoothing ->
expanding-percentile gate with hysteresis.

Everything here is careful to use only information available as of each
Friday close (point-in-time universe, expanding history for percentile
thresholds) so the backtest that consumes these gates is not look-ahead
biased.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from oglegrinden.data import Panel
from oglegrinden.topology import compute_topology, residualize_against
from oglegrinden.universe import MIN_ADV_USD, MIN_UNIVERSE_SIZE, BENCHMARK


def weekly_fridays(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last actual trading day of each ISO calendar week present in `index`
    (usually Friday, but holiday-shortened weeks end earlier). This is the
    strategy's decision date."""
    iso = index.isocalendar()
    key = pd.MultiIndex.from_arrays([iso["year"].values, iso["week"].values])
    s = pd.Series(index, index=key)
    last_per_week = s.groupby(level=[0, 1]).max()
    return pd.DatetimeIndex(sorted(last_per_week.values))


def compute_raw_signal_series(
    panel: Panel,
    corr_window: int,
    min_universe_size: int = MIN_UNIVERSE_SIZE,
    min_adv: float = MIN_ADV_USD,
    absorption_top_fraction: float = 0.2,
    benchmark: str = BENCHMARK,
    residualize: bool = False,
) -> pd.DataFrame:
    """For every Friday with a sufficiently large, ADV-eligible, complete-
    history universe, compute the raw weekly signal bundle:
    L_t (total H1 persistence), rho_bar, absorption_ratio, index_vol
    (annualized realized vol of `benchmark` over the same window), and
    n_assets used.

    Returns a DataFrame indexed by Friday date. Weeks where the universe is
    too small (early history, before enough ETFs existed / met the ADV
    filter) are simply absent from the index -- the gate logic downstream
    treats an absent observation as "hold previous state".
    """
    fridays = weekly_fridays(panel.log_returns.index)
    rows = []
    idx = []
    for t in fridays:
        eligible = panel.eligible_on(t, min_adv=min_adv)
        eligible = [tk for tk in eligible if tk != benchmark]
        if len(eligible) < min_universe_size:
            continue
        window = panel.log_returns.loc[:t, eligible].tail(corr_window)
        window = window.dropna(axis=1, how="any")
        if window.shape[1] < min_universe_size or window.shape[0] < corr_window:
            continue

        bench_window = panel.log_returns.loc[:t, benchmark].tail(corr_window)

        topology_input = window
        if residualize:
            if bench_window.isna().any() or bench_window.shape[0] < corr_window:
                continue
            topology_input = residualize_against(window, bench_window)

        snap = compute_topology(topology_input, absorption_top_fraction=absorption_top_fraction)

        index_vol = float(bench_window.std() * np.sqrt(252)) if bench_window.notna().all() else np.nan

        rows.append(
            {
                "L": snap.total_h1_persistence,
                "n_h1_features": snap.n_h1_features,
                "rho_bar": snap.rho_bar,
                "absorption_ratio": snap.absorption_ratio,
                "index_vol": index_vol,
                "n_assets": snap.n_points,
            }
        )
        idx.append(t)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date"))


def smooth_median4(series: pd.Series, window: int = 4) -> pd.Series:
    """Trailing median over the last `window` observations (spec: 4 Fridays)."""
    return series.rolling(window, min_periods=1).median()


def expanding_percentile(series: pd.Series, min_history_years: float = 3.0, obs_per_year: int = 52) -> pd.Series:
    """Percentile rank (0-100) of each observation within its own expanding
    history (inclusive), NaN until at least `min_history_years` worth of
    observations have accumulated. This is deliberately O(T^2) (T ~ a few
    thousand weeks at most) for clarity and to avoid any risk of a
    vectorized shortcut leaking future information into the threshold.
    """
    values = series.values.astype(float)
    n = len(values)
    min_obs = int(min_history_years * obs_per_year)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(values[i]):
            continue
        if i + 1 < min_obs:
            continue
        hist = values[: i + 1]
        hist = hist[~np.isnan(hist)]
        out[i] = 100.0 * np.mean(hist <= values[i])
    return pd.Series(out, index=series.index)


def hysteresis_gate(pct: pd.Series, upper: float = 60.0, lower: float = 40.0, direction: str = "primary") -> pd.Series:
    """Boolean ON/OFF gate state with hysteresis.

    direction='primary': ON when pct > upper, OFF when pct < lower, hold
        previous state in the band [lower, upper] (spec's declared rule --
        trade when topology is genuinely multi-dimensional).
    direction='mirror': ON when pct < lower, OFF when pct > upper (spec's
        declared competing/mirror variant -- trade when topology has
        collapsed, per Khandani-Lo "reversal pays best in panic").

    A missing (NaN) observation -- universe too small/thin that week, or
    still in the expanding-window warmup period -- holds the previous
    state rather than forcing a transition.
    """
    if direction not in ("primary", "mirror"):
        raise ValueError(f"unknown direction {direction!r}")

    state = np.zeros(len(pct), dtype=bool)
    current = False  # start OFF (cash) until the gate has enough history to speak
    values = pct.values
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            if direction == "primary":
                if v > upper:
                    current = True
                elif v < lower:
                    current = False
            else:  # mirror
                if v < lower:
                    current = True
                elif v > upper:
                    current = False
        state[i] = current
    return pd.Series(state, index=pct.index)


@dataclass
class GateBundle:
    """All raw/smoothed/percentile/gate series for one correlation-window
    choice, covering the primary H1 signal and the three twin/control
    signals (rho_bar, absorption_ratio, index_vol) built with *identical*
    smoothing/percentile/hysteresis rules.
    """

    corr_window: int
    raw: pd.DataFrame  # columns: L, rho_bar, absorption_ratio, index_vol, n_assets
    smoothed: pd.DataFrame  # same columns, 4-week trailing median
    percentile: pd.DataFrame  # same columns, expanding-window percentile
    gates: dict  # {(signal_name, direction): pd.Series[bool]}


SIGNAL_COLUMNS = ["L", "rho_bar", "absorption_ratio", "index_vol"]


def build_gate_bundle(
    panel: Panel,
    corr_window: int,
    upper: float = 60.0,
    lower: float = 40.0,
    min_history_years: float = 3.0,
    **raw_kwargs,
) -> GateBundle:
    raw = compute_raw_signal_series(panel, corr_window=corr_window, **raw_kwargs)
    smoothed = pd.DataFrame({c: smooth_median4(raw[c]) for c in SIGNAL_COLUMNS}, index=raw.index)
    percentile = pd.DataFrame(
        {c: expanding_percentile(smoothed[c], min_history_years=min_history_years) for c in SIGNAL_COLUMNS},
        index=raw.index,
    )
    gates = {}
    for c in SIGNAL_COLUMNS:
        for direction in ("primary", "mirror"):
            gates[(c, direction)] = hysteresis_gate(percentile[c], upper=upper, lower=lower, direction=direction)

    return GateBundle(corr_window=corr_window, raw=raw, smoothed=smoothed, percentile=percentile, gates=gates)


def regate(bundle: GateBundle, upper: float = 60.0, lower: float = 40.0) -> GateBundle:
    """Cheaply rebuild the ON/OFF gate states for new hysteresis thresholds,
    reusing the already-computed (expensive) raw/smoothed/percentile series
    from `bundle`. Used by the parameter grid to sweep gate percentile
    thresholds without re-running persistent homology for every threshold.
    """
    gates = {}
    for c in SIGNAL_COLUMNS:
        for direction in ("primary", "mirror"):
            gates[(c, direction)] = hysteresis_gate(bundle.percentile[c], upper=upper, lower=lower, direction=direction)
    return GateBundle(corr_window=bundle.corr_window, raw=bundle.raw, smoothed=bundle.smoothed, percentile=bundle.percentile, gates=gates)
