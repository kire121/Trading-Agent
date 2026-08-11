"""Flodmarket's own module (spec SS12) -- the first systematic reading of
the open/high/low columns in the EODHD panel.

API (spec SS12.1, locked):
    load_ohlc(tickers, start, end) -> DataFrame[MultiIndex(ticker,date), O,H,L,C,adjC]
    shadow_stats(O,H,L,C) -> DataFrame[U,D,b,s,flag_clamped,flag_synthopen,flag_zerorange]
    rolling_tstat(s, K, min_valid=0.8) -> Series S
    fe_demean(s, K) -> Series s_tilde

Edge cases (spec SS12.2, locked): R=0 => NaN. Clamp O,C into [L,H] + flag.
Flat bar (O=H=L=C) => NaN (subsumed by R=0). Synthetic open (SS9 0a) =>
ticker-year exclusion (handled in data_quality.py, not here -- this module
only computes the per-bar flag). K_eff < 0.8K => S=NaN => g=0 (position 0,
no error; g-computation lives in signal.py).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from lib import eodhd_client, hashutil

from . import config


class FrozenConfigMissingError(RuntimeError):
    """Raised when load_ohlc is called before config_frozen.yaml + its sha256
    exist on disk -- spec SS9 Steg0b: "pipeline-assertion vagrar hamta
    riktig data utan fryst config" (the synthetic-band derivation and the
    config freeze must happen BEFORE any real network fetch)."""


def assert_frozen_config(results_dir: str = config.RESULTS_DIR) -> str:
    """Verifies results/flodmarket/config_frozen.yaml + config_frozen.sha256
    exist and are mutually consistent; returns the config hash. Raises
    FrozenConfigMissingError otherwise. Called by load_ohlc before any
    network I/O."""
    import os
    frozen_path = os.path.join(results_dir, "config_frozen.yaml")
    hash_path = os.path.join(results_dir, "config_frozen.sha256")
    if not (os.path.exists(frozen_path) and os.path.exists(hash_path)):
        raise FrozenConfigMissingError(
            "config_frozen.yaml/.sha256 saknas -- de syntetiska banden (SS9 Steg0b) "
            "maste harledas och configen frysas INNAN riktig data hamtas. Kor "
            "synth_bands-steget forst."
        )
    with open(hash_path, "r", encoding="utf-8") as f:
        stored_hash = f.read().strip()
    with open(frozen_path, "rb") as f:
        raw = f.read()
    import yaml
    parsed = yaml.safe_load(raw)
    recomputed = hashutil.compute_config_hash(parsed)
    if recomputed != stored_hash:
        raise FrozenConfigMissingError(
            f"config_frozen.yaml stammer inte overens med sitt eget config_frozen.sha256 "
            f"(fick {recomputed}, forvantade {stored_hash}) -- fryst config ar korrupt/manipulerad."
        )
    return stored_hash


def load_ohlc(tickers: list, start: str, end: str, *, cache_dir: str = None) -> pd.DataFrame:
    """Fetches daily O,H,L,C + adjusted close for `tickers` over [start,end]
    via lib.eodhd_client, adjusting O,H,L by the adjusted_close/close ratio
    (spec SS5: "skala O,H,L med adjusted_close/close"; s itself is invariant
    to this, only the return calc consumes adjC). Asserts the frozen config
    exists (assert_frozen_config) before any network call, per SS12.1/SS9
    Steg0b ordering.

    Returns a DataFrame indexed by MultiIndex(ticker, date), columns
    O,H,L,C,adjC.
    """
    assert_frozen_config()

    cache_dir = cache_dir or config.DATA_CACHE_DIR
    frames = []
    for ticker in tickers:
        raw = eodhd_client.get_eod(ticker, start=start, end=end, exchange="US", cache_dir=cache_dir)
        if raw.empty:
            continue
        adj_factor = raw["adjusted_close"] / raw["close"]
        frame = pd.DataFrame({
            "O": raw["open"] * adj_factor,
            "H": raw["high"] * adj_factor,
            "L": raw["low"] * adj_factor,
            "C": raw["close"] * adj_factor,
            "adjC": raw["adjusted_close"],
        })
        frame.index.name = "date"
        frame["ticker"] = ticker
        frames.append(frame.reset_index().set_index(["ticker", "date"]))

    if not frames:
        return pd.DataFrame(columns=["O", "H", "L", "C", "adjC"],
                             index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]))
    return pd.concat(frames).sort_index()


def _shadow_stats_flat(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray,
                        prev_C: np.ndarray) -> dict:
    """Vectorized core (no grouping): O,H,L,C,prev_C are aligned 1-D arrays
    for a SINGLE ticker's own time series (prev_C already correctly shifted
    within that ticker)."""
    O = np.asarray(O, dtype=float)
    H = np.asarray(H, dtype=float)
    L = np.asarray(L, dtype=float)
    C = np.asarray(C, dtype=float)
    prev_C = np.asarray(prev_C, dtype=float)

    R = H - L
    zero_range = (R == 0) | ~np.isfinite(R)

    O_clamped = np.clip(O, L, H)
    C_clamped = np.clip(C, L, H)
    flag_clamped = (O_clamped != O) | (C_clamped != C)
    # Where R itself is degenerate (H<L, NaN, etc.) clip() output is meaningless;
    # those rows are already routed to NaN via zero_range below.

    with np.errstate(divide="ignore", invalid="ignore"):
        upper = np.maximum(O_clamped, C_clamped)
        lower = np.minimum(O_clamped, C_clamped)
        U = (H - upper) / R
        D = (lower - L) / R
        b = (C_clamped - O_clamped) / R
    s = D - U

    U = np.where(zero_range, np.nan, U)
    D = np.where(zero_range, np.nan, D)
    b = np.where(zero_range, np.nan, b)
    s = np.where(zero_range, np.nan, s)
    flag_clamped = np.where(zero_range, False, flag_clamped)

    with np.errstate(invalid="ignore"):
        flag_synthopen = np.round(O, 4) == np.round(prev_C, 4)
    flag_synthopen = np.where(np.isnan(prev_C), False, flag_synthopen)

    return {
        "U": U, "D": D, "b": b, "s": s,
        "flag_clamped": flag_clamped.astype(bool),
        "flag_synthopen": flag_synthopen.astype(bool),
        "flag_zerorange": zero_range.astype(bool),
    }


def shadow_stats(O: pd.Series, H: pd.Series, L: pd.Series, C: pd.Series) -> pd.DataFrame:
    """Per-bar shadow statistics (spec SS2.1):
        R = H - L
        U = (H - max(O,C)) / R
        D = (min(O,C) - L) / R
        b = (C - O) / R
        s = D - U
    Edge cases per SS12.2: R=0 => NaN (no exception). O,C clamped into
    [L,H] before the ratios are computed, flagged in flag_clamped.
    flag_synthopen: O == previous bar's C (4 decimals) -- if the inputs
    carry a MultiIndex with a 'ticker' level, the previous close is taken
    WITHIN each ticker (never across a ticker boundary); otherwise a plain
    single-series shift(1) is used (single-ticker / unit-test case, spec
    SS12.3's individual bar examples have no defined previous bar and get
    flag_synthopen=False).
    """
    idx = O.index
    if isinstance(idx, pd.MultiIndex) and "ticker" in (idx.names or []):
        prev_C = C.groupby(level="ticker").shift(1)
    else:
        prev_C = C.shift(1)

    out = _shadow_stats_flat(O.to_numpy(), H.to_numpy(), L.to_numpy(), C.to_numpy(),
                              prev_C.to_numpy())
    return pd.DataFrame(out, index=idx)


def rolling_tstat(s: pd.Series, K: int, min_valid: float = 0.8) -> pd.Series:
    """S_i(t) = mean(s_tilde) / (std(s_tilde)/sqrt(K_eff)) over the trailing
    K-day window ending at t (spec SS2.2). K_eff = count of non-NaN values
    in that window; requires K_eff >= min_valid*K, else NaN (=> g=0
    downstream, never an error). A degenerate window (std=0, e.g. all-equal
    values) is also NaN (undefined t-stat), same downstream treatment.

    Operates per-ticker (groupby 'ticker' level) if `s` carries a
    MultiIndex, so no window ever spans a ticker boundary.
    """
    def _one(series: pd.Series) -> pd.Series:
        r = series.rolling(K, min_periods=1)
        k_eff = r.count()
        mean_ = r.mean()
        std_ = r.std(ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            S = mean_ / (std_ / np.sqrt(k_eff))
        S = S.where(k_eff >= min_valid * K)
        S = S.where(std_ > 0)
        return S

    if isinstance(s.index, pd.MultiIndex) and "ticker" in (s.index.names or []):
        return s.groupby(level="ticker", group_keys=False).apply(_one)
    return _one(s)


def fe_demean(s: pd.Series, K: int, window: int = config.FE_DEMEAN_WINDOW) -> pd.Series:
    """s_tilde_tau = s_tau - m_i(t); m_i(t) = mean of s over the window
    [t-K-(window-1), t-K] (spec SS2.2: window disjoint from the trailing
    K-window [t-K+1, t] used by rolling_tstat). PIT-lagged: m_i(t) never
    uses s values from strictly after t-K.

    DECLARED (AVVIKELSER.md): the spec pins the window as exactly `window`
    (252) observations but does not state a minimum-valid-observations rule
    for it (unlike the K-window, which has an explicit K_eff>=0.8K rule).
    Least-favorable-to-the-strategy choice: require the FULL window (no
    partial-window demean), since a partial/early/noisy demean estimate
    could spuriously inflate apparent signal quality -- this simply delays
    the first available date, it does not alter any date where a full
    window exists.

    Operates per-ticker if `s` carries a MultiIndex.
    """
    def _one(series: pd.Series) -> pd.Series:
        m = series.shift(K).rolling(window, min_periods=window).mean()
        return series - m

    if isinstance(s.index, pd.MultiIndex) and "ticker" in (s.index.names or []):
        return s.groupby(level="ticker", group_keys=False).apply(_one)
    return _one(s)
