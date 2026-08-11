"""Steg 0b real-data checks (spec SS9 K0b.1-K0b.3), evaluated against the
synthetic achievability bands (bands.py) derived BEFORE this real data was
touched."""
import numpy as np

from . import config
from . import intrabar
from . import signal


def run_steg0b_real(shadow, tradeable_tickers: list, bands: dict) -> dict:
    s = shadow["s"].loc[shadow.index.get_level_values("ticker").isin(tradeable_tickers)]

    per_ticker_std = s.groupby(level="ticker").std(ddof=1)
    per_ticker_extreme_frac = s.groupby(level="ticker").apply(
        lambda x: float((x.abs() > config.STEG0B_EXTREME_S_THRESHOLD).mean())
    )

    low, high = bands["std_s_low_band"], bands["std_s_high_band"]
    k0b1_excluded = sorted(per_ticker_std[(per_ticker_std < low) | (per_ticker_std > high)].index)

    ext_band = bands["extreme_frac_band"]
    k0b2_excluded = sorted(per_ticker_extreme_frac[per_ticker_extreme_frac > ext_band].index)

    remaining_tickers = sorted(set(tradeable_tickers) - set(k0b1_excluded) - set(k0b2_excluded))
    n_remaining = len(remaining_tickers)
    floor_kill = n_remaining < config.STEG0A_MIN_TRADEABLE_TICKERS

    # K0b.3: NaN fraction in S (primary cell: K=40, FE-demean=252d -- the
    # canonical S per spec SS2.2's own definition in terms of s_tilde),
    # computed over the REMAINING (post K0b.1/K0b.2) tickers' asset-weeks.
    s_remaining = shadow["s"].loc[shadow.index.get_level_values("ticker").isin(remaining_tickers)]
    s_tilde = intrabar.fe_demean(s_remaining, config.K_PRIMARY, window=config.FE_DEMEAN_WINDOW)
    S = intrabar.rolling_tstat(s_tilde, config.K_PRIMARY, min_valid=config.K_EFF_MIN_FRACTION)

    decision_dates = signal.weekly_decision_dates(
        shadow.index.get_level_values("date").unique().sort_values())
    S_wide = S.unstack("ticker")
    S_at_decisions = S_wide.reindex(decision_dates.intersection(S_wide.index))
    total_asset_weeks = S_at_decisions.size
    nan_asset_weeks = int(S_at_decisions.isna().sum().sum())
    nan_fraction = nan_asset_weeks / total_asset_weeks if total_asset_weeks else float("nan")

    k0b3_kill = nan_fraction > config.STEG0B_NAN_S_KILL_FRACTION
    k0b3_flag = nan_fraction > config.STEG0B_NAN_S_FLAG_FRACTION

    kill = bool(floor_kill or k0b3_kill)

    return {
        "per_ticker_std_s": per_ticker_std.to_dict(),
        "per_ticker_extreme_frac": per_ticker_extreme_frac.to_dict(),
        "std_s_band": [low, high],
        "extreme_frac_band": ext_band,
        "k0b1_excluded_tickers": k0b1_excluded,
        "k0b2_excluded_tickers": k0b2_excluded,
        "remaining_tickers": remaining_tickers,
        "n_remaining_tickers": n_remaining,
        "floor_kill": bool(floor_kill),
        "nan_fraction_in_S": nan_fraction,
        "k0b3_kill": bool(k0b3_kill),
        "k0b3_flag": bool(k0b3_flag),
        "kill": kill,
    }
