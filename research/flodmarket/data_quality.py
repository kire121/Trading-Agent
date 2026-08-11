"""Steg 0a -- O/H/L data-quality gate (spec SS9, first systematic reading
of these columns). All checks operate PER TICKER-YEAR (or per ticker over
the full IS window, for the zero-range check) on the real EODHD panel.
"""
import numpy as np
import pandas as pd

from . import config


def _year_of(idx: pd.DatetimeIndex) -> pd.Index:
    return idx.year


def per_ticker_year_preclamp_and_missing(shadow: pd.DataFrame, raw: pd.DataFrame,
                                          all_trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (ticker, year): pre-clamp error fraction (flag_clamped -- the bar
    violated H>=max(O,C) or L<=min(O,C) before clamping) and missing-bar
    fraction (expected trading days, proxied by the UNION of trading dates
    observed across the whole panel, that are missing from this ticker's
    own observed range)."""
    rows = []
    tickers = shadow.index.get_level_values("ticker").unique()
    for ticker in tickers:
        t_shadow = shadow.xs(ticker, level="ticker")
        years = sorted(set(_year_of(t_shadow.index)))
        first_date, last_date = t_shadow.index.min(), t_shadow.index.max()
        expected_dates = all_trading_dates[(all_trading_dates >= first_date) & (all_trading_dates <= last_date)]
        observed_dates = set(t_shadow.index)
        for year in years:
            year_mask = _year_of(t_shadow.index) == year
            n_bars = int(year_mask.sum())
            preclamp_errors = int(t_shadow.loc[year_mask, "flag_clamped"].sum())
            expected_in_year = expected_dates[expected_dates.year == year]
            missing = sum(1 for d in expected_in_year if d not in observed_dates)
            denom = max(len(expected_in_year), 1)
            rows.append({
                "ticker": ticker, "year": int(year), "n_bars": n_bars,
                "preclamp_error_fraction": preclamp_errors / max(n_bars, 1),
                "missing_bar_fraction": missing / denom,
            })
    return pd.DataFrame(rows)


def per_ticker_year_synthetic_open(shadow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tickers = shadow.index.get_level_values("ticker").unique()
    for ticker in tickers:
        t_shadow = shadow.xs(ticker, level="ticker")
        years = sorted(set(_year_of(t_shadow.index)))
        for year in years:
            year_mask = _year_of(t_shadow.index) == year
            n_bars = int(year_mask.sum())
            n_synth = int(t_shadow.loc[year_mask, "flag_synthopen"].sum())
            rows.append({"ticker": ticker, "year": int(year), "n_bars": n_bars,
                         "synthetic_open_fraction": n_synth / max(n_bars, 1)})
    return pd.DataFrame(rows)


def per_ticker_zero_range_is(shadow: pd.DataFrame) -> pd.Series:
    return shadow["flag_zerorange"].groupby(level="ticker").mean()


def run_steg0a(shadow: pd.DataFrame, raw: pd.DataFrame) -> dict:
    all_trading_dates = pd.DatetimeIndex(sorted(shadow.index.get_level_values("date").unique()))

    py = per_ticker_year_preclamp_and_missing(shadow, raw, all_trading_dates)
    synth_open = per_ticker_year_synthetic_open(shadow)
    zero_range = per_ticker_zero_range_is(shadow)

    preclamp_fail_years = py[py["preclamp_error_fraction"] > config.STEG0A_PRECLAMP_ERROR_MAX_FRACTION]
    missing_fail_years = py[py["missing_bar_fraction"] > config.STEG0A_MISSING_BARS_MAX_FRACTION]
    excluded_ticker_years = set(
        tuple(x) for x in synth_open[synth_open["synthetic_open_fraction"]
                                      > config.STEG0A_SYNTHETIC_OPEN_MAX_FRACTION][["ticker", "year"]].to_numpy()
    )
    excluded_tickers_zero_range = set(zero_range[zero_range > config.STEG0A_ZERO_RANGE_MAX_FRACTION_IS].index)

    all_tickers = set(shadow.index.get_level_values("ticker").unique())
    # A ticker is fully excluded (not just a single ticker-year) if EITHER
    # it has >10% zero-range days over IS, OR it has no remaining valid
    # ticker-years after excluding synthetic-open years (every year for
    # that ticker was excluded).
    tickers_with_any_valid_year = set(synth_open["ticker"].unique()) - {
        t for t in synth_open["ticker"].unique()
        if all((t, y) in excluded_ticker_years for y in synth_open.loc[synth_open["ticker"] == t, "year"])
    }
    tradeable_tickers = sorted((all_tickers & tickers_with_any_valid_year) - excluded_tickers_zero_range)

    n_tradeable = len(tradeable_tickers)
    kill = n_tradeable < config.STEG0A_MIN_TRADEABLE_TICKERS

    return {
        "n_tickers_total": len(all_tickers),
        "n_tradeable_tickers": n_tradeable,
        "tradeable_tickers": tradeable_tickers,
        "excluded_tickers_zero_range": sorted(excluded_tickers_zero_range),
        "excluded_ticker_years_synthetic_open": sorted(excluded_ticker_years),
        "ticker_years_preclamp_fail": preclamp_fail_years[["ticker", "year", "preclamp_error_fraction"]].to_dict(
            orient="records"),
        "ticker_years_missing_bars_fail": missing_fail_years[["ticker", "year", "missing_bar_fraction"]].to_dict(
            orient="records"),
        "min_required_tickers": config.STEG0A_MIN_TRADEABLE_TICKERS,
        "kill": bool(kill),
        "per_ticker_year_quality": py.to_dict(orient="records"),
        "per_ticker_year_synthetic_open": synth_open.to_dict(orient="records"),
        "per_ticker_zero_range_fraction": zero_range.to_dict(),
    }
