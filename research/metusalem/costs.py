"""Flat one-way bps transaction costs (spec Sec.5 "Kostnader", Sec.13).

Deliberately NOT Smittotalet's ADV-bucketed cost model (research/smittotalet/
costs.py) -- that is a different, more elaborate cost model for a different
study; the spec here explicitly declares a flat-bps convention ("5 bp enkel
vag x omsattning ... Inga andra friktioner modelleras; deklarerad
forenkling"), so reusing the ADV-bucket machinery would silently substitute
a materially different (and untested-for-Metusalem) cost assumption. This
module is intentionally a few lines -- no "reuse before build" search is
warranted for something this simple and this explicitly specified.
"""
import pandas as pd

from . import config


def turnover_cost_returns(weights: pd.DataFrame, one_way_bps: float) -> pd.Series:
    """Per-period cost drag: sum_i |Delta w_i| * (one_way_bps / 1e4)."""
    turnover = weights.diff().abs().sum(axis=1, skipna=True)
    return turnover * (one_way_bps / 1e4)


def apply_costs(gross_returns: pd.Series, weights: pd.DataFrame, one_way_bps: float) -> pd.Series:
    cost = turnover_cost_returns(weights, one_way_bps).reindex(gross_returns.index).fillna(0.0)
    return gross_returns - cost


IS_PRIMARY_BPS = config.COST_BPS_IS_PRIMARY
IS_SENSITIVITY_BPS = config.COST_BPS_IS_SENSITIVITY
OOS_BPS = config.COST_BPS_OOS
