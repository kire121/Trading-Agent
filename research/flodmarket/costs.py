"""ADV-bucket transaction cost model.

Proveniens: research/smittotalet/costs.py, branch
claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b (itself ported
verbatim from research/dammluckan/costs.py <- research/formdriften/costs.py).
Reused per spec SS4: "ADV-bucket-modellen om den finns i Trading-Agent (grep
adv|cost -- Timglaset-larndomen: verifiera 'repo-standard' med grep)" -- it
was found (see AVVIKELSER.md), so the flat 5bp/12bp fallback is NOT used
for the IS US40 leg. Ported near-verbatim, only the config import changed.
"""
import numpy as np
import pandas as pd

from . import config

_ADV_BUCKETS = config.ADV_BUCKETS  # (min ADV $, half-spread bps), most liquid first


def half_spread_bps(adv_dollars):
    """Vectorized monotone ADV-bucket lookup. Accepts scalar, ndarray, or Series."""
    arr = np.asarray(adv_dollars, dtype=float)
    out = np.full(arr.shape, _ADV_BUCKETS[-1][1], dtype=float)
    for min_adv, bps in sorted(_ADV_BUCKETS, key=lambda x: x[0]):
        out = np.where(arr >= min_adv, bps, out)
    if np.isscalar(adv_dollars):
        return float(out)
    if isinstance(adv_dollars, pd.Series):
        return pd.Series(out, index=adv_dollars.index)
    if isinstance(adv_dollars, pd.DataFrame):
        return pd.DataFrame(out, index=adv_dollars.index, columns=adv_dollars.columns)
    return out


def round_trip_cost_bps(adv_dollars):
    return config.COMMISSION_BPS + half_spread_bps(adv_dollars)


def cost_fraction(adv_dollars):
    """Round-trip cost as a fraction of notional (not bps). NaN ADV -> illiquid default."""
    return round_trip_cost_bps(adv_dollars) / 10_000.0


def flat_cost_fraction(is_ucits: bool = False) -> float:
    """Fallback flat cost (spec SS4 caveat), used ONLY if the ADV model
    cannot be computed for a given leg (e.g. thin ADV history on a UCITS
    ETF once OOS is unlocked) -- not used on the IS US40 leg."""
    bps = config.UCITS_FLAT_COST_BPS if is_ucits else config.US_FLAT_COST_BPS
    return bps / 10_000.0
