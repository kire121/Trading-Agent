"""Transaction cost model: 3bp/side commission-equivalent + half-spread
bucketed by trailing dollar ADV. Ported verbatim (bucket thresholds + bps
schedule) from research/formdriften/costs.py via research/dammluckan/costs.py
-- the one established ADV-bucketed cost-model precedent in this repo's
research-branch series.

DOCUMENTED SPREAD CAVEAT (restated per house convention): EODHD does not
provide quoted bid/ask spread history, so the half-spread leg below is
approximated from trailing dollar-ADV via a monotone liquidity bucket -- a
documented approximation, not a measured spread. Only the commission leg
(3bp/side) is a clean, un-approximated assumption.
"""
import numpy as np

from research.omori import config


def half_spread_bps(adv_usd):
    """Vectorized: half-spread in bps for a scalar or array-like of trailing
    dollar-ADV values, via the monotone liquidity-bucket schedule."""
    adv = np.asarray(adv_usd, dtype=float)
    out = np.full(adv.shape, np.nan)
    for min_adv, bps in sorted(config.ADV_BUCKETS, key=lambda t: -t[0]):
        mask = np.isnan(out) & (adv >= min_adv)
        out[mask] = bps
    # ADV below every bucket's min (only the (0.0, x) bucket can catch that,
    # which always fires) or NaN ADV (insufficient history) -> worst bucket.
    out[np.isnan(out)] = config.ADV_BUCKETS[-1][1]
    return out


def round_trip_cost_bps(adv_usd):
    """One-way commission + one-way half-spread, applied on both entry and
    exit legs -- i.e. this returns the PER-LEG cost in bps; callers apply it
    once at entry and once at exit."""
    return config.COMMISSION_BPS + half_spread_bps(adv_usd)


def trade_cost_return(adv_usd_at_entry):
    """Per-leg cost expressed as a (positive) return drag, i.e. divide bps
    by 1e4."""
    return round_trip_cost_bps(adv_usd_at_entry) / 1e4
