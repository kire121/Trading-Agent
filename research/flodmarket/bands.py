"""Steg 0b achievability bands (spec SS9: "Banden harleds ur den seedade
syntetgeneratorn... INNAN riktig data hamtas"). ONE full-scale run
(40 tickers x 5000 days, theta=0/no planted effect) of the SS12.4
generator; per-ticker std(s) and per-ticker fraction(|s|>0.95) are computed
across the 40 synthetic tickers, and their quantiles become the K0b.1/K0b.2
bands real tickers are judged against.
"""
import numpy as np
import pandas as pd

from . import config
from . import intrabar
from . import synth

N_SYNTH_ASSETS = 40
N_SYNTH_DAYS = 5000


def derive_bands(seed: int = config.GLOBAL_SEED) -> dict:
    panel = synth.simulate_panel(n_assets=N_SYNTH_ASSETS, n_days=N_SYNTH_DAYS, seed=seed)
    mi = synth.panel_to_multiindex(panel)
    shadow = intrabar.shadow_stats(mi["O"], mi["H"], mi["L"], mi["C"])

    per_ticker_std = shadow["s"].groupby(level="ticker").std(ddof=1)
    per_ticker_extreme_frac = shadow["s"].groupby(level="ticker").apply(
        lambda s: float((s.abs() > config.STEG0B_EXTREME_S_THRESHOLD).mean())
    )

    q05_std = float(per_ticker_std.quantile(0.05))
    q95_std = float(per_ticker_std.quantile(0.95))
    q99_extreme = float(per_ticker_extreme_frac.quantile(0.99))

    return {
        "seed": seed,
        "n_synth_assets": N_SYNTH_ASSETS,
        "n_synth_days": N_SYNTH_DAYS,
        "q05_std_s": q05_std,
        "q95_std_s": q95_std,
        "q99_extreme_frac": q99_extreme,
        "std_s_low_band": config.STEG0B_STD_S_LOW_MULT * q05_std,
        "std_s_high_band": config.STEG0B_STD_S_HIGH_MULT * q95_std,
        "extreme_frac_band": config.STEG0B_EXTREME_S_MULT * q99_extreme,
        "per_ticker_std_s": per_ticker_std.to_dict(),
        "per_ticker_extreme_frac": per_ticker_extreme_frac.to_dict(),
    }
