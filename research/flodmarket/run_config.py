"""Canonical run configuration dict -- the object that gets frozen
(config_frozen.yaml + .sha256, docs/INSTRUKTION.md SS3/SS5) and hashed
(config_hash) throughout the delivery. Kept separate from config.py's
Python constants so the frozen artifact is a plain, yaml/json-serializable
dict (lib.hashutil.compute_config_hash requires JSON-serializable values).
"""
from . import config


def build_run_config() -> dict:
    return {
        "strategy_name": "flodmarket",
        "seed": config.GLOBAL_SEED,
        "selection_reading": config.SELECTION_READING,
        "n_effective_surface_reads": config.N_EFFECTIVE_SURFACE_READS,
        "is_universe": sorted(config.IS_UNIVERSE),
        "is_universe_sha256_prefix": config.UNIVERSE_SHA256_PREFIX,
        "oos_universe_sha256_prefix": config.OOS_UNIVERSE_SHA256_PREFIX,
        "is_start": config.IS_START,
        "is_end": config.IS_END,
        "data_end": config.IS_END,  # never advanced past is_end in this session (rule 3)
        "k_primary": config.K_PRIMARY,
        "z_star_primary": config.Z_STAR_PRIMARY,
        "fe_demean_window": config.FE_DEMEAN_WINDOW,
        "k_eff_min_fraction": config.K_EFF_MIN_FRACTION,
        "signal_vol_lookback": config.SIGNAL_VOL_LOOKBACK,
        "portfolio_vol_target": config.PORTFOLIO_VOL_TARGET,
        "gross_cap": config.GROSS_CAP,
        "commission_bps": config.COMMISSION_BPS,
        "adv_cost_lookback": config.ADV_COST_LOOKBACK,
        "k_grid": list(config.K_GRID),
        "z_star_grid": list(config.Z_STAR_GRID),
        "demean_grid": [d if d is not None else "none" for d in config.DEMEAN_GRID],
        "t3_n_draws": config.T3_N_DRAWS,
        "t3_block_days": config.T3_BLOCK_DAYS,
        "t5_simplicity_tolerance": config.T5_SIMPLICITY_TOLERANCE,
    }
