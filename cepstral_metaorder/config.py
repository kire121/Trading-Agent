"""
Spec constants for the cepstral metaorder-slicing signal.

Every threshold here is transcribed directly from the research hypothesis
(institutional metaorders sliced via scheduled TWAP-style child orders leave
a weak periodicity in intraday volume, detectable in the cepstrum). Nothing
in this file is fit to data -- it is pre-registered before any backtest runs.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class UniverseConfig:
    min_price: float = 5.0
    min_adv_usd: float = 25_000_000.0
    adv_window_days: int = 63
    top_n: int = 1000


@dataclass(frozen=True)
class SessionConfig:
    """Regular trading hours handling (US equities, America/New_York wall clock)."""
    rth_open: str = "09:30"
    rth_close: str = "16:00"
    exclude_open_minutes: int = 15
    exclude_close_minutes: int = 15
    tz: str = "America/New_York"


@dataclass(frozen=True)
class CepstrumConfig:
    detrend_window_days: int = 21
    quefrency_min_min: int = 2
    quefrency_max_min: int = 45
    epsilon: float = 1e-8
    slicing_score_avg_days: int = 5
    direction_window_days: int = 5
    # comb-filter (cepstral liftering) width around tau*, in minutes, used to
    # isolate the periodic component before reconstructing the burst mask.
    lifter_half_width_min: int = 1
    # burst mask threshold: top quantile of the reconstructed periodic
    # envelope counts as "inside a scheduled child-order burst".
    burst_quantile: float = 0.85


@dataclass(frozen=True)
class EntryExitConfig:
    entry_score_pctile: float = 80.0
    entry_min_abs_direction: float = 0.1
    exit_score_pctile: float = 50.0
    max_holding_days: int = 15


@dataclass(frozen=True)
class SizingConfig:
    max_weight_per_name: float = 0.02
    target_gross: float = 1.00
    target_net: float = 0.0  # 50/50 long/short -> dollar-neutral by construction
    vol_target_annual: float = 0.10
    vol_lookback_days: int = 63
    band_buffer_frac: float = 0.30  # only trade if |w - w_target| > 0.30 * |w_target|
    trading_days_per_year: int = 252


@dataclass(frozen=True)
class CostConfig:
    commission_bps_per_side: float = 7.0
    # measured half-spread is added on top per-symbol/day when available;
    # this is the floor used when no measured spread is available.
    fallback_half_spread_bps: float = 5.0


@dataclass(frozen=True)
class ValidationConfig:
    # Step 0: existence + breadth
    null_permutations: int = 200
    null_exceedance_multiple: float = 3.0  # must be >= 3x nominal 5% = 15%
    null_nominal_alpha: float = 0.05
    pc1_share_max: float = 0.40
    min_expected_positions: int = 50
    # Step 1: redundancy screen
    min_abs_newey_west_t: float = 2.0
    newey_west_lags: int = 5  # ~ holding-period-scale autocorrelation
    # Step 2: costs / robustness
    tau_window_grid: Tuple[Tuple[int, int], ...] = ((2, 30), (2, 45), (5, 60))
    holding_period_grid: Tuple[int, ...] = (10, 15, 20)
    regime_break_year: int = 2016
    # rejection thresholds. Spec: "sign consistency missing in >=2 of 4 IS
    # subperiods" => reject. Equivalently: pass requires consistency in >=3
    # of 4, i.e. at most 1 may be missing it.
    n_is_subperiods: int = 4
    max_missing_sign_subperiods: int = 1
    n_strategy_variants_tested: int = 9  # 3 tau windows x 3 holding periods, for DSR


@dataclass(frozen=True)
class DiversificationConfig:
    max_abs_beta_spy: float = 0.15
    max_abs_corr_tsmom: float = 0.25


@dataclass(frozen=True)
class SpecConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    cepstrum: CepstrumConfig = field(default_factory=CepstrumConfig)
    entry_exit: EntryExitConfig = field(default_factory=EntryExitConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    diversification: DiversificationConfig = field(default_factory=DiversificationConfig)


SPEC = SpecConfig()
