"""
Fasflocken (PH-1) -- configuration.

Central place for every tunable in the strategy spec: the sector/ETF
universe, default signal parameters, the declared grid, cost model,
vol-targeting constraints and the IS/OOS date boundaries.

Nothing here touches data or math; it is pure declaration so that
signals.py / portfolio.py / backtest.py / grid_search.py all read the
same numbers instead of re-declaring them.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Universe: the 11 GICS sectors traded via Select Sector SPDRs.
#
# Inception dates matter for point-in-time discipline: XLRE and XLC did not
# exist for most of the backtest window, and the Sept-2018 GICS
# reclassification moved constituents (notably telecom -> communication
# services) without changing the S&P 500 membership itself. Both are
# handled in universe.py; the dates below are what gate an ETF's eligibility
# in the ranking (see portfolio.eligible_sectors).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectorETF:
    gics_sector: str
    etf: str
    inception: _dt.date


SECTOR_ETFS: tuple[SectorETF, ...] = (
    SectorETF("Technology", "XLK", _dt.date(1998, 12, 22)),
    SectorETF("Financials", "XLF", _dt.date(1998, 12, 22)),
    SectorETF("Energy", "XLE", _dt.date(1998, 12, 22)),
    SectorETF("Health Care", "XLV", _dt.date(1998, 12, 22)),
    SectorETF("Industrials", "XLI", _dt.date(1998, 12, 22)),
    SectorETF("Consumer Discretionary", "XLY", _dt.date(1998, 12, 22)),
    SectorETF("Consumer Staples", "XLP", _dt.date(1998, 12, 22)),
    SectorETF("Utilities", "XLU", _dt.date(1998, 12, 22)),
    SectorETF("Materials", "XLB", _dt.date(1998, 12, 22)),
    # Real Estate carved out of Financials in the Sept-2016 GICS revision;
    # XLRE launched ahead of that in Oct-2015 to track the new sector.
    SectorETF("Real Estate", "XLRE", _dt.date(2015, 10, 7)),
    # Communication Services replaced Telecom in the Sept-2018 GICS
    # revision (absorbing parts of Tech and Consumer Discretionary); XLC
    # launched alongside it.
    SectorETF("Communication Services", "XLC", _dt.date(2018, 6, 19)),
)

GICS_SECTORS: tuple[str, ...] = tuple(s.gics_sector for s in SECTOR_ETFS)
SECTOR_TO_ETF: dict[str, str] = {s.gics_sector: s.etf for s in SECTOR_ETFS}
ETF_TO_SECTOR: dict[str, str] = {s.etf: s.gics_sector for s in SECTOR_ETFS}
SECTOR_INCEPTION: dict[str, _dt.date] = {s.gics_sector: s.inception for s in SECTOR_ETFS}

# The date GICS moved Telecom -> Communication Services and reshuffled
# constituents. Any point-in-time GICS lookup must branch on this date
# rather than trust a single static mapping.
GICS_RECLASSIFICATION_DATE = _dt.date(2018, 9, 24)


# ---------------------------------------------------------------------------
# Default signal parameters (one grid cell -- the spec's "declared" default).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalParams:
    band_low_days: int = 5          # short edge of the cycle-length band
    band_high_days: int = 20        # long edge of the cycle-length band
    filter_order: int = 2           # Butterworth order passed to scipy.signal.butter
    analytic_window: int = 90       # rolling Hilbert window, trading days
    z_lookback_weeks: int = 104     # trailing weekly history for Z_s
    n_legs: int = 3                 # long legs == short legs
    hysteresis_band: int = 4        # "top/bottom 4" retention band
    min_constituents: int = 5       # sector must have >= this many names for a valid R_s


DEFAULT_PARAMS = SignalParams()

# Declared grid (Section "Deklarerad grid"). Every cell must be pushed
# through the same DSR correction -- see grid_search.py.
BAND_GRID: tuple[tuple[int, int], ...] = ((3, 15), (5, 20), (10, 40))
WINDOW_GRID: tuple[int, ...] = (60, 90, 120)
Z_LOOKBACK_GRID_WEEKS: tuple[int, ...] = (52, 104, 156)
LEGS_GRID: tuple[int, ...] = (2, 3, 4)


# ---------------------------------------------------------------------------
# Costs, vol targeting, gross constraint.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    commission_bps_per_side: float = 2.0
    # Real half-spread should come from quoted data; this is a documented
    # placeholder (sector SPDRs typically trade ~1bp wide) used only when
    # no per-ETF spread series is supplied to the backtest.
    default_half_spread_bps: float = 0.5

    def total_bps_per_side(self, half_spread_bps: float | None = None) -> float:
        hs = self.default_half_spread_bps if half_spread_bps is None else half_spread_bps
        return self.commission_bps_per_side + hs


DEFAULT_COSTS = CostModel()

VOL_TARGET_ANN: float = 0.08
MAX_GROSS: float = 2.0
COV_WINDOW_DAYS: int = 60
TRADING_DAYS_PER_YEAR: int = 252
WEEKS_PER_YEAR: int = 52

# Execution calendar: signal computed at Friday close, executed at the
# following Monday open, held one week, no intra-week adjustments.
SIGNAL_WEEKDAY: int = 4   # Monday=0 ... Friday=4
EXEC_WEEKDAY: int = 0     # Monday


# ---------------------------------------------------------------------------
# Null-hypothesis baseline parameters.
# ---------------------------------------------------------------------------

BOOTSTRAP_BLOCK_WEEKS: int = 13
BOOTSTRAP_N_DRAWS: int = 1000
OVERLAP_CORR_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Sample window.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleWindow:
    burn_in_start: _dt.date = _dt.date(2002, 1, 1)
    is_start: _dt.date = _dt.date(2004, 1, 1)
    is_end: _dt.date = _dt.date(2017, 12, 31)
    oos_start: _dt.date = _dt.date(2018, 1, 1)
    oos_end: _dt.date = _dt.date(2026, 12, 31)

    # Sub-periods used for the sign-instability rejection check.
    sign_check_periods: tuple[tuple[_dt.date, _dt.date], ...] = field(
        default_factory=lambda: (
            (_dt.date(2004, 1, 1), _dt.date(2010, 12, 31)),
            (_dt.date(2011, 1, 1), _dt.date(2017, 12, 31)),
            (_dt.date(2018, 1, 1), _dt.date(2026, 12, 31)),
        )
    )


SAMPLE_WINDOW = SampleWindow()
