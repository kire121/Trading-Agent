"""
Dammluckan -- shared configuration and declared parameters.

Every constant that is DECLARED (i.e. not literally pinned by the brief) is
called out as such in a comment, following the house convention used by the
sibling research branches (Vridmomentet, Vindkastet, Formdriften): nothing
here is silently assumed.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Universe (reused verbatim from research/vindkastet/fetch_data.py)
# ---------------------------------------------------------------------------
PRIMARY = ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD",
           "SLV", "DBC", "USO", "UUP", "FXE", "FXY", "VNQ"]
SECONDARY = ["EWJ", "EWG", "EWU", "EWQ", "EWI", "EWP", "EWL", "EWA", "EWC",
             "EWY", "EWT", "EWZ", "EWW", "EWS", "EWH", "EWD"]

# ---------------------------------------------------------------------------
# Sample split (IS/OOS-1 boundary pinned by the brief; OOS-2 is the secondary
# universe, touched exactly once, at the very end of the protocol).
# ---------------------------------------------------------------------------
IS_START = "2004-01-01"
IS_END = "2017-12-31"
OOS1_START = "2018-01-01"
OOS1_END = "2026-08-07"          # last EODHD bar actually fetched
HISTORY_START = "2003-01-01"     # DECLARED: warm-up buffer before IS_START so
                                  # the n=160 rolling window has a full look-
                                  # back available on the very first IS date.

# ---------------------------------------------------------------------------
# Signal parameters (primary cell; grid varies n, theta_pctl, h around this)
# ---------------------------------------------------------------------------
N_DEFAULT = 120
THETA_PCTL_DEFAULT = 80
H_DEFAULT = 10

N_GRID = (80, 120, 160)
THETA_PCTL_GRID = (70, 80, 90)
H_GRID = (5, 10, 15)

# DECLARED: sigma-hat used only to normalize the occupation-band width is a
# trailing realized-vol estimate of daily log returns (raw close), matching
# the vol60 convention used throughout the sibling branches.
VOL_LOOKBACK = 60
VOL_MIN_PERIODS = 45

# DECLARED: band-width constant c is calibrated (once, on the primary
# universe, IS window only) so that the block-bootstrap null median of O+
# (and, by symmetry of the null construction, O-) is approx. 0.15, per the
# brief's calibration instruction. See signal.calibrate_band_constant().
# The frozen value is filled in by calibrate_band_constant() and cached to
# output/band_constant.json so OOS runs reuse the exact IS-frozen value.
C_BAND_TARGET_MEDIAN = 0.15
BAND_HORIZON_DAYS = 5             # the "sqrt(5)" horizon scaling in the brief

# ---------------------------------------------------------------------------
# Block-bootstrap null (used for theta_i calibration, c calibration, Steg-1
# IC null test, and the portfolio-level null-twin #3).
# ---------------------------------------------------------------------------
BLOCK_LENGTH = 20                 # DECLARED: ~1 trading month, matches the
                                   # block length formdriften's nulls.py used
                                   # for its own estimator null.
N_BLOCK_DRAWS = 500               # DECLARED: null-distribution resamples
N_THETA_CALIB_DRAWS = 1000        # DECLARED: per-asset draws for theta_i / c

# ---------------------------------------------------------------------------
# Portfolio / sizing
# ---------------------------------------------------------------------------
MAX_CONCURRENT = 12
GROSS_CAP = 2.0                   # 200%
PORTFOLIO_VOL_TARGET = 0.08       # 8% annualized ex-ante
SIZING_VOL_LOOKBACK = 20          # DECLARED: daily realized-vol lookback used
                                   # for sigma_hat_i in w_i = vol_target/sigma_hat_i
TRADING_DAYS_YEAR = 252

# ---------------------------------------------------------------------------
# Costs -- ADV-bucket model, ported verbatim from research/formdriften/costs.py
# (the only ADV-bucketed cost model anywhere in the research-branch series;
# "repots ADV-bucketmodell" in the brief refers to this one).
# ---------------------------------------------------------------------------
COMMISSION_BPS = 3.0
ADV_BUCKETS = (   # (min ADV $, half-spread bps), most liquid first
    (500e6, 0.5),
    (200e6, 1.0),
    (100e6, 1.5),
    (50e6, 2.5),
    (20e6, 4.0),
    (0.0, 7.0),
)
ADV_COST_LOOKBACK = 63
# DOCUMENTED SPREAD CAVEAT (verbatim convention from Formdriften): EODHD does
# not provide quoted bid/ask spread history, so the half-spread leg is
# approximated from trailing dollar-ADV via a monotone liquidity bucket, not
# measured from actual NBBO data. Only the commission leg (3bp/side) is a
# clean, un-approximated assumption.

# ---------------------------------------------------------------------------
# Steg 0 / Steg 1 gates
# ---------------------------------------------------------------------------
MIN_EVENTS_PER_SIDE = 300
MAX_SINGLE_ASSET_EVENT_SHARE = 0.5
IC_P_KILL = 0.05
IC_ABS_KILL = 0.03

# ---------------------------------------------------------------------------
# Kill criteria (pre-registered, frozen before OOS-1/OOS-2 are inspected)
# ---------------------------------------------------------------------------
DSR_OOS_KILL = 0.0
TSMOM_CORR_KILL = 0.4
SIGN_INCONSISTENCY_SUBPERIODS_KILL = 2      # of 4
NET_EXCESS_OVER_DONCHIAN_TWIN_KILL = 0.0

N_SUBPERIODS = 4   # brief: "teckeninkonsistens i >= 2 av 4 delperioder" --
                    # DECLARED: the full IS+OOS-1 sample (2004-2026) is cut
                    # into 4 equal-length calendar sub-periods; see
                    # stats.sub_period_boundaries().


@dataclass(frozen=True)
class SignalParams:
    n: int = N_DEFAULT
    theta_pctl: float = THETA_PCTL_DEFAULT
    h: int = H_DEFAULT
    c_band: float = None   # filled in after calibration


@dataclass(frozen=True)
class PortfolioParams:
    max_concurrent: int = MAX_CONCURRENT
    gross_cap: float = GROSS_CAP
    vol_target: float = PORTFOLIO_VOL_TARGET
    sizing_vol_lookback: int = SIZING_VOL_LOOKBACK
