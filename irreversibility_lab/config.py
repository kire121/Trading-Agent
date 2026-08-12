"""Configuration for the irreversibility regime-switching lab."""

from dataclasses import dataclass, field

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "DBC", "UUP", "HYG"]

START_DATE = "2000-01-01"
# End date is resolved to "today" at run time by the pipeline, not hardcoded here.

IS_START = "2000-01-01"
IS_END = "2018-12-31"
OOS_START = "2019-01-01"
# OOS_END = today (locked forward, no re-optimization after this date)

SUBPERIODS = {
    "2000-07": ("2000-01-01", "2007-12-31"),
    "2008-12": ("2008-01-01", "2012-12-31"),
    "2013-18": ("2013-01-01", "2018-12-31"),
}

# --- Signal ---
W_DEFAULT = 250            # HVG / estimator window (trading days)
W_GRID = [125, 250, 500]
Z_HISTORY = 750             # rolling history for z-scoring I_t
Z_THRESHOLD_DEFAULT = 0.5
Z_THRESHOLD_GRID = [0.25, 0.5, 1.0]

TREND_LOOKBACK = 63
MEANREV_LOOKBACK = 5

# --- Sizing ---
VOL_TARGET_ANNUAL = 0.10
N_INSTRUMENTS = 12
PER_INSTRUMENT_CAP = 0.20
GROSS_CAP = 1.50
VOL_LOOKBACK = 60
ANNUALIZATION = 252

# --- Costs ---
# 2bp/side commission+slippage assumption (per spec) plus an assumed half
# bid/ask spread per instrument, in decimal (one-way). These are documented
# ballpark liquidity assumptions since historical NBBO spread data is not
# available in this environment -- treat cost results as approximate and
# re-validate against real spread data before sizing real capital.
COMMISSION_BP_PER_SIDE = 2.0
HALF_SPREAD_BP = {
    "SPY": 0.5, "QQQ": 0.5, "IWM": 1.0, "EFA": 2.0, "EEM": 2.0,
    "TLT": 1.0, "IEF": 1.0, "GLD": 1.0, "SLV": 2.0, "DBC": 3.0,
    "UUP": 3.0, "HYG": 2.0,
}

# --- KL smoothing ---
KL_SMOOTHING = 0.5

# --- Ordinal irreversibility ---
ORDINAL_PATTERN_LENGTH = 3

# --- Third moment statistic ---
PSI_TAU = 1

# --- Bootstrap null ---
BOOTSTRAP_BLOCK_LEN = 20
BOOTSTRAP_RUNS = 1000

# --- Robustness / DSR ---
# The lab spec pre-registers "~80 configurations" as the honest multiple-
# testing count for the deflated Sharpe ratio. Our executed grid (W x
# threshold x estimator = 3*3*3 = 27) is smaller than that; 80 additionally
# accounts for informal researcher degrees of freedom exercised while
# building this lab (estimator formulation choices, direction-rule lookback
# choices, etc.) that a strict grid count would under-state. We use the
# user's stated ~80 rather than only the executed-grid count of 27, since
# using the smaller number would understate the multiple-testing penalty.
N_CONFIGS_FOR_DSR = 80
EXECUTED_GRID_SIZE = len(W_GRID) * len(Z_THRESHOLD_GRID) * 3

# --- Hard limits ---
TSMOM_CORR_REJECT = 0.7

DATA_DIR = "irreversibility_lab/data"
RESULTS_DIR = "irreversibility_lab/results"
