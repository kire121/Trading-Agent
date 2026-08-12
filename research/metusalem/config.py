"""Metusalem -- shared configuration.

Trendalderns hasard som tvarsnittstilt (survival-analysis/renewal-theory tilt
on trend-episode age). See docs/metusalem_forregistrering.md for the full,
locked pre-registration -- this module implements exactly its Section 5/6/9/10/12
numeric constants, nothing more.

Provenance of the two universes (rule 2 -- reuse before build, verified via
lib.hashutil.compute_config_hash convention, sha256(sorted(tickers))):

- IS_UNIVERSE: verbatim copy of research/smittotalet/config.py::IS_UNIVERSE
  (branch claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b).
  sha256(sorted(...)) = 0792f63ab2e055f5ead458523ba99169bcfd89daaaafd105dbbb32278b1dbc8a,
  which matches the spec's stated prefix "0792f63ab2e0..." and the
  registry/ytor.jsonl Y1/Y8 entries' tickerlista_sha256 -- confirmed identical
  panel, verified independently rather than assumed.
- OOS_UNIVERSE: verbatim flattened copy of research/runraden/config.py::OOS_UNIVERSE
  (branch claude/runraden-vecko-ordning-vvztim, commit a4e0d53), ISIN-locked list.
  sha256(sorted(...)) = cf601d404e858cfa2fa151f24c9c00a5355e25e093f97b08b0a18fcdd5f57029,
  matching the spec's stated prefix "cf601d404e85...".
"""
import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Identity / seed
# ---------------------------------------------------------------------------
STRATEGY_NAME = "metusalem"
SEED = 20260812

# ---------------------------------------------------------------------------
# IS universe -- verbatim from research/smittotalet/config.py::IS_UNIVERSE.
# ---------------------------------------------------------------------------
IS_UNIVERSE = [
    "SPY", "QQQ", "IWM", "IJH", "VTI", "VUG",
    "EFA", "EEM", "VEA", "VWO",
    "XLF", "XLK", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "XBI", "KRE", "XRT", "IYT",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "MUB", "AGG", "EMB",
    "UUP", "FXE", "FXY",
    "VNQ",
    "DBC",
]
assert len(IS_UNIVERSE) == 40
assert len(set(IS_UNIVERSE)) == 40
IS_UNIVERSE_SHA256 = "0792f63ab2e055f5ead458523ba99169bcfd89daaaafd105dbbb32278b1dbc8a"

# ---------------------------------------------------------------------------
# OOS universe -- verbatim flattened from research/runraden/config.py::OOS_UNIVERSE.
# Ticker strings carry the exchange suffix as Runraden wrote them
# (e.g. "IWDA.LSE"); lib.eodhd_client wants a bare ticker + separate
# `exchange=` argument (docs/INSTRUKTION.md sec.7 "tysta kontraktsandringen"
# warning) -- data.py splits on the LAST "." before use, never assumes.
# ---------------------------------------------------------------------------
OOS_UNIVERSE = [
    "IWDA.LSE", "CSPX.LSE", "EIMI.LSE",
    "EUNK.XETRA", "EXW1.XETRA", "EUNN.XETRA", "ISF.LSE", "EXS1.XETRA",
    "ICGA.XETRA", "R2US.LSE", "INRG.LSE",
    "EUNH.XETRA", "IEAC.LSE", "EUNA.XETRA", "IGLT.LSE", "7USH.XETRA",
    "HYEA.LSE", "IEMB.LSE", "IUST.XETRA", "IUS5.XETRA",
    "SGLN.LSE", "SSLV.LSE", "ICOM.LSE",
    "DPYA.LSE",
]
assert len(OOS_UNIVERSE) == 24
assert len(set(OOS_UNIVERSE)) == 24
OOS_UNIVERSE_SHA256 = "cf601d404e858cfa2fa151f24c9c00a5355e25e093f97b08b0a18fcdd5f57029"

# ---------------------------------------------------------------------------
# Dates (spec S6, S11)
# ---------------------------------------------------------------------------
# History buffer before IS-A start: enough calendar time for the earliest
# panel entrants to clear the 78-week warmup before 2004-01-01. 2001-01-01
# gives >3 years of headroom for the first Friday-with->=30-warmed assertion.
HISTORY_START = "2001-01-01"
IS_A_START = "2004-01-01"
IS_A_END = "2017-12-31"
IS_B_START = "2018-01-01"
IS_B_END = "2026-06-30"
IS_END = "2026-06-30"          # is_end for the OOS lock (spec S11)
IS_DATA_END = "2026-06-30"     # data_end for the IS read (== is_end -> locked)
OOS_DATA_END = "2026-06-30"    # OOS data_end, upon unlock only
# Sentinel is_end used ONLY to drive lib.oos_loader's date-based gate for the
# OOS panel's universe-based lock (see data.py::load_oos_panel docstring).
# Not a real date in this study -- deliberately before any OOS ticker's
# inception so data_end > is_end is unconditionally true.
OOS_LOCK_SENTINEL_IS_END = "1900-01-01"

TRADING_DAYS_YEAR = 252

# ---------------------------------------------------------------------------
# Warmup / entry (spec S5 "Alder", S6, S10 Steg 0a)
# ---------------------------------------------------------------------------
# "+78v uppvarmning: 52v signal + 26v vol": declared as an atomic 78-week
# (546 calendar day) minimum-history gate before an instrument enters the
# tradable panel -- 52w covers the 12m TSMOM lookback, +26w is the author's
# declared extra buffer for a stable vol-sizing denominator. The 52/26
# split is explanatory, not two separately-gated checkpoints; operationalized
# here as the single mechanical number actually given: 78 weeks.
WARMUP_WEEKS = 78
MIN_WARMED_INSTRUMENTS = 30
BACKTEST_START_DEADLINE = "2006-12-31"  # S6: assert backtest start <= this date

# ---------------------------------------------------------------------------
# Base book (frozen, reused verbatim from research/smittotalet/tsmom.py)
# ---------------------------------------------------------------------------
TSMOM_LOOKBACK = 252            # 12-month sign momentum, trading days
TSMOM_VOL_LOOKBACK = 20         # inverse-vol sizing window, trading days
PORTFOLIO_VOL_TARGET = 0.10     # 10% annualized ex-ante
GROSS_CAP = 2.0                 # 200%
VOL_TARGET_SOLVE_MAX_ITER = 25
VOL_TARGET_SOLVE_TOL = 1e-4
REBALANCE_WEEKDAY = "FRI"

# ---------------------------------------------------------------------------
# Tilt (spec S5)
# ---------------------------------------------------------------------------
KAPPA_PRIMARY = 0.5
KAPPA_GRID = (0.25, 0.5, 0.75)
EXEC_LAG_GRID = (1, 2)          # trading days beyond the base book's own
                                 # Friday-close->Monday cadence; see
                                 # signal_construction.py docstring for the
                                 # exact operationalization and its AVVIKELSE.
EXEC_LAG_PRIMARY = 1
GRID_CELLS = tuple((k, lag) for k in KAPPA_GRID for lag in EXEC_LAG_GRID)  # 6 cells
assert len(GRID_CELLS) == 6

# ---------------------------------------------------------------------------
# Costs (spec S5 "Kostnader", S13) -- flat bps, deliberately simple (no ADV
# buckets -- that is Smittotalet's DIFFERENT, more elaborate cost model,
# not reused here since the spec explicitly declares a flat-bps convention).
# ---------------------------------------------------------------------------
COST_BPS_IS_PRIMARY = 5.0
COST_BPS_IS_SENSITIVITY = (2.0, 5.0, 10.0)
COST_BPS_OOS = 10.0

# ---------------------------------------------------------------------------
# T-B permutation twin (spec S9)
# ---------------------------------------------------------------------------
TB_REDRAW_WEEKS = 8
TB_N_DRAWS_IC = 500
TB_N_DRAWS_PORTFOLIO = 200

# ---------------------------------------------------------------------------
# Liveness assertions (spec S9)
# ---------------------------------------------------------------------------
TA_SORT_TOLERANCE = 1e-9
EXANTE_VOL_TOLERANCE_BPS = 1.0  # "+-1 bp" ex-ante vol equality across variants
MIN_EFFECTIVE_N_RATIO = 0.8     # effective-N (tilted) >= 0.8x T0, every week

# ---------------------------------------------------------------------------
# Steg 0c -- machine gate (spec S10)
# ---------------------------------------------------------------------------
STEG0C_N_SIMS = 100
STEG0C_PASS_MIN = 90     # planted Weibull k=0.7 must pass Steg1a+1b in >=90/100
STEG0C_FALSEPASS_MAX = 10  # exponential mixture (k=1) false-pass in <=10/100
STEG0C_BOOTSTRAP_B = 500
STEG0C_N_INSTRUMENTS = 40
STEG0C_N_WEEKS = 730
STEG0C_WEIBULL_K = 0.7
STEG0C_EXP_LAMBDA_RANGE = (1.0 / 40.0, 1.0 / 10.0)  # per week

PRODUCTION_BOOTSTRAP_B = 2000

# ---------------------------------------------------------------------------
# Steg 0b -- episode inventory + oracle ceiling (spec S10)
# ---------------------------------------------------------------------------
K0B1_MIN_COMPLETED_EPISODES = 400
K0B2_MIN_INSTRUMENTS_WITH_5 = 25
K0B2_MIN_EPISODES_PER_INSTR = 5
K0B3_MIN_A_MAX_WEEKS = 26
K0B3_MIN_N_AT_RISK = 30
K0B4_MIN_ORACLE_SR_UPLIFT = 0.30

# ---------------------------------------------------------------------------
# Steg 1 -- hazard structure (spec S10)
# ---------------------------------------------------------------------------
K1A_MAX_K_HAT = 0.90
K1A_MAX_CI95_UPPER = 1.00
K1B_MIN_DELTA_AIC = 6.0
K1_WEIBULL_K_BOUNDS = (0.2, 3.0)

# ---------------------------------------------------------------------------
# Steg 2 -- redundancy screen (spec S10)
# ---------------------------------------------------------------------------
K2_MAX_R2 = 0.5
K2_HAC_LAGS = 8  # 1-week non-overlapping outcome -> lag 8, house convention

# ---------------------------------------------------------------------------
# Steg 3 -- IC (spec S10)
# ---------------------------------------------------------------------------
K3_1_MIN_IC = 0.02
K3_1_MIN_NW_T = 2.5
K3_2_MIN_IC_RES = 0.015
K3_2_MIN_NW_T = 2.0
NW_HAC_LAGS = 8

# ---------------------------------------------------------------------------
# Steg 4 -- effective breadth (spec S10)
# ---------------------------------------------------------------------------
K4_1_MAX_PC1_SHARE = 0.35
K4_2_COMPRESSION_AGE_SPREAD_WEEKS = 8   # a_q90 - a_q10 < 8w counts as "compressed"
K4_2_MAX_COMPRESSION_SHARE = 0.25
K4_3_MIN_RANK_PERSISTENCE = 0.6
K4_3_LAG_WEEKS = 4
K4_4_MAX_TURNOVER_RATIO = 1.3

# ---------------------------------------------------------------------------
# Steg 5a -- IS-A backtest + grid (spec S10)
# ---------------------------------------------------------------------------
K5_1_MIN_PRIMARY_UPLIFT = 0.15
K5_3_MIN_THIRDS_POSITIVE = 2   # out of 3
K5_3_MIN_THIRD_UPLIFT = -0.10
K5_5_TB_N_DRAWS = TB_N_DRAWS_PORTFOLIO
N_IS_A_SUBPERIODS = 3

# ---------------------------------------------------------------------------
# Steg 6 -- OOS (spec S10, S11)
# ---------------------------------------------------------------------------
N_EFFECTIVE_SURFACE_READS = 9  # US 40-ETF: 8 prior reads (registry Y1..Y8) + this study

# ---------------------------------------------------------------------------
# Diversification (spec S13)
# ---------------------------------------------------------------------------
MAX_ABS_BETA_SPY_UPLIFT = 0.15

# ---------------------------------------------------------------------------
# Paths / EODHD access
# ---------------------------------------------------------------------------
EODHD_API_KEY_ENV = "EODHD_API_KEY"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
# EODHD data is licensed and must NEVER be committed (lib/eodhd_client.py's
# own convention: "cache_dir default ligger UTANFOR repot") -- default cache
# lives under the system tempdir, matching lib.eodhd_client.DEFAULT_CACHE_DIR,
# NOT under research/metusalem/.
import tempfile as _tempfile
CACHE_DIR = os.path.join(_tempfile.gettempdir(), "eodhd_cache_metusalem")
DATA_DIR = os.path.join(_THIS_DIR, "data")
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results", STRATEGY_NAME)

FIELDS = ("open", "high", "low", "close", "adjusted_close", "volume")


@dataclass(frozen=True)
class GridCell:
    kappa: float
    exec_lag: int


GRID = tuple(GridCell(kappa=k, exec_lag=lag) for k, lag in GRID_CELLS)
PRIMARY_CELL = GridCell(kappa=KAPPA_PRIMARY, exec_lag=EXEC_LAG_PRIMARY)
