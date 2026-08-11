"""Timglaset -- shared configuration.

Locked per docs/timglaset_forregistrering.md (pre-registration, 2026-08-11,
"LÅST FÖRE ALL DATAAVLÄSNING"). Every constant below is either (a) pinned
verbatim by the spec, (b) imported verbatim from a named sibling module per
the spec's own instruction, or (c) DECLARED with a one-line rationale where
the spec leaves an operational detail unpinned -- house convention, see
docs/INSTRUKTION.md avsnitt 7 and every sibling branch's own config.py.

Nothing here is silently assumed; see AVVIKELSER.md for the full log of
interpretation choices.
"""
import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------
GLOBAL_SEED = 20260811  # spec header

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(_THIS_DIR, "data_cache")  # gitignored, never committed
RESULTS_DIR = os.path.join(_THIS_DIR, "..", "..", "results", "timglaset")
REGISTRY_PATH = os.path.join(_THIS_DIR, "..", "..", "registry", "ytor.jsonl")
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# ---------------------------------------------------------------------------
# Universe -- "US 40-ETF-panelen, tickerlista importerad verbatim från
# research/smittotalet/config.py" (spec §5, §10). Copied verbatim (spec
# demands verbatim import; the smittotalet branch is frozen history so a
# literal cross-branch `import` isn't possible -- this is the closest
# faithful equivalent: a verbatim COPY with provenance, same discipline the
# lib/ consolidation itself used for every other cross-branch port).
# Provenance: research/smittotalet/config.py:IS_UNIVERSE, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b.
# NOTE: bare tickers (no ".US" suffix) -- this is smittotalet's own
# convention and also matches lib.eodhd_client's expected bare-ticker +
# exchange="US" calling convention, so no suffix-stripping is needed.
# Do NOT substitute research/runraden/config.py:IS_UNIVERSE here even though
# both are "the 40-ETF panel" -- they are two independently-built panels
# that only share 29/40 tickers (the exact "29/40-incidenten" the spec's
# §1.2.4 warns about). The spec is unambiguous: smittotalet's list, verbatim.
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

# ---------------------------------------------------------------------------
# OOS universe -- "UCITS-panelen, ISIN-lista verbatim från
# research/runraden/config.py (låst före all avläsning av Runraden, k=0 --
# jungfrulig)" (spec §10). Provenance: research/runraden/config.py:
# OOS_UNIVERSE, branch claude/runraden-vecko-ordning-vvztim, commit a4e0d53.
# Runraden's own README confirms k=0 ("It remains genuinely untouched (k=0)
# for any future OOS read") and calls this dict "the full list with ISINs"
# (ISINs are in the inline comments; tickers carry the EODHD .LSE/.XETRA
# exchange suffix needed for fetching -- both are reproduced verbatim below).
#
# HARD RULE (session instruction, stricter than the spec's own gating):
# this universe's price/volume history is NEVER fetched or read in this
# session, regardless of whether Steg 0-6 pass. See oos_guard.py.
# ---------------------------------------------------------------------------
OOS_UNIVERSE = {
    "broad_equity": [
        "IWDA.LSE",   # iShares Core MSCI World, IE00B4L5Y983
        "CSPX.LSE",   # iShares Core S&P 500, IE00B5BMR087
        "EIMI.LSE",   # iShares Core MSCI EM IMI, IE00BKM4GZ66
    ],
    "regional_equity": [
        "EUNK.XETRA",  # iShares Core MSCI Europe, IE00B4K48X80
        "EXW1.XETRA",  # iShares Core EURO STOXX 50, DE0005933956
        "EUNN.XETRA",  # iShares Core MSCI Japan IMI, IE00B4L5YX21
        "ISF.LSE",     # iShares Core FTSE 100, IE0005042456
        "EXS1.XETRA",  # iShares Core DAX, DE0005933931
        "ICGA.XETRA",  # iShares MSCI China, IE00BJ5JPG56
        "R2US.LSE",    # SPDR Russell 2000 (US small cap), IE00BJ38QD84
        "INRG.LSE",    # iShares Global Clean Energy (thematic/sector proxy), IE00B1XNHC34
    ],
    "rates_credit": [
        "EUNH.XETRA",  # iShares Core Euro Govt Bond, IE00B4WXJJ64
        "IEAC.LSE",    # iShares Core Euro Corp Bond, IE00B3F81R35
        "EUNA.XETRA",  # iShares Core Global Aggregate Bond EUR Hgd, IE00BDBRDM35
        "IGLT.LSE",    # iShares Core UK Gilts, IE00B1FZSB30
        "7USH.XETRA",  # Amundi US Treasury 7-10Y EUR Hgd
        "HYEA.LSE",    # iShares Global High Yield Corp Bond, IE00BYWZ0440
        "IEMB.LSE",    # iShares JPM $ EM Bond, IE00B2NPKV68
        "IUST.XETRA",  # iShares $ TIPS USD Acc, IE00B1FZSC47
        "IUS5.XETRA",  # iShares Global Inflation Linked Govt Bond, IE00B3B8PX14
    ],
    "commodities": [
        "SGLN.LSE",   # iShares Physical Gold, IE00B4ND3602
        "SSLV.LSE",   # Invesco Physical Silver, IE00B43VDT70
        "ICOM.LSE",   # iShares Diversified Commodity Swap, IE00BDFL4P12
    ],
    "real_assets": [
        "DPYA.LSE",   # iShares Developed Markets Property Yield, IE00BFM6T921
    ],
}


def oos_universe_flat() -> list:
    tickers = []
    for bucket in OOS_UNIVERSE.values():
        tickers.extend(bucket)
    return tickers


assert 20 <= len(oos_universe_flat()) <= 25
assert len(set(oos_universe_flat())) == len(oos_universe_flat())

# ---------------------------------------------------------------------------
# IS/OOS date window (spec §10, locked)
# ---------------------------------------------------------------------------
IS_START = "2004-01-01"
IS_END = "2026-06-30"          # locked; loader refuses data_end > is_end without --unlock-oos
OOS_START = "2010-01-01"
OOS_END = "2026-06-30"
TRADING_DAYS_YEAR = 252
WEEKS_YEAR = 52

# Per-ticker history fetch: full available history (PIT entry per ticker,
# spec §5), not a single fixed HISTORY_START -- burn-in (below) is measured
# from each ticker's OWN first observed trading day, not from a panel-wide
# constant, so there is no "one size fits all" fetch-start date to declare.
FETCH_START = None  # None = EODHD's full available history

# ---------------------------------------------------------------------------
# Burn-in / validity (spec §5, §13, §14)
# ---------------------------------------------------------------------------
NORMALIZER_WINDOW = 252            # rolling median window for tau's normalizer
BURN_IN_CALENDAR_DAYS = 252 + 63   # (i) calendar index >= this many days from PIT-start
# (ii) T_t >= 2*HL_op is checked per grid cell inside opclock.compute_signal.

# ---------------------------------------------------------------------------
# Signal sizing -- "w_i = f(z_i) / sigma_hat_{i,20d} (invers 20-dagars
# EWMA-vol, repo-standard)" (spec §5).
#
# DECLARED / AVVIKELSE (logged in full in AVVIKELSER.md): exhaustive search
# (git grep across all 11 strategy branches + lib/) found NO existing
# "repo-standard" EWMA-based vol-sizing implementation anywhere in this
# repo's history -- every prior "inverse-vol sizing" (smittotalet/tsmom.py,
# dammluckan/portfolio.py, formdriften/portfolio.py) uses plain ROLLING
# (non-exponential) std over a lookback window. The spec's "repo-standard"
# claim does not check out; there is nothing to reuse per rule 2. Since this
# is a standard, unambiguous finance-industry convention (Timglaset's own
# author's use of "20-dagars EWMA-vol" -> "N-day EMA", the universal
# trading-desk meaning of which is pandas span=N, i.e. decay alpha=2/(N+1))
# rather than a genuine inferential-methodology fork (contrast: DSR formula
# family, block-bootstrap vs block-permutation -- both of which the spec
# elsewhere resolves unambiguously), this is treated as an ordinary DECLARED
# operational constant, not a STOP-worthy design gap.
SIGNAL_VOL_EWMA_SPAN = 20
SIGNAL_VOL_EWMA_MIN_PERIODS = 20

# ---------------------------------------------------------------------------
# Portfolio construction -- "exakt samma mekanism som Smittotalets basbok"
# (spec §5). Provenance: research/smittotalet/config.py, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b.
# ---------------------------------------------------------------------------
PORTFOLIO_VOL_TARGET = 0.10        # 10% annualized
GROSS_CAP = 2.0                    # 200%
VOL_TARGET_SOLVE_MAX_ITER = 25
VOL_TARGET_SOLVE_TOL = 1e-4
REBALANCE_WEEKDAY = "FRI"          # signal Friday close -> fill Monday close (t+1 lock)

# Base-book TSMOM proxy (T0 pipeline-sanity twin) -- verbatim mechanism,
# same provenance as above (research/smittotalet/tsmom.py).
BASE_TSMOM_LOOKBACK = 252          # 12-month sign momentum
BASE_TSMOM_VOL_LOOKBACK = 20       # plain rolling std, NOT the EWMA above --
                                    # T0 must reproduce smittotalet's book
                                    # verbatim, which used rolling std.
T0_TARGET_IS_SHARPE = 0.5330209642908955  # smittotalet output/is_results_summary.json,
                                            # gate "2_base_engine_alpha", IS 2004-2017
T0_TARGET_TOLERANCE = 0.03
T0_IS_START = "2004-01-01"
T0_IS_END = "2017-12-31"           # smittotalet's own IS_END, for the T0 reproduction check only

# ---------------------------------------------------------------------------
# Costs -- ADV-bucket model, repo-standard, flat cost forbidden (spec §5,
# §1.2.10). Provenance: research/smittotalet/costs.py (itself ported
# verbatim from dammluckan/costs.py <- formdriften/costs.py), branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b.
# ---------------------------------------------------------------------------
COMMISSION_BPS = 3.0
ADV_BUCKETS = (
    (500e6, 0.5),
    (200e6, 1.0),
    (100e6, 1.5),
    (50e6, 2.5),
    (20e6, 4.0),
    (0.0, 7.0),
)
ADV_COST_LOOKBACK = 63

# ---------------------------------------------------------------------------
# Parameter grid (spec §9): HL_op x c x f = 3x3x3 = 27 cells.
# ---------------------------------------------------------------------------
HL_OP_GRID = (21, 63, 126)
C_GRID = (3, 5, 8)
F_GRID = ("sign", "tanh", "clip2")


@dataclass(frozen=True)
class GridCell:
    hl_op: int
    c: float
    f: str


GRID = tuple(
    GridCell(hl_op=hl, c=c, f=f)
    for hl in HL_OP_GRID
    for c in C_GRID
    for f in F_GRID
)
assert len(GRID) == 27
PRIMARY_CELL = GridCell(hl_op=63, c=5, f="tanh")
assert PRIMARY_CELL in GRID

# ---------------------------------------------------------------------------
# T2 shuffleklocka (spec §7): 21d block, 200 draws, seeds 20260811+0..199.
# ---------------------------------------------------------------------------
T2_BLOCK_LEN = 21
T2_N_DRAWS = 200
T2_SEEDS = tuple(GLOBAL_SEED + i for i in range(T2_N_DRAWS))
T2_TOLERANCE_FRACTION = 0.20       # E|z| per draw within +-20% of primary's
T2_MIN_PASS_FRACTION = 0.95        # >=95% of draws must pass

# T3 variance clock: m5 = 5d rolling mean of r^2 (spec §7, §13)
T3_VARIANCE_WINDOW = 5

# ---------------------------------------------------------------------------
# Fast-exit-stege numeric criteria (spec §8, exact)
# ---------------------------------------------------------------------------
STEG0A_MIN_VOLUME_COVERAGE = 0.98
STEG0A_MAX_ZERO_VOLUME_FRACTION_PER_YEAR = 0.01
STEG0A_MAX_FAILED_TICKERS = 8

STEG0B_TAU_MEAN_LOW = 0.7
STEG0B_TAU_MEAN_HIGH = 1.4
STEG0B_MIN_DAYS_IN_RANGE_FRACTION = 0.95

STEG0C_MIN_ORACLE_SR_INCREMENT_OVER_T1 = 0.40

STEG1_T1_MIN_NET_SHARPE = 0.20

STEG2_MIN_ABS_IC = 0.015
STEG2_MIN_IC_MINUS_MEAN_T2 = 0.005
# "IC_op > p95 av T2:s 200 IC-värden" -- percentile computed at runtime.

STEG3_R2_KILL = 0.5
STEG3_MIN_DELTA_R2 = 0.0005
STEG3_MIN_NW_T = 2.0
HAC_LAGS = 8  # weekly outcomes, spec §1.2.11

STEG4_MAX_PC1_SHARE = 0.60
STEG4_DISPERSION_NULL_PERCENTILE = 95

STEG5_MIN_IS_OVERLAY_SHARPE = 0.30
STEG5_BOOTSTRAP_BLOCK_WEEKS = 13
STEG5_BOOTSTRAP_N_DRAWS = 2000
STEG5_BOOTSTRAP_CI = 0.90
STEG5_SUBPERIODS = (("2004-01-01", "2011-12-31"),
                     ("2012-01-01", "2019-12-31"),
                     ("2020-01-01", "2026-06-30"))
STEG5_MIN_POSITIVE_SUBPERIODS = 2
STEG5_NEIGHBOR_MIN_SR_RETENTION = 0.50

STEG6_N_GRID_CELLS = 27
# "DSR-poolning via tiling-mekanismen i research/smittotalet" (spec §1.2.3,
# §8 Steg 6). Provenance: research/smittotalet/run_research.py:168,
# `np.tile(grid_sharpes, config.N_EFFECTIVE_SURFACE_READS)`, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b -- reused as a
# one-line technique (see ladder.py), not a separate module, since that is
# literally its entire implementation.
N_EFFECTIVE_SURFACE_READS = 7  # spec §1.2.3: "6 -> 7" (smittotalet's own
                                # counter of 6 + this reading)

STEG7_MIN_VOLUME_COVERAGE = 0.95
STEG7_MAX_ZERO_VOLUME_FRACTION_PER_YEAR = 0.03
STEG7_MIN_ISIN_COVERAGE_FRACTION = 0.70
STEG7_MIN_IS_SHARPE_FRACTION = 0.40

# ---------------------------------------------------------------------------
# EODHD access
# ---------------------------------------------------------------------------
EODHD_API_KEY_ENV = "EODHD_API_KEY"
FIELDS = ("open", "high", "low", "close", "adjusted_close", "volume")
