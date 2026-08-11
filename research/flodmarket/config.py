"""Flodmarket -- shared configuration.

Locked per docs/flodmarket_forregistrering.md (pre-registration v1.0,
2026-08-11, is_end 2026-06-30, seed 20260811, selektionslasning #8 av
EODHD-US40-ytan). "Detta dokument ar komplett: implementationen ska aldrig
behova fatta designbeslut. Alla trosklar ar lasta har" -- every constant
below is either (a) pinned verbatim by the spec, (b) reused verbatim from a
named sibling-branch module per the spec's own reuse instruction (session
rule 2), or (c) DECLARED with a one-line rationale where the spec leaves an
operational detail unpinned. See AVVIKELSER.md for the full interpretation
log -- nothing here is silently assumed.
"""
import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------
GLOBAL_SEED = 20260811          # spec header
SELECTION_READING = 8            # "selektionslasning #8 av EODHD-US40-ytan"
N_EFFECTIVE_SURFACE_READS = 8    # spec SS9 Steg5 [K5.2], SS11 registerappend -- LOCKED
                                  # by this spec directly, not inherited from any prior
                                  # strategy's own counter.

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DATA_CACHE_DIR = os.path.join(_THIS_DIR, "data_cache")  # gitignored, never committed
RESULTS_DIR = os.path.join(_REPO_ROOT, "results", "flodmarket")
REGISTRY_PATH = os.path.join(_REPO_ROOT, "registry", "ytor.jsonl")

# ---------------------------------------------------------------------------
# Universe -- "US40-panelen -- tickerlista verbatim fran registry/ytor.jsonl
# Y1 / research/timglaset/config_frozen.yaml; assert sha256(sorterad lista)
# prefix 0792f63ab2e0" (spec SS4).
#
# Provenance: registry/ytor.jsonl entry yta_id="Y1_timglaset_us40etf"
# (present on branches claude/strategy-spec-implementation-axn5co and
# claude/timglaset-levande-komponenter-g8v0j3, tickerlista_sha256 =
# 0792f63ab2e055f5ead458523ba99169bcfd89daaaafd105dbbb32278b1dbc8a), itself
# copied verbatim from research/smittotalet/config.py:IS_UNIVERSE, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b.
#
# AVVIKELSE (logged in AVVIKELSER.md): the spec's alternate source path
# "research/timglaset/config_frozen.yaml" does not exist under that literal
# path in this repo (the actual delivered frozen config lives at
# results/timglaset/config_frozen.yaml, per docs/INSTRUKTION.md's delivery
# schema). Not material: the spec names registry/ytor.jsonl Y1 as an
# equally valid alternate source, and its hash independently verified below
# to match the required prefix -- see tests/test_config.py.
# ---------------------------------------------------------------------------
IS_UNIVERSE = [
    "AGG", "DBC", "EEM", "EFA", "EMB", "FXE", "FXY", "HYG", "IEF", "IJH",
    "IWM", "IYT", "KRE", "LQD", "MUB", "QQQ", "SHY", "SMH", "SPY", "TIP",
    "TLT", "UUP", "VEA", "VNQ", "VTI", "VUG", "VWO", "XBI", "XLB", "XLC",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XRT",
]
assert len(IS_UNIVERSE) == 40
assert len(set(IS_UNIVERSE)) == 40
UNIVERSE_SHA256_PREFIX = "0792f63ab2e0"

# ---------------------------------------------------------------------------
# OOS universe -- UCITS panel, ISIN-lista sha256-prefix cf601d404e85, "Runradens
# config.py" (spec SS11). Provenance: research/runraden/config.py:OOS_UNIVERSE,
# branch claude/runraden-vecko-ordning-vvztim, commit a4e0d53, copied via
# research/timglaset/config.py:OOS_UNIVERSE (branch
# claude/strategy-spec-implementation-axn5co, commit 408c81f), which itself
# reproduced it verbatim with provenance. Hash independently re-verified
# here (see tests/test_config.py) to match the required prefix.
#
# HARD RULE (session instruction, stricter than the spec's own gating,
# rule 3): this universe's price history is NEVER fetched or read in this
# session, regardless of whether Steg 0-4 pass. See oos_guard.py. Unlock
# requires explicit user go-ahead in this session AFTER the full fast-exit
# ladder has passed IS -- never during development/debugging.
# ---------------------------------------------------------------------------
OOS_UNIVERSE = {
    "broad_equity": ["IWDA.LSE", "CSPX.LSE", "EIMI.LSE"],
    "regional_equity": ["EUNK.XETRA", "EXW1.XETRA", "EUNN.XETRA", "ISF.LSE",
                         "EXS1.XETRA", "ICGA.XETRA", "R2US.LSE", "INRG.LSE"],
    "rates_credit": ["EUNH.XETRA", "IEAC.LSE", "EUNA.XETRA", "IGLT.LSE",
                      "7USH.XETRA", "HYEA.LSE", "IEMB.LSE", "IUST.XETRA", "IUS5.XETRA"],
    "commodities": ["SGLN.LSE", "SSLV.LSE", "ICOM.LSE"],
    "real_assets": ["DPYA.LSE"],
}


def oos_universe_flat() -> list:
    out = []
    for bucket in OOS_UNIVERSE.values():
        out.extend(bucket)
    return out


assert len(oos_universe_flat()) == 24
assert len(set(oos_universe_flat())) == 24
OOS_UNIVERSE_SHA256_PREFIX = "cf601d404e85"

# ---------------------------------------------------------------------------
# IS/OOS date window (spec SS0, SS11, locked)
# ---------------------------------------------------------------------------
IS_START = "2004-01-01"
IS_END = "2026-06-30"
TRADING_DAYS_YEAR = 252
WEEKS_YEAR = 52

# Burn-in: "Ticker handlas fr.o.m. dag 252+K+20 av egen historik" (spec SS4).
BURN_IN_DAYS_BASE = 252 + 20     # + K (grid-dependent, added at call site)

# ---------------------------------------------------------------------------
# Signal (spec SS2.2, primary cell locked SS9 Steg4)
# ---------------------------------------------------------------------------
K_PRIMARY = 40
Z_STAR_PRIMARY = 2.0
FE_DEMEAN_WINDOW = 252            # window [t-K-251, t-K], disjoint from K-window
FE_DEMEAN_PRIMARY = True          # primary cell: demean="252d"
K_EFF_MIN_FRACTION = 0.8          # K_eff >= 0.8*K else S=NaN => g=0

# ---------------------------------------------------------------------------
# Sizing -- "raw_i = g_i / sigma_hat_i, sigma_hat = rullande 20d std av
# dagliga log-avkastningar (adj close), annualiserad. w = k*raw, k lost
# iterativt mot 10% arsvolmal med bruttotak sum|w| <= 200% -- exakt samma
# kalibreringsvag som Smittotalets basbok" (spec SS4).
#
# The sigma_hat FORMULA itself is pinned verbatim by the spec (annualized
# 20d rolling std of log returns) and takes precedence over Smittotalet's
# own base-book convention (plain, non-annualized pct-change std) where the
# two differ -- "exakt samma kalibreringsvag" refers to the ITERATIVE-K
# SOLVE MECHANISM (fixed-point iteration against realized IS vol, gross cap
# applied before, never a one-shot rescale on top of a binding cap --
# Dammluckans bugglarning), not to the per-asset vol estimator itself.
#
# Iterative-k engine ported verbatim: research/smittotalet/tsmom.py::
# solve_k_for_target_vol + apply_gross_cap, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b. See sizing.py.
# ---------------------------------------------------------------------------
SIGNAL_VOL_LOOKBACK = 20          # rolling 20d std of daily LOG returns, annualized
PORTFOLIO_VOL_TARGET = 0.10       # 10% annualized
GROSS_CAP = 2.0                   # 200%
VOL_TARGET_SOLVE_MAX_ITER = 25    # Smittotalet/Dammluckan iterative-k-solve pattern
VOL_TARGET_SOLVE_TOL = 1e-4
REBALANCE_WEEKDAY = "FRI"         # signal Friday close -> fill Monday close (spec SS4)

# ---------------------------------------------------------------------------
# Costs -- ADV-bucket model (spec SS4: "ADV-bucket-modellen om den finns i
# Trading-Agent (grep adv|cost)" -- found, see AVVIKELSER.md; the flat
# 5bp/12bp fallback is therefore NOT used).
#
# Provenance: research/smittotalet/costs.py, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b (itself ported
# verbatim from research/dammluckan/costs.py <- research/formdriften/costs.py).
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
# UCITS fallback (spec SS4, only used if/when OOS is unlocked and the ADV
# model cannot be computed there, e.g. thin ADV history for a UCITS ETF):
UCITS_FLAT_COST_BPS = 12.0
US_FLAT_COST_BPS = 5.0            # declared caveat fallback, not expected to trigger

# ---------------------------------------------------------------------------
# Grid (spec SS9 Steg4): K x z* x demean = 3x3x2 = 18 cells.
# ---------------------------------------------------------------------------
K_GRID = (20, 40, 60)
Z_STAR_GRID = (1.5, 2.0, 3.0)
DEMEAN_GRID = (None, 252)         # None = "ingen", 252 = "252d"


@dataclass(frozen=True)
class GridCell:
    K: int
    z_star: float
    demean: object  # None or 252


GRID = tuple(
    GridCell(K=k, z_star=z, demean=d)
    for k in K_GRID
    for z in Z_STAR_GRID
    for d in DEMEAN_GRID
)
assert len(GRID) == 18
PRIMARY_CELL = GridCell(K=K_PRIMARY, z_star=Z_STAR_PRIMARY, demean=FE_DEMEAN_WINDOW)
assert PRIMARY_CELL in GRID

IS_THIRDS = (
    ("2004-01-01", "2011-06-30"),
    ("2011-07-01", "2018-12-31"),
    ("2019-01-01", "2026-06-30"),
)

# ---------------------------------------------------------------------------
# Twins (spec SS8)
# ---------------------------------------------------------------------------
T3_N_DRAWS = 500
T3_BLOCK_DAYS = 21
T5_SIMPLICITY_TOLERANCE = 0.05    # SR_T5 >= SR_primary - 0.05 => adopt T5's form

# Liveness assertions (spec SS8, ovillkorade -- Ekolodet/Dammluckan): reuses
# lib.twins.twin_is_alive (branch claude/cepstral-metaorder-detection-b8nvwb,
# commit 0edbc4a, twin_is_alive; branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b,
# quantile_map_to -- consolidated verbatim in lib/twins.py on main).
LIVENESS_MIN_NONZERO_WEEK_FRACTION = 0.95
LIVENESS_GROSS_RATIO_RANGE = (0.5, 2.0)     # vs primary's median gross
LIVENESS_T3_TURNOVER_RATIO_RANGE = (0.5, 1.5)  # T3-null median turnover vs primary's

# ---------------------------------------------------------------------------
# Fast-exit-stege numeric criteria (spec SS9, exact)
# ---------------------------------------------------------------------------
# Steg 0a
STEG0A_PRECLAMP_ERROR_MAX_FRACTION = 0.01     # per ticker-year
STEG0A_MISSING_BARS_MAX_FRACTION = 0.05
STEG0A_SYNTHETIC_OPEN_MAX_FRACTION = 0.30     # per ticker-year -> excludes that year
STEG0A_ZERO_RANGE_MAX_FRACTION_IS = 0.10      # per ticker over IS -> excludes ticker
STEG0A_MIN_TRADEABLE_TICKERS = 25             # KILL below this

# Steg 0b
STEG0B_STD_S_LOW_MULT = 0.5    # * q05_synt
STEG0B_STD_S_HIGH_MULT = 2.0   # * q95_synt
STEG0B_EXTREME_S_THRESHOLD = 0.95
STEG0B_EXTREME_S_MULT = 3.0    # * q99_synt
STEG0B_NAN_S_KILL_FRACTION = 0.20
STEG0B_NAN_S_FLAG_FRACTION = 0.10
STEG0B_AB_PLANTED_TRUE_IC = 0.03
STEG0B_AB_PLANTED_MEASURED_IC_MIN = 0.02
STEG0B_AB_NULL_EXCEEDANCE_TARGET = 0.05
STEG0B_AB_NULL_EXCEEDANCE_TOLERANCE = 0.03
STEG0B_AB_N_SIMS = 200

# Steg 1
STEG1_MIN_POOLED_RANK_IC = 0.015
STEG1_MIN_NW_T = 2.5
STEG1_NW_MAXLAGS = 4
STEG1_N_NULL_DRAWS = 500
STEG1_NULL_BLOCK_DAYS = 21
STEG1_NULL_PERCENTILE = 95

# Steg 2
STEG2_MIN_INCREMENTAL_NW_T = 2.0
STEG2_MIN_INCREMENTAL_RANK_IC = 0.010
STEG2_MAX_R2 = 0.5
STEG2_REFORMULATION_POOL_ADD = 18

# Steg 3
STEG3_MAX_PC1_SHARE = 0.35
STEG3_MIN_WITHIN_ASSET_VAR_SHARE = 0.50
STEG3_RANK_AC_RANGE = (0.40, 0.97)
STEG3_MIN_POSITIVE_ASSET_IC_FRACTION = 0.60
STEG3_MIN_POOLED_IC_EXCL_TOP3 = 0.010

# Steg 4
STEG4_MIN_PRIMARY_NET_SHARPE = 0.40
STEG4_MIN_POSITIVE_THIRDS = 2
STEG4_MIN_POSITIVE_GRID_CELLS = 12
STEG4_MIN_SR_DELTA_VS_T2 = 0.10
STEG4_MIN_SR_DELTA_VS_T4 = 0.10
STEG4_T3_NULL_PERCENTILE = 95
STEG4_MAX_MEDIAN_WEEKLY_TURNOVER_FRACTION = 0.35

# Steg 5 (LOCKED -- never executed without explicit user go-ahead, rule 3)
STEG5_MIN_OOS_NET_SHARPE = 0.25
STEG5_BOOTSTRAP_BLOCK_WEEKS = 4
STEG5_BOOTSTRAP_N_DRAWS = 1000
STEG5_BOOTSTRAP_MIN_P_POSITIVE = 0.90
STEG5_MAX_ABS_BETA_SPY = 0.20
STEG5_MAX_ABS_RHO_TSMOM = 0.30
STEG5_MIN_POSITIVE_ASSET_IC_FRACTION_OOS = 0.55
STEG5_DSR_FALLBACK_M_EFF = 207     # 18 + 7*27, SR* = 0 -- only if tiling module missing
                                    # (it is not missing here, see nulls.py)

# ---------------------------------------------------------------------------
# EODHD access
# ---------------------------------------------------------------------------
EODHD_API_KEY_ENV = "EODHD_API_KEY"
FIELDS = ("open", "high", "low", "close", "adjusted_close")  # volume NOT used (spec)
