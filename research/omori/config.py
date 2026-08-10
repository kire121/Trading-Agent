"""
Efterskalvsklockan (the "aftershock clock") -- shared configuration.

Every constant that is DECLARED (not literally pinned by the brief) is
flagged as such, following the house convention used by the sibling
research branches (Dammluckan, Vridmomentet, Vindkastet, Formdriften):
nothing here is silently assumed.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# IS / exploration panel: "den ursprungliga EODHD-multi-asset-panelen (~40
# US-ETF:er)". No single prior branch's universe both (a) came over EODHD and
# (b) avoids overlapping the locked OOS lands panel below -- Formdriften's own
# ~40-ETF universe (its closest textual match) was fetched over Yahoo (EODHD
# access was not yet working when it ran) and its 29-name COUNTRY_ETFS leg
# overlaps 15/16 of the OOS lands panel, which would silently spend the
# "virgin" OOS surface during "free" IS design. DECLARED: we build a fresh
# ~40-ticker, EODHD-native, cross-asset US ETF panel for IS design that is
# constructed to have ZERO ticker overlap with the OOS panel (no single-
# country, no single-commodity names) -- broad equity, sector SPDRs,
# industry/thematic, fixed income, FX, REIT, diversified commodity. This
# panel is spent freely per the brief ("redan kontaminerad, spenderas fritt
# på design").
IS_UNIVERSE = [
    # Broad US equity (6)
    "SPY", "QQQ", "IWM", "IJH", "VTI", "VUG",
    # Broad international, diversified only -- no single-country (4)
    "EFA", "EEM", "VEA", "VWO",
    # Sector SPDRs (11)
    "XLF", "XLK", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC",
    # Industry / thematic (5)
    "SMH", "XBI", "KRE", "XRT", "IYT",
    # Fixed income (9)
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "MUB", "AGG", "EMB",
    # FX (3)
    "UUP", "FXE", "FXY",
    # REIT (1)
    "VNQ",
    # Diversified commodity (1)
    "DBC",
]  # 40 tickers

# OOS lock #1: "16-lands-ETF-panelen" -- the identical 16-name single-country
# equity ETF panel used, verbatim, by both Vindkastet (research/vindkastet)
# and Dammluckan (research/dammluckan) as their own secondary/OOS-2 universe.
# This is the SAME list (not independently re-derived) so that this
# strategy's DSR correction can honestly say "3rd read" rather than silently
# under-counting. See REPORT.md Section "DSR-korrigering" for the accounting.
OOS_LANDS = ["EWJ", "EWG", "EWU", "EWQ", "EWI", "EWP", "EWL", "EWA", "EWC",
             "EWY", "EWT", "EWZ", "EWW", "EWS", "EWH", "EWD"]

# OOS lock #2: single-commodity ETF/ETC panel. The brief names GLD, SLV, USO,
# UNG, DBA, DBB, CPER explicitly ("...") as an opening list; DECLARED: we
# extend it with five more liquid, genuinely single-commodity names (CORN,
# WEAT, SOYB, PALL, PPLT) to raise the OOS event count. CAVEAT (must be
# reported, not hidden): GLD, SLV and USO are NOT fully "jungfruliga" at the
# instrument level -- all three already sit inside Dammluckan's and
# Vindkastet's own 16-ticker PRIMARY (IS design) panel, so their volume/
# return dynamics have already informed two prior strategies' design
# choices, even though neither has spent them on a locked OOS confirmation
# read. REPORT.md reports the OOS result both with and without these three
# tickers as a sensitivity check.
OOS_COMMODITIES = ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "CPER",
                    "CORN", "WEAT", "SOYB", "PALL", "PPLT"]
OOS_COMMODITIES_CONTAMINATED = ["GLD", "SLV", "USO"]  # see caveat above

OOS_UNIVERSE = OOS_LANDS + OOS_COMMODITIES

# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------
DOLLAR_VOLUME_LOOKBACK = 120       # rolling median/MAD window, brief-pinned
RETURN_VOL_LOOKBACK = 60           # sigma_hat_60, brief-pinned
# DECLARED (ambiguity resolution): both rolling baselines are computed over
# the window (t-N .. t-1), i.e. EXCLUDING day t0 itself, then evaluated at
# t0. This is what "endast data <= t0" cashes out to without being
# self-referential (an extreme day inflating its own baseline would bias
# detection toward conservatism, not toward lookahead -- but excluding it is
# the cleaner, more standard causal construction and is declared as such).
MAD_SCALE = 1.4826                 # standard consistency constant for a
                                    # normal-equivalent MAD-based z-score

Z_VOLUME_THRESHOLD = 4.0            # brief-pinned primary cell
R0_SIGMA_THRESHOLD = 2.0            # brief-pinned primary cell ("2 sigma_60")

# Grid neighborhood for robustness (Section "Fallgropar bevakade")
Z_VOLUME_GRID = (3.0, 4.0, 5.0)

# Same-day clustering / de-dup ("Dammluckans samtidighetslärdom")
CLUSTER_CORR_LOOKBACK = 60          # DECLARED: matches RETURN_VOL_LOOKBACK
CLUSTER_CORR_THRESHOLD = 0.7        # brief-pinned

# ---------------------------------------------------------------------------
# Omori fit (sequential, causal)
# ---------------------------------------------------------------------------
FIT_START_TAU = 5                   # brief-pinned: "fran tau=5"
C_PROFILE_GRID = (0, 1, 2)          # brief-pinned
MIN_POSITIVE_EXCESS_DAYS = 4        # brief-pinned: "kraver >=4 dagar positiv
                                     # excess, annars full shrinkage"

# Empirical-Bayes shrinkage strength kappa. NOT pinned by the brief.
# DECLARED: selected once via leave-one-instrument-out cross-validation on
# the IS panel only (minimize MSE of p_tilde against each held-out
# instrument's own realized halving-time-implied p), over this candidate
# grid, then frozen. See signal.calibrate_kappa() / output/kappa.json.
KAPPA_GRID = (2.0, 5.0, 10.0, 20.0)
KAPPA_DEFAULT = 5.0                 # overwritten by calibrate_kappa() cache

# Reference exponent p* used only in sizing (w = sign(r0)*min(1, p*/p_tilde)).
# NOT pinned by the brief. DECLARED: p* = the frozen IS-panel median of
# ENTRY-day (tau=1) p_tilde across all valid IS events, i.e. "typical" decay
# speed -- events decaying slower than typical get full-size tilt, faster
# get proportionally downsized. Computed once by run_research.py and cached
# to output/p_star.json; OOS reads the frozen value, never recomputes it.
P_STAR_DEFAULT = None               # filled in by calibrate step

# ---------------------------------------------------------------------------
# Exit clock
# ---------------------------------------------------------------------------
THETA_DEFAULT = 0.25                # brief-pinned primary cell
THETA_GRID = (0.15, 0.25, 0.35)     # brief-pinned grid neighborhood
TAU_EXIT_FLOOR = 3                  # brief-pinned
TAU_EXIT_CAP_DEFAULT = 20           # brief-pinned primary cell
TAU_EXIT_CAP_GRID = (15, 20, 25)    # brief-pinned grid neighborhood

# ---------------------------------------------------------------------------
# Sizing / portfolio
# ---------------------------------------------------------------------------
VOL_TARGET_DAILY = 0.0040           # 40 bps, brief-pinned
VOL_TARGET_LOOKBACK = 20            # DECLARED: trailing realized-vol lookback
                                     # for sigma_hat_i, matches Dammluckan's
                                     # SIZING_VOL_LOOKBACK convention.
GROSS_CAP = 1.50                    # 150%, brief-pinned
HARD_STOP_MULT = -2.0               # brief-pinned: "-2x dagsriskbudget
                                     # kumulativt" -- cumulative position P&L
                                     # (in return space, since entry) breaches
                                     # HARD_STOP_MULT * VOL_TARGET_DAILY.
TRADING_DAYS_YEAR = 252

# ---------------------------------------------------------------------------
# Costs -- ADV-bucket model, ported verbatim from research/formdriften/costs.py
# via research/dammluckan/costs.py (the one established ADV-bucketed cost
# model precedent in this repository's research-branch series).
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
# DOCUMENTED SPREAD CAVEAT (verbatim convention, restated from Formdriften /
# Dammluckan): EODHD does not provide quoted bid/ask spread history, so the
# half-spread leg below is approximated from trailing dollar-ADV via a
# monotone liquidity bucket -- a documented approximation, not a measured
# spread. Only the commission leg (3bp/side) is a clean, un-approximated
# assumption.

# ---------------------------------------------------------------------------
# Redundancy / estimator-null / IC gates
# ---------------------------------------------------------------------------
REDUNDANCY_R2_KILL = 0.50           # DECLARED: matches Dammluckan's own
                                     # redundancy_regression kill threshold.
BLOCK_LENGTH = 20                   # DECLARED: ~1 trading month, matches the
                                     # block length convention used across
                                     # Formdriften/Dammluckan for their own
                                     # estimator nulls and block permutation.
N_BLOCK_DRAWS = 500                 # DECLARED, matches Dammluckan.
N_ESTIMATOR_NULL_DRAWS = 100        # within-event block-shuffle null draws.
                                     # DECLARED (computational-cost driven,
                                     # restated in REPORT.md): 500 draws over
                                     # the full ~1900-event IS population
                                     # would mean ~1.7M Huber refits at ~6ms
                                     # each (~3h); 100 draws over a random
                                     # ESTIMATOR_NULL_SUBSAMPLE keeps this to
                                     # single-digit minutes while still
                                     # giving a well-populated null.
ESTIMATOR_NULL_SUBSAMPLE = 300      # DECLARED, see above.
ESTIMATOR_NULL_BLOCK = 3            # DECLARED: circular block length for the
                                     # within-event shuffle of e(1..tau) --
                                     # short relative to typical event
                                     # length (median ~10-20 days) so the
                                     # shuffle still meaningfully scrambles
                                     # the decay ordering.
IC_PERMUTATION_DRAWS = 1000         # DECLARED: block-permutation draws for
                                     # the rank-IC / signed-IC significance
                                     # tests ("p<0.05" per brief).
IC_ALPHA = 0.05                     # brief-pinned

MIN_EVENTS_IS = 250                 # brief-pinned
MIN_EVENTS_OOS = 150                # brief-pinned

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------
HISTORY_FROM = "2003-01-01"
HISTORY_TO = "2026-08-10"           # last EODHD bar actually fetched; see
                                     # fetch_data.py for the exact per-run cut

# ---------------------------------------------------------------------------
# Three-era sub-period boundaries (IS panel), for sign-stability checks.
# DECLARED: matches the sibling-branch convention of splitting IS history
# into three roughly-equal eras rather than picking macro-narrative dates.
# ---------------------------------------------------------------------------
ERA_BOUNDARIES = ("2003-01-01", "2010-06-30", "2017-06-30", "2026-08-10")
