"""
Smittotalet ("the infection count") -- shared configuration.

An epidemic effective-reproduction-number (R_t) overlay on a TSMOM base book.
Positioned against Efterskalvsklockan (omori): same self-exciting-process
phenomenon family, opposite estimand. Efterskalvsklockan asked "does this
event have its own clock?" (per-instrument Omori decay, answer: no -- the
strategy was REJECTED). Smittotalet asks "how loaded is the whole system
right now?" and uses Efterskalvsklockan's frozen, verified pooled-event
Omori-decay kernel as a FIXED INPUT (the serial-interval kernel w_s), not as
something to re-estimate.

House convention (Dammluckan / Efterskalvsklockan / Runraden / Vindkastet /
Oglegrinden): every constant that is not literally pinned by the brief is
flagged DECLARED, with a one-line rationale. Nothing here is silently
assumed.
"""
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Universe -- "IS = 40-ETF-panelen (tickerlista pinnad ur Efterskalvsklockans
# repo -- och verifieras mot Runradens rekonstruktion, per registerluckan
# baada gravarna flaggar)"
# ---------------------------------------------------------------------------
# Pinned verbatim from research/omori/config.py:IS_UNIVERSE (Efterskalvsklockan).
IS_UNIVERSE = [
    # Broad US equity (6)
    "SPY", "QQQ", "IWM", "IJH", "VTI", "VUG",
    # Broad international, diversified only (4)
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

# CAVEAT (registerluckan, flagged by BOTH prior "graves" that touched this
# panel): Runraden's own README states "No survivorship-bias handling beyond
# what EODHD serves for the listed tickers; all 40 IS tickers are currently-
# listed, still-trading ETFs" and its precursor Formdriften's REPORT.md
# independently flags "a small residual survivorship bias versus a true PIT
# vendor panel". Smittotalet inherits the same gap: this panel is fetched as
# of today, not reconstructed point-in-time, so any ETF that delisted or was
# renamed inside the sample window is silently absent. Verified against
# Runraden's own reconstruction (research/runraden/config.py:IS_UNIVERSE,
# also ~40 EODHD-native US ETFs): 29/40 tickers are shared between the two
# independently-built panels (see tests/test_config.py), which is the
# cross-check the brief asks for -- it is not, and cannot be, a fix for the
# underlying registry gap.

# ---------------------------------------------------------------------------
# Sample split. DECLARED: mirrors the IS/OOS-1 boundary used by every sibling
# branch that touches this same EODHD panel (Dammluckan, Efterskalvsklockan),
# so Smittotalet's IS design is honestly "already contaminated" like theirs,
# and its OOS read stays comparable. The brief itself proposes exactly this
# split under "OOS-val" ("... den slitna 2018-26-multi-asset-ytan").
HISTORY_START = "2003-01-01"     # warm-up buffer before IS_START (252d TSMOM
                                  # lookback + 252d event-threshold window)
IS_START = "2004-01-01"
IS_END = "2017-12-31"
OOS_START = "2018-01-01"
OOS_END = None                    # None = through the latest fetched bar

TRADING_DAYS_YEAR = 252

# ---------------------------------------------------------------------------
# Event definition -- "E_{i,t} = 1{|r_{i,t}| > rullande 252d-percentil q,
# PIT t-1}"
# ---------------------------------------------------------------------------
EVENT_LOOKBACK = 252              # rolling window for the percentile threshold
# PIT discipline (house convention, same as omori/data.py): the threshold at
# t is computed on returns.shift(1).rolling(EVENT_LOOKBACK), i.e. strictly
# t-252..t-1, never touching r_t itself.

# ---------------------------------------------------------------------------
# Frozen Omori kernel -- "w kalibreras en gaang IS med Efterskalvsklockans
# verifierade Omori-fitter paa poolade haendelser, sedan fryses"
# ---------------------------------------------------------------------------
# The exponent is Efterskalvsklockan's own frozen POOLED (population-level,
# cross-instrument) Omori prior -- research/omori/output/priors.json:"global"
# -- i.e. the one calibrated-once, frozen-before-OOS number in that repo that
# is literally a fit "on pooled events" rather than a per-instrument fit.
# Per-instrument priors exist in that same file but are explicitly NOT what
# the brief asks for here.
OMORI_KERNEL_P = 0.5758457103777312   # frozen, verbatim from omori/output/priors.json
# DECLARED: omori's own Omori-law offset "c" is profiled per-event over the
# grid {0, 1, 2} (omori/priors.py) -- no single frozen *pooled* c is reported
# anywhere in that repo (c is a per-fit nuisance parameter, not a population
# prior). We take the midpoint of that same profiling grid, c=1, as the one
# frozen value, and normalize w_s to sum to 1 over the 10-day kernel so that
# Lambda_t is a weighted average of recent X (EpiEstim/Cori serial-interval
# convention: the serial-interval distribution integrates to 1).
OMORI_KERNEL_C = 1.0
OMORI_KERNEL_LAGS = 10             # s = 1..10 in Lambda_t = sum_s w_s X_{t-s}
OMORI_KERNEL_ROBUSTNESS_PCT = 0.5  # +-50% perturbation of OMORI_KERNEL_P,
                                    # sensitivity check outside the main grid

# ---------------------------------------------------------------------------
# Cori R_t estimator and G_t tilt
# ---------------------------------------------------------------------------
# R_hat_t = (1 + sum_{u=t-tau+1..t} X_u) / (1 + sum_u Lambda_u), gamma(1,1) prior
TAU_GRID = (10, 21)
KAPPA_GRID = (1, 2)                # tilt exponent in G_t = clip((1/R_hat)^kappa, .3, 1.3)
Q_GRID = (90, 95)                  # event-threshold percentile
G_CLIP_LOW = 0.3
G_CLIP_HIGH = 1.3
REBALANCE_WEEKDAY = "FRI"          # G_t updated Friday, applied to next ISO week

# ---------------------------------------------------------------------------
# Base engine -- "repots dokumenterade TSMOM-proxy (Vindkastets levande
# komponent)": Vindkastet's own signal hypothesis (ridge-VAR/transient-growth)
# is DEAD (research/vindkastet/REPORT.md: "DOED"); the only TSMOM code in that
# branch is the academic 12m-sign / inverse-20d-vol proxy documented in its
# REPORT.md ("Substituerat med en enkel akademisk 12-manaders TSMOM-portfolj
# ... standardproxy i litteraturen (Moskowitz-Ooi-Pedersen-stil)"), used
# there only as an un-gated correlation benchmark. That signal, with
# portfolio-level vol targeting and a gross cap added on top (the brief's own
# parenthetical: "protokollregeln om portfoljniva-volskalning ligger alltsaa
# i basen, fore overlagget"), is what "the repo's documented TSMOM proxy" is
# taken to mean here.
TSMOM_LOOKBACK = 252               # 12-month sign momentum
TSMOM_VOL_LOOKBACK = 20            # inverse-vol sizing window, verbatim from
                                    # vindkastet/run_backtest_summary.py:trend_proxy_returns
PORTFOLIO_VOL_TARGET = 0.10        # 10% annualized ex-ante, per the brief
GROSS_CAP = 2.0                    # 200%, per the brief
VOL_TARGET_SOLVE_MAX_ITER = 25     # Dammluckan's iterative k-solve pattern
VOL_TARGET_SOLVE_TOL = 1e-4

# ---------------------------------------------------------------------------
# Base-engine gate -- "Steg 1 basmotor: TSMOM-sleeven sjaelv maaste ha
# netto-alfa paa IS-panelen (Oeglegrinden-regeln)"
# ---------------------------------------------------------------------------
# Oglegrinden's own literal wording (REPORT.md): "en grind paa doed alfa aer
# fortfarande doed" -- a gate on dead alpha is still dead. No numeric bar is
# named there beyond "positive net-of-cost Sharpe IS"; DECLARED: we use that
# qualitative bar literally, ann_sharpe(base, IS, net of costs) > 0.
BASE_ENGINE_MIN_IS_SHARPE = 0.0

# ---------------------------------------------------------------------------
# Costs -- "repots block-bootstrap och ADV-bucket-kostnader (ej Runradens
# platta 2 bp)". Ported verbatim from research/dammluckan/costs.py, which is
# itself ported verbatim from research/formdriften/costs.py -- the only
# ADV-bucketed cost model anywhere in this repo's research-branch series.
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
# DOCUMENTED SPREAD CAVEAT (verbatim convention, restated here as in every
# sibling that reuses this cost model): EODHD does not provide quoted
# bid/ask spread history, so the half-spread leg is approximated from
# trailing dollar-ADV via a monotone liquidity bucket -- a documented
# approximation, not a measured spread. Only the commission leg (3bp/side)
# is a clean, un-approximated assumption.

# ---------------------------------------------------------------------------
# Redundancy screen kill threshold ("R^2 > 0.5 eller inkrementell delta-R^2 /
# NW-t ~ 0 for framatblickande vol/drawdown -> doed"). DECLARED minimum-effect
# floor paired with the null-percentile comparison (Runraden's own post-mortem
# lesson: "K1a/K2 pass but clear a threshold that itself sits essentially at
# zero" -- a percentile-only gate is not enough).
REDUNDANCY_R2_KILL = 0.5
REDUNDANCY_MIN_DELTA_R2 = 0.01     # incremental R^2 must clear this AND be
                                    # NW-significant to count as "not dead"

# ---------------------------------------------------------------------------
# Dispersion / episode-count gate
# ---------------------------------------------------------------------------
EPISODE_MIN_RUN_DAYS = 5           # R_hat > 1 for >= 5 consecutive days
EPISODE_MIN_SEPARATION_DAYS = 21   # >= 21 days between episodes to count distinct
EPISODE_MIN_COUNT_IS = 10
BINDING_G_LOW_THRESHOLD = 0.7      # share of weeks with G < 0.7
BINDING_G_LOW_SHARE_RANGE = (0.05, 0.30)

# ---------------------------------------------------------------------------
# Step 2 acceptance bar
# ---------------------------------------------------------------------------
STEP2_MIN_SR_INCREMENT = 0.10
STEP2_MIN_SIGN_CONSISTENCY = 0.60
N_SUBPERIODS = 2                   # "teckenkonsistens mellan IS-halvor" -- 2 halves
DSR_OOS_KILL = 0.0
ORACLE_CAP_MIN_SR_INCREMENT = 0.15  # "ger den < +0.15 netto-SR over basen ar
                                     # hela reglageklassen doed" -- cheapest,
                                     # first gate

# "DSR >= 0 inklusive poolad korrektion for ytans tidigare avlasningar":
# DECLARED count of prior strategies in this repo's research-branch series
# that already spent selection budget on this same general EODHD US-ETF
# panel / IS date range: Dammluckan, Efterskalvsklockan (omori), Vindkastet,
# Runraden, Oglegrinden = 5 prior reads. Total effective reads including
# Smittotalet itself = 6. Used to inflate the DSR trial-pool SIZE (not its
# spread) when computing the pooled-surface-corrected DSR, by tiling the
# grid's own Sharpe trials -- tiling preserves std(sr_trials) exactly while
# raising n, which is what expected_max_sharpe's order-statistics
# approximation actually consumes.
N_PRIOR_SURFACE_READS = 5
N_EFFECTIVE_SURFACE_READS = N_PRIOR_SURFACE_READS + 1

# ---------------------------------------------------------------------------
# EODHD access -- env var convention pinned by the eodhd-api-key-setup branch,
# HTTP client ported from research/runraden/eodhd_client.py.
# ---------------------------------------------------------------------------
EODHD_API_KEY_ENV = "EODHD_API_KEY"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_THIS_DIR, "data", "raw")
DATA_DIR = os.path.join(_THIS_DIR, "data")
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")

FIELDS = ("open", "high", "low", "close", "adjusted_close", "volume")


@dataclass(frozen=True)
class GridCell:
    q: int
    tau: int
    kappa: int


GRID = tuple(
    GridCell(q=q, tau=tau, kappa=kappa)
    for q in Q_GRID
    for tau in TAU_GRID
    for kappa in KAPPA_GRID
)  # 2 x 2 x 2 = 8 cells

DEFAULT_CELL = GridCell(q=95, tau=21, kappa=1)
