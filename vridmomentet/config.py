"""Central, pure-declaration configuration for Vridmomentet.

Nothing here touches data or math; every numeric knob other modules use is
declared exactly once here so signal.py / portfolio.py / backtest.py /
grid.py / stats.py / run.py all read the same numbers instead of
re-declaring them (same rationale, and largely the same shape, as
fasflocken/config.py in a sibling branch of this repo).

The source brief ("Vridmomentet") is explicit about the signal construction,
the portfolio rules, the null-hypothesis baseline, and the neighborhood grid.
It is *not* explicit about exact backtest cost assumptions or the numeric
rejection thresholds for the falsification suite -- the brief text was cut
off mid-sentence at "Backtestskiss (trestegsraket per Oglegrindens
protokoll)" before any of that was written down. Every constant below that
is not directly traceable to an explicit brief rule is marked DECLARED and
chosen to match the convention already established by Oglegrinden (the named
reference protocol) and its sibling strategy branches in this repository.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Universe (brief: "Punkt-i-tid S&P 500 + 400 ... pris > 5 USD, ADV20 > 20 MUSD")
# --------------------------------------------------------------------------

SP500_INDEX_TICKER = "GSPC.INDX"
SP400_INDEX_TICKER = "MID.INDX"

PRICE_MIN_USD = 5.0
ADV_MIN_USD = 20_000_000.0
ADV_LOOKBACK_DAYS = 20  # brief: "ADV20"

# DECLARED: history window. Point-in-time S&P 500 membership from EODHD's
# HistoricalTickerComponents is usable from ~2000 onward; we start a few
# years later to give the 60d ADV/vol windows and 20d signal window a clean
# burn-in before the first tradeable Friday.
HISTORY_START = _dt.date(2003, 1, 1)

# DECLARED: IS/OOS split point. Reused verbatim from Oglegrinden's own
# split (oglegrinden/run.py OOS_START) so results across this research
# program's strategies sit on a common IS/OOS boundary.
OOS_START = _dt.date(2018, 1, 1)

# DECLARED: three named sub-periods for the sign-stability check, chosen to
# each contain a different macro/vol regime within the in-sample span.
SUBPERIODS: dict[str, tuple[_dt.date, _dt.date]] = {
    "2003-08": (_dt.date(2003, 1, 1), _dt.date(2008, 12, 31)),
    "2009-13": (_dt.date(2009, 1, 1), _dt.date(2013, 12, 31)),
    "2014-17": (_dt.date(2014, 1, 1), _dt.date(2017, 12, 31)),
}


# --------------------------------------------------------------------------
# Signal (brief: u_s, P/V paths, Levy area, cross-sectional signal)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalParams:
    window_days: int = 20          # brief: "n = 20 d"
    adv_norm_days: int = 60        # brief: "ADV60" in u_s = sign(r_s)*DV_s/ADV60
    forward_horizon_days: int = 5  # brief: "E[r_{+5d} | s]"
    winsor_lo: float = 0.01        # brief: "winsoriserad 1/99 %"
    winsor_hi: float = 0.99
    return_transform: str = "sign"  # "sign" (brief default) or "tanh" (neighborhood variant)
    tanh_scale_days: int = 20       # sigma lookback for tanh(R/sigma) variant


DEFAULT_SIGNAL_PARAMS = SignalParams()

# brief: "Grannskap att testa: n in {10, 15, 20, 30, 40}, decilvariant,
# tanh(R/sigma) i stallet for sign(R)"
WINDOW_GRID = (10, 15, 20, 30, 40)
BUCKET_GRID = ("quintile", "decile")
TRANSFORM_GRID = ("sign", "tanh")


# --------------------------------------------------------------------------
# Portfolio (brief: quintile long/short, inverse-vol sizing, caps)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioParams:
    n_buckets: int = 5             # brief: "topp-kvintil" / "botten-kvintil" -> 5 buckets
    sizing_vol_days: int = 60      # brief: "w_i propto 1/sigma_i(60d)"
    gross_per_leg: float = 0.50    # brief: "50 % brutto per ben"
    per_name_cap: float = 0.02     # brief: "tak 2 % per namn"
    min_names_per_leg: int = 5     # DECLARED: floor below which a weekly slice is too thin to trade


DEFAULT_PORTFOLIO_PARAMS = PortfolioParams()


# --------------------------------------------------------------------------
# Execution timing (brief: Friday close signal; Monday close primary,
# Monday open variant; full weekly replacement; no stops/discretion).
# The Friday decision date itself is derived structurally in
# backtest.py::weekly_decision_dates ("last trading day of the ISO week",
# holiday-robust) rather than checked against an explicit weekday number.
# --------------------------------------------------------------------------

EXECUTION_VARIANTS = ("monday_close", "monday_open")
PRIMARY_EXECUTION = "monday_close"


# --------------------------------------------------------------------------
# Costs -- DECLARED. The brief does not pin down a spread assumption (its
# own text is cut off before the backtest-cost section). Individual
# ADV20>$20M large/mid caps trade with materially wider spreads than the
# sector ETFs Oglegrinden traded, so we do not reuse Oglegrinden's 1bp
# half-spread; we pick a documented, conservative default instead.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    commission_bps_per_side: float = 2.0
    half_spread_bps: float = 5.0

    def one_way_bps(self) -> float:
        """Cost per $1 of ONE-WAY trading (executing a single buy or sell).
        This is the rate to multiply against `portfolio.turnover()`, which
        already sums both legs of a trade (sum(|new_w - prev_w|) counts a
        name's exit AND another name's entry separately) -- multiplying an
        already-both-sides turnover figure by a round-trip rate would
        double-count every dollar actually traded.
        """
        return self.commission_bps_per_side + self.half_spread_bps

    def round_trip_bps(self) -> float:
        """Cost of a complete buy-then-later-sell round trip on a SINGLE
        fixed $1 position (i.e. one_way_bps() charged twice). Provided for
        reference/alternate turnover conventions; NOT the rate to use
        against portfolio.turnover()'s sum-of-both-legs figure -- see
        one_way_bps().
        """
        return 2.0 * self.one_way_bps()


DEFAULT_COSTS = CostModel()


# --------------------------------------------------------------------------
# Null hypothesis (brief: "Huvudnull: shuffla dagordningen inom varje
# fonster. E[A]=0 exakt under utbytbarhet ... 1000 rep, aven blockvis")
# --------------------------------------------------------------------------

N_SHUFFLE_REPS = 1000
SHUFFLE_BLOCK_SIZES = (1, 4)  # 1 = fully iid permutation; 4 = ~one-week blocks, "aven blockvis"

# Portfolio-level stationary block bootstrap of weekly strategy returns
# (brief lists `arch` (block-bootstrap) under Bibliotek; mechanics follow
# Oglegrinden's stats.py stationary bootstrap, Politis & Romano 1994).
N_BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_BLOCK_WEEKS = 8.0


# --------------------------------------------------------------------------
# Active twins (brief: "ren momentum; momentum x turnover-niva
# (Lee-Swaminathan); lag-1-korskorrelation u->r")
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TwinParams:
    momentum_lookback_days: int = 20         # same formation length as the primary signal
    turnover_lookback_days: int = 60         # Lee & Swaminathan (2000) turnover window
    lag1_lookback_days: int = 20


DEFAULT_TWIN_PARAMS = TwinParams()


# --------------------------------------------------------------------------
# Rejection criteria -- DECLARED thresholds (brief's own falsification
# section is cut off before stating numbers). Chosen to match the
# convention of the sibling branches (fasflocken's p<0.10 / delta-Sharpe
# 0.15; Oglegrinden's DSR-z<=0 / concentration>0.5 / beats-all-twins).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RejectionThresholds:
    dsr_z_min: float = 0.0                 # DSR z-statistic must exceed this OOS
    bootstrap_p_max: float = 0.10          # block-bootstrap p-value must be below this
    shuffle_p_max: float = 0.10            # stage-1 within-window shuffle null p-value
    pnl_concentration_max: float = 0.5     # max share of total PnL from one 8-week window
    pead_delta_sharpe_min: float = 0.0     # must survive with earnings-spike weeks excluded
    ic_t_stat_min: float = 2.0             # stage-1 kill gate: |t(mean weekly IC)| must clear this


DEFAULT_REJECTION = RejectionThresholds()

PNL_CONCENTRATION_WINDOW_WEEKS = 8

WEEKS_PER_YEAR = 52
