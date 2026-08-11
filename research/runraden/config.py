"""Universe, hyperparameter and path configuration for the Runraden study.

Runraden ("the rune row") tests whether the *order* of daily sign returns
within a trading week carries information beyond the additive day-position
effects -- see research/runraden/README.md for the full write-up of the
hypothesis, methodology and results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(RESEARCH_DIR, "data_cache")
RESULTS_DIR = os.path.join(RESEARCH_DIR, "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EODHD_API_KEY_ENV = "EODHD_API_KEY"

# ---------------------------------------------------------------------------
# IS universe: 40 liquid US-listed ETFs spanning the major asset-class
# buckets used throughout the panel. This is the "40-ETF-EODHD-panelen":
# mined freely, never held out as an OOS surface for this or any other
# hypothesis in this line of work.
# ---------------------------------------------------------------------------
IS_UNIVERSE = {
    "broad_equity": ["SPY.US", "QQQ.US", "IWM.US", "DIA.US", "MDY.US"],
    "sector_equity": [
        "XLK.US", "XLF.US", "XLE.US", "XLV.US", "XLI.US", "XLY.US",
        "XLP.US", "XLB.US", "XLU.US", "XLRE.US", "XLC.US",
    ],
    "intl_equity": ["EFA.US", "EEM.US", "VGK.US", "EWJ.US", "FXI.US"],
    "rates_credit": [
        "TLT.US", "IEF.US", "SHY.US", "LQD.US", "HYG.US", "TIP.US",
        "BND.US", "AGG.US",
    ],
    "commodities": ["GLD.US", "SLV.US", "USO.US", "DBC.US", "UNG.US"],
    "real_estate": ["VNQ.US"],
    "currency": ["UUP.US", "FXE.US"],
    "other_equity": ["SMH.US", "ITB.US", "KRE.US"],
}


def is_universe_flat() -> list[str]:
    tickers: list[str] = []
    for bucket in IS_UNIVERSE.values():
        tickers.extend(bucket)
    return tickers


assert len(is_universe_flat()) == 40, "IS universe must contain exactly 40 tickers"
assert len(set(is_universe_flat())) == 40, "IS universe must not contain duplicates"

# ---------------------------------------------------------------------------
# OOS universe: ~20-25 European UCITS ETFs on XETRA/LSE, matched to the same
# asset-class mix as the IS panel. Candidates were pulled from EODHD's
# XETRA/LSE symbol lists (2026-08-11) and restricted to flagship "Core" /
# largest-AUM share classes per exposure as a liquidity (ADV) proxy -- see
# README "OOS universe" section for the full derivation and caveats.
#
# This universe is LOCKED here (before Step 2 is ever run) but its price
# history is NOT fetched or analysed unless the IS kill-criteria (K1, K2)
# are survived on the IS panel -- see pipeline.py / README "Execution log".
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


def oos_universe_flat() -> list[str]:
    tickers: list[str] = []
    for bucket in OOS_UNIVERSE.values():
        tickers.extend(bucket)
    return tickers


assert 20 <= len(oos_universe_flat()) <= 25, "OOS universe must have 20-25 tickers"
assert len(set(oos_universe_flat())) == len(oos_universe_flat())

# ---------------------------------------------------------------------------
# Estimation / hyperparameters
# ---------------------------------------------------------------------------
BURN_IN_YEARS = 3
REFIT_FREQ = "M"  # monthly refit of the expanding-window additive model
EXECUTION_LAG_WEEKS = 1  # signal at Friday close of week t, executed Monday close of t+1

# Primary (pre-registered) cell of the Step-2 grid.
PRIMARY_KAPPA = 300
PRIMARY_VOL_WINDOW = 60
PRIMARY_NO_TRADE_BAND = 0.15

GRID_KAPPA = [100, 300, 1000]
GRID_VOL_WINDOW = [20, 40, 60]
GRID_NO_TRADE_BAND = [0.0, 0.15, 0.30]

TARGET_GROSS_VOL = 0.08
MAX_GROSS = 2.00           # 200% gross cap
MAX_GROSS_PER_NAME = 0.10  # 10% gross per name

# Costs (Step 2 / K3)
ONE_WAY_COST_BP = 2.0          # midpoint of 1-3bp stated in the hypothesis
COST_STRESS_MULTIPLIER = 1.5

# DSR effective-trials accounting: 27 grid cells + 4 twins + k prior reads (0).
N_GRID_CELLS = len(GRID_KAPPA) * len(GRID_VOL_WINDOW) * len(GRID_NO_TRADE_BAND)
N_TWINS = 4
K_PRIOR_READS = 0
DSR_N_TRIALS = N_GRID_CELLS + N_TWINS + K_PRIOR_READS

RANDOM_SEED = 20260811
