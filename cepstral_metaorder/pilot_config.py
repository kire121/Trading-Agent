"""
Pilot-scope constants. The spec calls for a point-in-time top-1000 US
universe over 1-minute bars from 2012-2025. Fetching and processing that (an
estimated ~1B+ minute-bars) is far outside what a single interactive coding
session can pull through a REST API and process -- see README.md for the
budget math. This file defines the deliberately reduced real-data pilot:

  - Universe: a fixed, curated pool of 78 large/liquid US common stocks
    spanning 11 sectors, standing in for "apply the real PIT filter to a
    liquid-name candidate pool" rather than screening the full market. Nearly
    all of these will pass price>$5 and ADV(63d)>$25M on nearly every day,
    so this pilot mainly exercises the SIGNAL and VALIDATION machinery, not
    the universe churn that matters at full 1000-name scope.
  - Period: ~7 months of real 1-minute bars (with a 21-session detrend
    warm-up before the scored window), chosen for session time/API budget,
    not cherry-picked after seeing results -- it was fixed before any
    fetch or backtest ran.

Every function this file feeds (universe.py, signal.py, validation.py, ...)
takes the candidate list and date range as plain arguments and has no
knowledge that they're reduced -- point them at eodhd_client's full US
common-stock tape and a 2012-2025 range and they do the real thing.
"""

import datetime as dt

PILOT_SYMBOLS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO",
    "INTC", "TXN", "QCOM", "IBM", "NOW", "INTU",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MDLZ", "CL",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "SPGI",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "HON", "UPS", "CAT", "BA", "GE", "RTX", "LMT", "DE",
    # Materials
    "LIN", "APD",
    # Utilities
    "NEE", "DUK",
    # Real Estate
    "PLD", "AMT",
]

TSMOM_BASKET = ["SPY", "TLT", "GLD", "DBC", "UUP"]
MARKET_PROXY = "SPY"

# Intraday 1-minute fetch window. Detrend warm-up needs 21 sessions before
# the first SCORED day, so the effective scored window starts ~1 calendar
# month after INTRADAY_START.
INTRADAY_START = dt.date(2024, 11, 1)
INTRADAY_END = dt.date(2025, 6, 30)

# EOD fetch window: needs 63 trading days (~3 months) before INTRADAY_START
# for ADV/vol controls, and TSMOM's basket needs ~252 trading days (~1 year)
# trailing return, hence starting well over a year before INTRADAY_START.
EOD_START = "2023-06-01"
EOD_END = "2025-07-05"
