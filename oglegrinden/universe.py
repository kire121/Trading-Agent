"""ETF universe definition for Oglegrinden.

~24 US sector / industry ETFs as specified in the strategy brief, plus SPY
as the market-beta hedge/reference instrument. Inception dates are NOT
hardcoded here -- they are read from each ticker's own price history
(first available trading day) in `data.py`, so the point-in-time universe
filter reflects actual data availability rather than a fixed table that
could go stale or be wrong.
"""

SECTOR_ETFS = [
    # SPDR sector funds (Select Sector SPDRs)
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE",
    # Industry / thematic ETFs
    "SMH", "XBI", "KRE", "XOP", "OIH", "XHB", "XRT", "ITA", "IYT", "GDX",
    "IYR", "KBE", "XME",
]

BENCHMARK = "SPY"

ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

# Minimum average-dollar-volume (ADV) filter, USD, per the strategy brief.
MIN_ADV_USD = 20_000_000

# Minimum number of live, ADV-eligible names required before the topology
# signal is considered meaningful. Below this the correlation cloud is too
# thin to support a non-trivial simplicial complex.
MIN_UNIVERSE_SIZE = 15
